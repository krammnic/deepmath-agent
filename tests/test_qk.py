"""Adapter contract tests; no model download, GPU, or simulated QK inference."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cayleypy.research.qk import DisabledQKVerifier, QKConfig, QKUnavailableError, QKVerifier


class QKAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "artifact.json"
        self.path.write_text('{"fixture": "adapter contract only"}')
        self.artifact = SimpleNamespace(
            model_id="reviewed/test-model",
            model_revision="immutable-revision",
            layer_start_fraction=0.5,
            validate=Mock(),
        )
        self.scorer = SimpleNamespace(rewards=Mock(return_value=[0.4, 0.0, -0.3]), close=Mock())
        self.backend = SimpleNamespace(
            ZeroShotQKArtifact=SimpleNamespace(load=Mock(return_value=self.artifact)),
            ZeroShotQKScorer=Mock(return_value=self.scorer),
        )

    def verifier(self):
        return QKVerifier(QKConfig(self.path, device="cpu", dtype="float32"))

    def test_initialization_and_empty_batch_are_lazy(self):
        with patch("cayleypy.research.qk.import_module") as importer:
            with self.verifier() as verifier:
                self.assertEqual(verifier.score("problem", []), [])
            importer.assert_not_called()

    def test_prepare_loads_backend_without_scoring(self):
        with patch("cayleypy.research.qk.import_module", return_value=self.backend):
            with self.verifier() as verifier:
                verifier.prepare()
                verifier.prepare()
        self.backend.ZeroShotQKScorer.assert_called_once()
        self.scorer.rewards.assert_not_called()
        self.scorer.close.assert_called_once()

    def test_scorer_receives_artifact_and_pairs(self):
        with patch("cayleypy.research.qk.import_module", return_value=self.backend) as importer:
            with self.verifier() as verifier:
                scores = verifier.score("problem", ["idea a", "idea b", "idea c"])
            importer.assert_called_once_with("cayleypy.qk")
        self.backend.ZeroShotQKArtifact.load.assert_called_once_with(self.path)
        self.artifact.validate.assert_called_once()
        self.backend.ZeroShotQKScorer.assert_called_once_with(
            model_id="reviewed/test-model", model_revision="immutable-revision",
            device="cpu", dtype="float32", max_length=4096,
            micro_batch_size=2, layer_start_fraction=0.5,
        )
        self.scorer.rewards.assert_called_once_with(
            ["problem"] * 3, ["idea a", "idea b", "idea c"], self.artifact,
        )
        self.assertEqual([score.margin for score in scores], [0.4, 0.0, -0.3])
        self.assertEqual([score.prediction for score in scores], ["correct", "incorrect", "incorrect"])
        for score in scores:
            self.assertFalse(score.to_dict()["is_proof"])
            self.assertEqual(score.artifact_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.scorer.close.assert_called_once()

    def test_multiple_batches_reuse_model_and_close_is_idempotent(self):
        self.scorer.rewards.return_value = [0.2]
        with patch("cayleypy.research.qk.import_module", return_value=self.backend):
            verifier = self.verifier()
            verifier.score_one("p", "a")
            verifier.score_one("p", "b")
            verifier.close()
            verifier.close()
        self.backend.ZeroShotQKScorer.assert_called_once()
        self.scorer.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            verifier.score("p", ["a"])

    def test_missing_artifact_does_not_import_backend(self):
        with patch("cayleypy.research.qk.import_module") as importer:
            verifier = QKVerifier(QKConfig(self.path.with_name("missing.json")))
            with self.assertRaisesRegex(QKUnavailableError, "cayleypy-research qk fit"):
                verifier.score_one("p", "a")
            importer.assert_not_called()

    def test_missing_dependency_has_actionable_error(self):
        with patch("cayleypy.research.qk.import_module", side_effect=ModuleNotFoundError("numpy")):
            with self.assertRaisesRegex(QKUnavailableError, "pip install"):
                self.verifier().score_one("p", "a")

    def test_lazy_inference_dependency_error_is_explicit(self):
        self.backend.ZeroShotQKScorer.side_effect = ImportError("torch")
        with patch("cayleypy.research.qk.import_module", return_value=self.backend):
            with self.assertRaisesRegex(QKUnavailableError, "torch"):
                self.verifier().score_one("p", "a")

    def test_invalid_artifact_does_not_initialize_model(self):
        self.artifact.validate.side_effect = ValueError("not calibration-free")
        with patch("cayleypy.research.qk.import_module", return_value=self.backend):
            with self.assertRaisesRegex(ValueError, "not calibration-free"):
                self.verifier().score_one("p", "a")
        self.backend.ZeroShotQKScorer.assert_not_called()

    def test_backend_failure_propagates_and_context_closes_hooks(self):
        self.scorer.rewards.side_effect = RuntimeError("out of memory")
        with patch("cayleypy.research.qk.import_module", return_value=self.backend):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                with self.verifier() as verifier:
                    verifier.score_one("p", "a")
        self.scorer.close.assert_called_once()

    def test_invalid_score_count_or_range_never_enters_results(self):
        for output in ([], [0.0, 0.1], [float("nan")], [float("inf")], [1.01], [-1.01]):
            with self.subTest(output=output):
                self.scorer.rewards.return_value = output
                with patch("cayleypy.research.qk.import_module", return_value=self.backend):
                    with self.verifier() as verifier:
                        with self.assertRaises(ValueError):
                            verifier.score_one("p", "a")

    def test_disabled_qk_is_unavailable_not_a_fabricated_score(self):
        with patch("cayleypy.research.qk.import_module") as importer:
            score = DisabledQKVerifier("No accelerator configured").score_one("p", "a")
            importer.assert_not_called()
        self.assertIsNone(score.margin)
        self.assertEqual(score.status, "unavailable")
        self.assertEqual(score.prediction, "unavailable")
        self.assertEqual(score.reason, "No accelerator configured")
        self.assertFalse(score.is_proof)

    def test_invalid_inputs_fail_before_loading(self):
        with patch("cayleypy.research.qk.import_module") as importer:
            for problem, candidates in [("", ["a"]), ("p", [""]), ("p", [None])]:
                with self.subTest(problem=problem, candidates=candidates):
                    with self.assertRaises(ValueError):
                        self.verifier().score(problem, candidates)
            with self.assertRaises(TypeError):
                self.verifier().score("p", "not a batch")
            importer.assert_not_called()

    def test_invalid_configuration_fails_without_inference(self):
        for kwargs in ({"dtype": "int8"}, {"max_length": 127}, {"max_length": 512.5},
                       {"micro_batch_size": 0}, {"micro_batch_size": True}, {"device": ""}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    QKConfig(self.path, **kwargs)


if __name__ == "__main__":
    unittest.main()
