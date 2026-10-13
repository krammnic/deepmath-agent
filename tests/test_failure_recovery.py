"""Failure recovery must preserve finite evidence without upgrading its status."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepmath.engine import run
from deepmath.experiments import execute
from deepmath.ledger import export_automathlab, verification_status, verify_run
from deepmath.models import Experiment, Problem, RunConfig
from deepmath.providers import ProviderError
from provider_fixture import FixtureProvider
from deepmath.qk import DisabledQKVerifier, QKUnavailableError


class FailingRoleProvider(FixtureProvider):
    def __init__(self, role):
        self.role = role

    def complete(self, role, system, payload):
        if role == self.role:
            raise ProviderError(f"simulated {role} failure")
        return super().complete(role, system, payload)


class FailureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.output = self.base / "run with spaces"
        self.problem = Problem.from_dict({
            "id": "lrx-test", "title": "LRX test", "statement": "Check one finite instance.",
            "graph": {"family": "lrx", "n": 4}, "objective": "diameter", "expected_diameter": 6,
        })
        self.config = RunConfig(rounds=1, top_k=2)

    def read_summary(self):
        return json.loads((self.output / "summary.json").read_text())

    def run_with(self, provider):
        return run(self.problem, self.config, provider, DisabledQKVerifier(), self.output)

    def test_critic_failure_keeps_completed_observations_and_failed_checkpoint(self):
        with self.assertRaisesRegex(ProviderError, "critic"):
            self.run_with(FailingRoleProvider("critic"))
        summary = self.read_summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["verification_status"], "author-checked")
        self.assertEqual(summary["error"]["stage"], "critic")
        self.assertEqual(len(summary["rounds"]), 1)
        observations = summary["rounds"][0]["observations"]
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(item["result"]["status"] == "completed" for item in observations))
        self.assertEqual(json.loads((self.output / "checkpoint.json").read_text()), summary)
        report = (self.output / "report.md").read_text()
        self.assertIn("Run status: failed", report)
        self.assertIn("simulated critic failure", report)
        self.assertIn("exact_finite_diameter", report)
        self.assertTrue(verify_run(self.output)["valid"])

    def test_synthesizer_failure_preserves_successful_critique(self):
        with self.assertRaisesRegex(ProviderError, "synthesizer"):
            self.run_with(FailingRoleProvider("synthesizer"))
        record = self.read_summary()["rounds"][0]
        self.assertIn("critique", record)
        self.assertNotIn("synthesis", record)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["observations"]), 2)

    def test_proposal_error_produces_partial_status_with_successful_evidence(self):
        summary = self.run_with(FailingRoleProvider("holographer"))
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["verification_status"], "author-checked")
        self.assertEqual(summary["rounds"][0]["status"], "completed_with_errors")
        self.assertEqual(summary["errors"][0]["role"], "holographer")

    def test_all_tool_errors_never_receive_author_checked_status(self):
        with patch("deepmath.engine.execute", side_effect=RuntimeError("search unavailable")):
            summary = self.run_with(FixtureProvider())
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["verification_status"], "not-reviewed")
        self.assertEqual(len(summary["errors"]), 2)
        self.assertIn("Verification: not-reviewed", (self.output / "report.md").read_text())
        project = export_automathlab(self.output, self.base / "export")
        exported_text = (project / "experiments" / "run-with-spaces.md").read_text()
        self.assertIn("Run status: completed_with_errors", exported_text)
        self.assertIn("Verification status: not-reviewed", exported_text)
        self.assertIn("search unavailable", exported_text)

    def test_missing_required_qk_fails_before_llm_but_keeps_artifacts(self):
        provider = FixtureProvider()
        with patch.object(provider, "complete", side_effect=AssertionError("LLM should not run")) as complete:
            with self.assertRaises(QKUnavailableError):
                run(self.problem, RunConfig(rounds=1, require_qk=True), provider,
                    DisabledQKVerifier(), self.output)
            complete.assert_not_called()
        summary = self.read_summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["verification_status"], "not-reviewed")
        self.assertEqual(summary["rounds"], [])
        self.assertTrue(verify_run(self.output)["valid"])

    def test_export_of_failed_run_copies_only_verified_files(self):
        with self.assertRaises(ProviderError):
            self.run_with(FailingRoleProvider("critic"))
        (self.output / "unlisted.txt").write_text("not part of this run")
        extra = self.base / "unrelated.txt"
        extra.write_text("unrelated local content")
        (self.output / "unlisted-link.txt").symlink_to(extra)
        project = export_automathlab(self.output, self.base / "export")
        artifacts = project / "experiments" / "run-with-spaces"
        expected = {entry["path_or_url"] for entry in json.loads((self.output / "manifest.json").read_text())["artifacts"]}
        self.assertEqual({path.name for path in artifacts.iterdir()}, expected | {"manifest.json"})
        self.assertTrue(verify_run(artifacts)["valid"])
        handoff = (project / "handoffs" / "run-with-spaces.md").read_text()
        self.assertIn("Run status: failed", handoff)
        self.assertIn("simulated critic failure", handoff)
        self.assertNotIn("run completed", handoff)

    def test_verification_status_requires_factual_result_flags(self):
        def summary(result):
            return {"verification_status": "author-checked", "rounds": [{"observations": [{"result": result}]}]}
        for result in [
            {"status": "error", "evidence": "exact_finite_diameter"},
            {"status": "completed", "evidence": "inconclusive"},
            {"status": "completed", "evidence": "invalid_word", "path_verified": False},
            {"status": "completed", "evidence": "exact_finite_diameter", "exact": True, "complete": False, "diameter": 6},
            {"status": "completed", "evidence": "verified_path_upper_bound", "path_verified": False},
            {"status": "completed", "evidence": "finite_heuristic_probe", "explored_states": 0, "search": {"exact": True}},
        ]:
            with self.subTest(result=result):
                self.assertEqual(verification_status(summary(result)), "not-reviewed")

    def test_completed_area_probe_is_finite_author_checked_evidence(self):
        probe = execute(self.problem, Experiment.from_dict({"kind": "area_probe"}, self.problem), self.config)
        with patch("deepmath.engine.execute", return_value=probe):
            summary = self.run_with(FixtureProvider())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["verification_status"], "author-checked")
        report = (self.output / "report.md").read_text()
        self.assertIn("lower-bound check=counterexample_found", report)
        self.assertIn("equality check=counterexample_found", report)
        self.assertIn("universal claim remains unestablished", report)

    def test_manifest_paths_cannot_reenter_root_using_parent_segments(self):
        self.run_with(FixtureProvider())
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        entry = dict(manifest["artifacts"][0])
        entry["path_or_url"] = f"../{self.output.name}/{entry['path_or_url']}"
        manifest["artifacts"].append(entry)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "plain filenames"):
            verify_run(self.output)


if __name__ == "__main__":
    unittest.main()
