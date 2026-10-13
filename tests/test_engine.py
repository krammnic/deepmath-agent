import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cayleypy.research.cli import default_problem, main
from cayleypy.research.engine import run, select_ideas
from cayleypy.research.ledger import export_research, verify_run
from cayleypy.research.models import Experiment, Idea, Problem, RunConfig
from provider_fixture import FixtureProvider
from cayleypy.research.qk import DisabledQKVerifier, QKScore, QKUnavailableError


class RecordingProvider(FixtureProvider):
    def __init__(self):
        self.payloads = []

    def complete(self, role, system, payload):
        self.payloads.append((role, payload))
        return super().complete(role, system, payload)


class ResearchTests(unittest.TestCase):
    def test_complete_loop_feedback_integrity_replay_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            provider = RecordingProvider()
            result = run(default_problem(5), RunConfig(rounds=2, top_k=3), provider, DisabledQKVerifier(), root)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["claim_status"], "open")
            self.assertEqual(len(provider.payloads), 10)
            second = next(payload for role, payload in provider.payloads if role == "theorist" and payload.get("round") == 2)
            self.assertIn("observations", second["feedback"])
            observations = result["rounds"][0]["observations"]
            self.assertTrue(any(r["result"].get("diameter") == 10 for r in observations))
            self.assertTrue(all(score["margin"] is None for score in result["rounds"][0]["scores"]))
            self.assertTrue(verify_run(root)["valid"])
            self.assertEqual(main(["replay", str(root)]), 0)
            exported = export_research(root, Path(tmp) / "export")
            self.assertTrue((exported / "experiments" / "run" / "summary.json").is_file())
            with (root / "report.md").open("a") as stream:
                stream.write("tampered")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_run(root)

    def test_qk_required_fails_before_provider_is_called(self):
        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(QKUnavailableError):
                run(default_problem(4), RunConfig(require_qk=True), provider, DisabledQKVerifier(), Path(tmp) / "absent")
            self.assertEqual(provider.payloads, [])

    def test_qk_selection_keeps_exploration_and_does_not_prove(self):
        problem = default_problem(4)
        ideas = [Idea.from_dict({"title": str(i), "claim": "all n", "rationale": "try", "experiments": [{"kind": "diameter"}]},
                               idea_id=str(i), role="theorist", problem=problem) for i in range(3)]
        scores = [QKScore(0.9, "correct", "scored"), QKScore(-0.9, "incorrect", "scored"), QKScore(0.5, "correct", "scored")]
        chosen = select_ideas(ideas, scores, 2, 0)
        self.assertEqual([idea.id for idea in chosen], ["0", "1"])
        self.assertFalse(scores[0].is_proof)

    def test_refuted_finite_diameter_does_not_become_universal_proof(self):
        problem = default_problem(4).to_dict()
        problem["expected_diameter"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            result = run(Problem.from_dict(problem), RunConfig(rounds=1, top_k=3), FixtureProvider(), DisabledQKVerifier(), Path(tmp) / "run")
            observations = result["rounds"][0]["observations"]
            self.assertTrue(any(x["result"].get("expected_diameter_check") == "refuted_at_this_n" for x in observations))
            self.assertEqual(result["claim_status"], "open")

    def test_untrusted_tool_plans_rejected(self):
        problem = default_problem(4)
        for data in [{"kind": "shell", "command": "echo unsafe"}, {"kind": "path", "target": [0, 1, 1, 3]},
                     {"kind": "word", "target": [0, 1, 2, 3], "word": "LXXX"}, {"kind": "diameter", "n": 20}]:
            with self.assertRaises(ValueError):
                Experiment.from_dict(data, problem)

    def test_budget_exhaustion_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run(default_problem(6), RunConfig(rounds=1, top_k=3, max_states=4, max_depth=2),
                         FixtureProvider(), DisabledQKVerifier(), Path(tmp) / "run")
            diameter = next(x["result"] for x in result["rounds"][0]["observations"] if x["experiment"]["kind"] == "diameter")
            self.assertEqual(diameter["expected_diameter_check"], "inconclusive")
            self.assertIsNone(diameter["diameter"])

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run(default_problem(4), RunConfig(), FixtureProvider(), DisabledQKVerifier(), Path(tmp))

    def test_replay_detects_changed_heuristic_upper_bound(self):
        old = {"status": "completed", "evidence": "verified_path_upper_bound", "path_verified": True,
               "path_length": 28, "exact": False, "complete": False, "diameter": None}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with patch("cayleypy.research.engine.execute", return_value=old):
                run(default_problem(4), RunConfig(rounds=1, top_k=1), FixtureProvider(), DisabledQKVerifier(), root)
            with patch("cayleypy.research.cli.execute", return_value={**old, "path_length": 30}):
                self.assertEqual(main(["replay", str(root)]), 2)


if __name__ == "__main__":
    unittest.main()
