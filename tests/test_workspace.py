"""Continuity and evidence integrity across independently finalized research runs."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepmath.engine import run
from deepmath.ledger import canonical, verify_run
from deepmath.models import Problem, RunConfig
from deepmath.providers import ProviderError
from provider_fixture import FixtureProvider
from deepmath.qk import DisabledQKVerifier
from deepmath.workspace import ResearchWorkspace, source_directory


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = ResearchWorkspace(self.root / "research")
        self.workspace.initialize()
        self.problem = Problem.from_dict({"id": "lrx-4", "title": "A finite LRX question", "statement": "Examine the LRX diameter for n=4.",
                                          "graph": {"family": "lrx", "n": 4}, "objective": "diameter", "expected_diameter": 6})

    def make_run(self, directory="run", provider=None):
        destination = self.root / directory
        run(self.problem, RunConfig(rounds=1, top_k=3), provider or FixtureProvider(), DisabledQKVerifier(), destination)
        return destination

    def test_persists_full_evidence_and_returns_same_problem_memory(self):
        destination = self.make_run()
        (destination / "unlisted.txt").write_text("excluded")
        task = self.workspace.publish_run(destination)
        reopened = ResearchWorkspace(self.workspace.root)
        self.assertEqual(reopened.initialize()["protocol"], "AutoMathLab")
        memory = reopened.context(self.problem)
        self.assertEqual(memory["matching_runs"], 1)
        previous = memory["previous_runs"][0]
        self.assertEqual(previous["task_id"], task["id"])
        self.assertEqual(previous["claim_status"], "open")
        self.assertEqual(len(previous["ideas"]), 3)
        self.assertTrue(any(item["result"].get("diameter") == 6 for item in previous["observations"]))
        self.assertTrue(memory["open_proof_obligations"])
        self.assertTrue(memory["next_steps"])
        self.assertIn("critic", memory["role_protocol"]["role_contracts"])
        project = reopened.root / task["path"]
        archive = project / "experiments" / task["id"]
        self.assertTrue(verify_run(archive)["valid"])
        self.assertFalse((archive / "unlisted.txt").exists())
        for kind in ("hypotheses", "results", "reviews", "handoffs"):
            self.assertTrue((project / kind / f"{task['id']}.md").is_file())
            self.assertTrue((project / kind / f"{task['id']}.json").is_file())
        review = json.loads((project / "reviews" / f"{task['id']}.json").read_text())
        self.assertEqual(review["verification_status"], "not-reviewed")

    def test_exact_claim_and_graph_scope_prevents_memory_contamination(self):
        self.workspace.publish_run(self.make_run())
        for key, value in (("statement", "Prove all n, not merely n=4."), ("graph", {"family": "lx", "n": 4}),
                           ("expected_diameter", 7), ("target", (1, 0, 3, 2))):
            changed = {**self.problem.to_dict(), key: value}
            with self.subTest(key=key):
                memory = self.workspace.context(changed)
                self.assertEqual(memory["previous_runs"], [])
                self.assertEqual(memory["open_proof_obligations"], [])

    def test_idempotent_publication_and_colliding_run_names(self):
        first = self.make_run("first/run")
        second = self.make_run("second/run")
        task = self.workspace.publish_run(first)
        self.assertEqual(task, self.workspace.publish_run(first))
        other = self.workspace.publish_run(second)
        self.assertNotEqual(task["id"], other["id"])
        self.assertEqual(self.workspace.context(self.problem)["matching_runs"], 2)
        registry = json.loads((self.workspace.root / "registry.json").read_text())
        self.assertEqual(registry["revision"], 2)

    def test_failed_run_preserves_observations_critic_errors_and_handoff(self):
        provider = FixtureProvider()
        original = provider.complete

        def complete(role, system, payload):
            if role == "synthesizer":
                raise ProviderError("failed after critique")
            return original(role, system, payload)

        with patch.object(provider, "complete", side_effect=complete):
            with self.assertRaises(ProviderError):
                self.make_run(provider=provider)
        task = self.workspace.publish_run(self.root / "run")
        self.assertEqual(task["status"], "blocked")
        memory = self.workspace.context(self.problem)
        previous = memory["previous_runs"][0]
        self.assertEqual(previous["status"], "failed")
        self.assertEqual(previous["errors"][-1]["message"], "failed after critique")
        self.assertTrue(previous["observations"])
        self.assertTrue(previous["critiques"])
        self.assertTrue(memory["open_proof_obligations"])
        handoff = self.workspace.root / task["path"] / "handoffs" / f"{task['id']}.md"
        self.assertIn("failed after critique", handoff.read_text())

    def test_tampered_archive_and_protocol_are_not_fed_to_agents(self):
        task = self.workspace.publish_run(self.make_run())
        archive = self.workspace.root / task["path"] / "experiments" / task["id"]
        (archive / "report.md").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.workspace.context(self.problem)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.workspace.publish_run(self.root / "run")
        (self.workspace.root / "protocol" / "roles.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "protocol digest mismatch"):
            self.workspace.initialize()

    def test_protocol_contains_exact_upstream_templates(self):
        source = source_directory()
        for kind in ("hypothesis", "experiment", "review", "result", "handoff"):
            relative = Path("templates") / f"{kind}.md"
            self.assertEqual((source / relative).read_bytes(), (self.workspace.root / "protocol" / relative).read_bytes())
        self.assertEqual((source / "AGENTS.md").read_bytes(), (self.workspace.root / "protocol" / "AGENTS.md").read_bytes())

    def test_context_has_hard_size_limit_and_keeps_complete_archive(self):
        provider = FixtureProvider()
        original = provider.complete

        def complete(role, system, payload):
            response = original(role, system, payload)
            if role == "synthesizer":
                response.data["proof_obligations"] = [str(index) + "π" * 3990 for index in range(30)]
                response.data["next_steps"] = [str(index) + "π" * 3990 for index in range(30)]
            return response

        with patch.object(provider, "complete", side_effect=complete):
            task = self.workspace.publish_run(self.make_run(provider=provider))
        context = self.workspace.context(self.problem)
        self.assertTrue(context["truncated"])
        self.assertLessEqual(len(canonical(context)), 64000)
        original_summary = self.workspace.root / task["path"] / "experiments" / task["id"] / "summary.json"
        self.assertEqual(len(json.loads(original_summary.read_text())["synthesis"]["proof_obligations"]), 30)

    def test_never_overwrites_unrelated_directory_or_accepts_symlink_project(self):
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        (unrelated / "notes.txt").write_text("research")
        with self.assertRaises(FileExistsError):
            ResearchWorkspace(unrelated).initialize()
        projects = self.workspace.root / "projects"
        projects.rmdir()
        projects.symlink_to(unrelated, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the workspace"):
            self.workspace.publish_run(self.make_run())
        self.assertEqual(list(unrelated.iterdir()), [unrelated / "notes.txt"])

    def test_simultaneous_writer_is_rejected_without_losing_registry(self):
        destination = self.make_run()
        lock = self.workspace.root / ".write-lock"
        lock.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Another workspace writer"):
            self.workspace.publish_run(destination)
        self.assertEqual(self.workspace.context(self.problem)["matching_runs"], 0)
        lock.rmdir()
        self.workspace.publish_run(destination)
        self.assertEqual(self.workspace.context(self.problem)["matching_runs"], 1)


if __name__ == "__main__":
    unittest.main()
