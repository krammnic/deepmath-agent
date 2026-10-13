"""Verify the public commands carry research context across separate sessions."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cayleypy.research.cli import default_problem, main
from cayleypy.research.providers import ProviderError
from provider_fixture import FixtureProvider


class RecordingProvider(FixtureProvider):
    def __init__(self):
        self.requests = []

    def complete(self, role, system, payload):
        self.requests.append((role, payload))
        return super().complete(role, system, payload)


class ContinuingResearchTests(unittest.TestCase):
    def test_separate_cli_sessions_load_research_memory_and_register_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            workspace, problem, config = root / "research", root / "problem.json", root / "config.json"
            problem.write_text(json.dumps(default_problem(4).to_dict()))
            config.write_text(json.dumps({"run": {"rounds": 1, "top_k": 3},
                                          "llm": {"model": "test-fixture", "api_key_env": ""}}))
            self.assertEqual(main(["workspace", "init", str(workspace)]), 0)
            command = ["run", str(problem), "--config", str(config), "--workspace", str(workspace)]
            first = RecordingProvider()
            with patch("cayleypy.research.cli.ChatCompletionsProvider", return_value=first):
                self.assertEqual(main(command + ["--output", str(root / "first")]), 0)
            self.assertEqual(first.requests[0][1]["research_memory"]["matching_runs"], 0)

            second = RecordingProvider()
            with patch("cayleypy.research.cli.ChatCompletionsProvider", return_value=second):
                self.assertEqual(main(command + ["--output", str(root / "second")]), 0)
            prior = second.requests[0][1]["research_memory"]
            self.assertEqual(prior["matching_runs"], 1)
            self.assertTrue(prior["previous_runs"][0]["observations"])
            self.assertTrue(prior["open_proof_obligations"])
            for role, payload in second.requests:
                self.assertEqual(payload["research_memory"]["matching_runs"], 1, role)
                self.assertIn("feedback", payload, role)

            unrelated = root / "new-research"
            with patch("cayleypy.research.cli.ChatCompletionsProvider", return_value=RecordingProvider()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", str(problem), "--config", str(config),
                                       "--workspace", str(unrelated), "--output", str(root / "first")]), 2)
            self.assertFalse(unrelated.exists())

            failing = RecordingProvider()
            original = failing.complete
            def fail_critic(role, system, payload):
                if role == "critic":
                    raise ProviderError("test review unavailable")
                return original(role, system, payload)
            failing.complete = fail_critic
            with patch("cayleypy.research.cli.ChatCompletionsProvider", return_value=failing), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command + ["--output", str(root / "third")]), 2)
            registry = json.loads((workspace / "registry.json").read_text())
            self.assertEqual(len(registry["tasks"]), 3)
            self.assertEqual(registry["tasks"][-1]["run_status"], "failed")
            self.assertEqual(main(["workspace", "context", str(workspace), str(problem)]), 0)
            self.assertEqual(main(["workspace", "add", str(workspace), str(root / "first")]), 0)
            self.assertEqual(len(json.loads((workspace / "registry.json").read_text())["tasks"]), 3)


if __name__ == "__main__":
    unittest.main()
