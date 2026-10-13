import unittest
from unittest.mock import patch

from deepmath.cli import default_problem
from deepmath.experiments import execute, graph_for
from deepmath.models import Experiment, RunConfig
from deepmath.search import BackendUnavailable


class ExperimentDispatchTests(unittest.TestCase):
    def test_native_dispatch_uses_supported_options(self):
        with patch("deepmath.experiments.multigpu_search", side_effect=BackendUnavailable("hardware")) as call:
            with self.assertRaises(BackendUnavailable):
                execute(default_problem(4), Experiment("path", (1, 0, 3, 2)), RunConfig(backend="multigpu", num_gpus=2, seed=42))
        self.assertEqual(call.call_args.kwargs["num_gpus"], 2)
        self.assertEqual(call.call_args.kwargs["seed"], 42)
        self.assertNotIn("device", call.call_args.kwargs)

    def test_diameter_always_uses_exact_bfs(self):
        with patch("deepmath.experiments.multigpu_search") as call:
            result = execute(default_problem(4), Experiment("diameter"), RunConfig(backend="multigpu"))
        self.assertEqual(result["diameter"], 6)
        call.assert_not_called()

    def test_invalid_word_is_not_a_verified_upper_bound(self):
        result = execute(default_problem(4), Experiment("word", (1, 0, 3, 2), ("X",)), RunConfig())
        self.assertFalse(result["path_verified"])
        self.assertEqual(result["evidence"], "invalid_word")


if __name__ == "__main__":
    unittest.main()
