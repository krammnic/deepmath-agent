"""Research exports preserve native graph semantics and immutable evidence."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepmath.experiments import execute, graph_for
from deepmath.ledger import Ledger, export_automathlab, now, verify_run
from deepmath.models import Experiment, Problem, RunConfig


class GeneralGraphExportTests(unittest.TestCase):
    def export_problem(self, graph):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        problem = Problem.from_dict({"id": "export-native-graph", "title": "Exact finite graph",
                                    "statement": "Measure the diameter of the specified generated component.",
                                    "graph": graph, "objective": "diameter"})
        config = RunConfig(rounds=1, max_states=1000)
        result = execute(problem, Experiment("diameter"), config)
        self.assertTrue(result["complete"])
        self.assertIsNotNone(result["diameter"])
        inputs = {"problem": problem.to_dict(), "config": asdict(config), "created": now()}
        summary = {**inputs, "mode": "test_fixture", "status": "completed", "claim_status": "open",
                   "rounds": [{"round": 1, "observations": [{"idea_id": "finite-diameter",
                              "experiment": asdict(Experiment("diameter")), "result": result}]}]}
        source = base / "run"
        ledger = Ledger(source)
        ledger.save("input.json", inputs)
        ledger.event("experiment", summary["rounds"][0]["observations"][0])
        ledger.finalize(summary)
        # Export/recovery must use recorded native definitions, without loading
        # or reinterpreting an optional graph constructor during reproduction.
        with patch("deepmath.search._load_optional", side_effect=AssertionError("use the recorded graph definition")):
            exported = export_automathlab(source, base / "export") / "experiments" / "run"
            self.assertTrue(verify_run(exported)["valid"])
            for name in ("input.json", "summary.json", "manifest.json", "events.jsonl", "report.md"):
                self.assertEqual((source / name).read_bytes(), (exported / name).read_bytes())
            recovered = Problem.from_dict(json.loads((exported / "input.json").read_text())["problem"])
            self.assertEqual(graph_for(recovered).to_dict(), graph_for(problem).to_dict())
            replay = execute(recovered, Experiment("diameter"), config)
            self.assertEqual(replay["diameter"], result["diameter"])
            self.assertEqual(replay["graph"], result["graph"])
        return graph_for(recovered), result, recovered

    def test_matrix_export_preserves_modulus_action_and_nonidentity_root(self):
        graph, _, _ = self.export_problem({"family": "matrix", "modulo": 3,
            "generators": {"e": [[1, 1], [0, 1]], "f": [[1, 0], [1, 1]]},
            "central_state": [1, 1, 0, 1]})
        self.assertEqual(graph.action, "matrix")
        self.assertEqual(graph.modulo, 3)
        self.assertEqual(graph.identity, (1, 1, 0, 1))
        self.assertTrue(graph.vertex_transitive)

    def test_schreier_export_keeps_root_distinct_from_diameter_witness(self):
        graph, result, _ = self.export_problem({"family": "permutation", "n": 4,
            "generators": {"a": [1, 0, 2, 3], "b": [0, 2, 1, 3], "c": [0, 1, 3, 2]},
            "central_state": [0, 1, 0, 1]})
        self.assertFalse(graph.vertex_transitive)
        self.assertEqual(graph.identity, (0, 1, 0, 1))
        self.assertNotEqual(tuple(result["start"]), graph.identity)
        self.assertEqual(result["diameter"], 4)

    @unittest.skipUnless(importlib.util.find_spec("cayleypy"), "CayleyPy is not installed")
    def test_catalog_export_retains_frozen_constructor_and_custom_root(self):
        graph, _, problem = self.export_problem({"family": "cayleypy",
            "constructor": "MatrixGroups.special_linear_fundamental_roots",
            "kwargs": {"n": 2, "modulo": 3}, "central_state": [1, 1, 0, 1]})
        self.assertEqual(problem.graph["constructor"], "MatrixGroups.special_linear_fundamental_roots")
        self.assertEqual(problem.graph["resolved_definition"]["central_state"], [1, 1, 0, 1])
        self.assertEqual(graph.action, "matrix")
        self.assertEqual(graph.modulo, 3)
        self.assertEqual(graph.identity, (1, 1, 0, 1))


if __name__ == "__main__":
    unittest.main()
