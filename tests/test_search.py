"""Evidence-boundary tests; optional backend contracts are exercised on CPU."""

import importlib.util
import itertools
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cayleypy.research.search import (
    BackendUnavailable,
    GraphSpec,
    InvalidWitnessError,
    apply_move,
    cayleypy_search,
    exact_bfs,
    lrx_graph,
    multigpu_search,
    replay_path,
    verify_path,
)


class ExactSearchTests(unittest.TestCase):
    def test_lrx_two_coincident_generators(self):
        graph = lrx_graph(2)
        result = exact_bfs(graph)
        self.assertEqual(result.layer_sizes, (1, 1))
        self.assertEqual(result.diameter, 1)
        self.assertEqual(result.explored_states, 2)
        self.assertTrue(result.complete)
        self.assertTrue(result.path_verified)
        self.assertEqual(len(set(graph.generators.values())), 1)

    def test_exact_distances_against_independent_floyd_warshall(self):
        # An independent all-pairs implementation checks directed generators,
        # move orientation and shortest distances without calling BFS helpers.
        graph = GraphSpec("lx-4", {k: v for k, v in lrx_graph(4).generators.items() if k != "R"})
        states = list(itertools.permutations(range(4)))
        index = {state: i for i, state in enumerate(states)}
        matrix = [[100 if i != j else 0 for j in range(24)] for i in range(24)]
        for i, state in enumerate(states):
            for permutation in graph.generators.values():
                neighbor = tuple(state[k] for k in permutation)
                matrix[i][index[neighbor]] = min(matrix[i][index[neighbor]], 1)
        for middle in range(24):
            for i in range(24):
                for j in range(24):
                    matrix[i][j] = min(matrix[i][j], matrix[i][middle] + matrix[middle][j])
        result = exact_bfs(graph)
        self.assertTrue(result.complete)
        self.assertEqual(result.diameter, max(max(row) for row in matrix))
        for state, distance in result.distances.items():
            self.assertEqual(distance, matrix[index[graph.identity]][index[state]])

    def test_target_search_has_shortest_replayed_word_but_no_diameter(self):
        graph = lrx_graph(4)
        target = (3, 2, 1, 0)
        result = exact_bfs(graph, target=target)
        all_states = exact_bfs(graph)
        self.assertEqual(replay_path(graph, graph.identity, result.path), target)
        self.assertEqual(result.distance, all_states.distances[target])
        self.assertTrue(result.shortest_path)
        self.assertFalse(result.complete)
        self.assertIsNone(result.diameter)

    def test_state_budget_never_overflows_or_claims_diameter(self):
        result = exact_bfs(lrx_graph(4), max_states=7)
        self.assertEqual(result.explored_states, 7)
        self.assertEqual(sum(result.layer_sizes), 7)
        self.assertEqual(result.termination, "max_states")
        self.assertFalse(result.complete)
        self.assertIsNone(result.diameter)
        self.assertTrue(result.path_verified)

    def test_exact_state_budget_can_complete(self):
        result = exact_bfs(lrx_graph(3), max_states=6)
        self.assertTrue(result.complete)
        self.assertEqual(result.explored_states, 6)

    def test_depth_budget_closure_and_partial_search(self):
        graph = lrx_graph(4)
        full = exact_bfs(graph)
        partial = exact_bfs(graph, max_depth=full.diameter - 1)
        self.assertFalse(partial.complete)
        self.assertIsNone(partial.diameter)
        self.assertEqual(partial.termination, "max_depth")
        closure = exact_bfs(graph, max_depth=full.diameter)
        self.assertTrue(closure.complete)
        self.assertEqual(closure.diameter, full.diameter)

    def test_generated_subgroup_is_not_all_permutations(self):
        graph = GraphSpec("cyclic-3", {"L": (1, 2, 0)}, expected_order=3)
        result = exact_bfs(graph)
        self.assertEqual(result.diameter, 2)
        self.assertEqual(result.explored_states, 3)
        self.assertEqual(result.metadata["scope"], "generated_component")
        wrong_order = GraphSpec("wrong-cyclic-order", graph.generators, expected_order=6)
        with self.assertRaisesRegex(ValueError, "exhaustive component"):
            exact_bfs(wrong_order)

    def test_unreachable_target_has_no_path_even_on_complete_component(self):
        graph = GraphSpec("cyclic-3", {"L": (1, 2, 0)})
        result = exact_bfs(graph, target=(1, 0, 2))
        self.assertTrue(result.complete)
        self.assertFalse(result.path_found)
        self.assertFalse(result.path_verified)
        self.assertIsNone(result.distance)

    def test_identity_target_is_empty_path(self):
        graph = lrx_graph(3)
        result = exact_bfs(graph, target=graph.identity)
        self.assertEqual(result.path, ())
        self.assertTrue(result.path_verified)
        self.assertEqual(result.distance, 0)
        self.assertFalse(result.complete)

    def test_graph_and_state_validation(self):
        for n in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                lrx_graph(n)
        for generators in ({}, {"X": (0, 0)}, {"X": (False, 1)}, {"": (1, 0)}):
            with self.assertRaises(ValueError):
                GraphSpec("bad", generators)
        graph = lrx_graph(3)
        with self.assertRaises(ValueError):
            exact_bfs(graph, max_states=0)
        with self.assertRaises(ValueError):
            exact_bfs(graph, max_depth=-1)
        with self.assertRaises(ValueError):
            exact_bfs(graph, start=(0, 1, 1))
        self.assertFalse(verify_path(graph, graph.identity, ("NOPE",), graph.identity))
        self.assertFalse(verify_path(graph, graph.identity, (), (0, 0, 0)))

    def test_defensive_graph_copy_and_json_report(self):
        generators = {"L": [1, 2, 0]}
        graph = GraphSpec("cycle", generators)
        generators["L"][0] = 0
        self.assertEqual(apply_move(graph, graph.identity, "L"), (1, 2, 0))
        result = exact_bfs(graph)
        report = json.loads(json.dumps(result.to_dict(include_distances=True)))
        self.assertEqual(len(report["distances"]), 3)
        self.assertTrue(report["shortest_path"])


class BackendContractTests(unittest.TestCase):
    def fake_cayleypy(self, raw):
        calls = {}

        class Definition:
            @staticmethod
            def create(**kwargs):
                calls["definition"] = kwargs
                return kwargs

        class Graph:
            def __init__(self, definition, **kwargs):
                calls["graph"] = kwargs

            def beam_search(self, **kwargs):
                calls["beam"] = kwargs
                return raw

        return SimpleNamespace(CayleyGraphDef=Definition, CayleyGraph=Graph), calls

    def test_cayleypy_actual_contract_sets_target_as_center(self):
        raw = SimpleNamespace(path_found=True, path=[2], path_length=1)
        cp, calls = self.fake_cayleypy(raw)
        with patch("cayleypy.research.search.importlib.import_module", return_value=cp):
            result = cayleypy_search(lrx_graph(3), start=(0, 1, 2), target=(1, 0, 2), max_depth=5)
        self.assertEqual(calls["definition"]["central_state"], [1, 0, 2])
        self.assertEqual(calls["beam"]["max_steps"], 5)
        self.assertTrue(calls["beam"]["return_path"])
        self.assertEqual(result.path, ("X",))
        self.assertTrue(result.path_verified)
        self.assertFalse(result.exact)
        self.assertFalse(result.complete)
        self.assertFalse(result.shortest_path)
        self.assertIsNone(result.diameter)
        self.assertIsNone(result.explored_states)

    def test_beam_failure_provides_no_distance_claim(self):
        cp, _ = self.fake_cayleypy(SimpleNamespace(path_found=False, path=None, path_length=0))
        with patch("cayleypy.research.search.importlib.import_module", return_value=cp):
            result = cayleypy_search(lrx_graph(3), start=(1, 0, 2), max_depth=0)
        self.assertIsNone(result.distance)
        self.assertIsNone(result.diameter)
        self.assertFalse(result.path_verified)
        self.assertEqual(result.termination, "not_found_within_budget")
        self.assertFalse(result.exact)

    def test_invalid_backend_paths_are_rejected(self):
        for raw in (
            SimpleNamespace(path_found=True, path=None, path_length=1),
            SimpleNamespace(path_found=True, path=[-1], path_length=1),
            SimpleNamespace(path_found=True, path=[True], path_length=1),
            SimpleNamespace(path_found=True, path=[0], path_length=1),
            SimpleNamespace(path_found=True, path=[2], path_length=5),
        ):
            cp, _ = self.fake_cayleypy(raw)
            with patch("cayleypy.research.search.importlib.import_module", return_value=cp):
                with self.assertRaises(InvalidWitnessError):
                    cayleypy_search(lrx_graph(3), start=(1, 0, 2))

    def test_missing_optional_package_has_actionable_error(self):
        with patch("cayleypy.research.search.importlib.import_module", side_effect=ImportError("no package")):
            with self.assertRaisesRegex(BackendUnavailable, "Install"):
                cayleypy_search(lrx_graph(3), start=(1, 0, 2))

    def test_native_refuses_unsupported_hardware_without_importing(self):
        with patch("cayleypy.research.search.platform.system", return_value="Darwin"):
            with patch("cayleypy.research.search.importlib.import_module") as importer:
                with self.assertRaisesRegex(BackendUnavailable, "Linux"):
                    multigpu_search(lrx_graph(3), start=(1, 0, 2))
                importer.assert_not_called()

    def test_native_dispatch_is_strict_and_replayed(self):
        raw = SimpleNamespace(path_found=True, path=[2], path_length=1, native_metadata={"devices": [0, 1]})
        cp, calls = self.fake_cayleypy(raw)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))

        class Options:
            def __init__(self, num_gpus):
                self.num_gpus = num_gpus
                self.devices = None

        def search(graph, **kwargs):
            calls["native"] = kwargs
            return raw

        native = SimpleNamespace(NativeOptions=Options, beam_search=search)
        modules = {"cayleypy": cp, "torch": torch, "cayleypy.native": native}
        with patch("cayleypy.research.search.platform.system", return_value="Linux"):
            with patch("cayleypy.research.search.importlib.import_module", side_effect=modules.__getitem__):
                result = multigpu_search(lrx_graph(3), start=(1, 0, 2), num_gpus=2)
        self.assertEqual(calls["native"]["backend"], "native")
        self.assertEqual(calls["native"]["native_options"].num_gpus, 2)
        self.assertEqual(calls["graph"]["device"], "cuda")
        self.assertEqual(result.backend, "multigpu_native_beam")
        self.assertTrue(result.path_verified)
        self.assertFalse(result.exact)

    @unittest.skipUnless(importlib.util.find_spec("cayleypy"), "optional CayleyPy is not installed")
    def test_real_cayleypy_cpu_smoke(self):
        result = cayleypy_search(lrx_graph(3), start=(1, 0, 2), beam_width=16, max_depth=3)
        self.assertTrue(result.path_verified)
        self.assertEqual(result.path, ("X",))


if __name__ == "__main__":
    unittest.main()
