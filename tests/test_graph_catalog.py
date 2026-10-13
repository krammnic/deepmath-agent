"""Native graph definitions remain replayable across regular and Schreier actions."""

import importlib.util
import itertools
import json
import unittest
from unittest.mock import patch

from deepmath.experiments import execute, graph_for
from deepmath.holography import probe_area
from deepmath.models import Experiment, Problem, RunConfig
from deepmath.search import (BackendUnavailable, GraphSpec, apply_move, cayleypy_graph,
                             cayleypy_search, exact_bfs, exact_diameter, graph_catalog,
                             graph_from_config, multigpu_search, replay_path)


def make_problem(graph, **kwargs):
    return Problem.from_dict({"id": "native-graph", "title": "Finite research instance",
                              "statement": "Measure distances at these exact parameters.",
                              "graph": graph, **kwargs})


class GeneralGraphEvidenceTests(unittest.TestCase):
    def test_schreier_root_eccentricity_is_not_a_diameter(self):
        graph = GraphSpec("binary-adjacent-swaps", {
            "a": (1, 0, 2, 3), "b": (0, 2, 1, 3), "c": (0, 1, 3, 2),
        }, central_state=(0, 1, 0, 1))
        root = exact_bfs(graph)
        self.assertTrue(root.complete)
        self.assertEqual(root.metadata["root_eccentricity"], 3)
        self.assertIsNone(root.diameter)
        full = exact_diameter(graph)
        self.assertEqual(full.diameter, 4)
        self.assertEqual(full.metadata["completed_roots"], 6)
        self.assertEqual(full.metadata["total_state_visits"], 36)
        self.assertEqual(replay_path(graph, full.start, full.path), full.target)
        self.assertNotEqual(full.start, graph.identity)
        # Independent Hamming transport distance on binary strings.
        positions = lambda state: [i for i, bit in enumerate(state) if bit]
        self.assertEqual(full.distance, sum(abs(a - b) for a, b in zip(positions(full.start), positions(full.target))))

    def test_all_roots_shares_a_hard_total_budget(self):
        graph = GraphSpec("binary-adjacent-swaps", {
            "a": (1, 0, 2, 3), "b": (0, 2, 1, 3), "c": (0, 1, 3, 2),
        }, central_state=(0, 1, 0, 1))
        result = exact_diameter(graph, max_states=10)
        self.assertLessEqual(result.metadata["total_state_visits"], 10)
        self.assertFalse(result.complete)
        self.assertIsNone(result.diameter)
        self.assertTrue(result.path_verified)
        self.assertTrue(result.metadata["component_exhausted"])

    def test_explicit_matrix_actions_and_targets(self):
        graph_config = {"family": "matrix", "modulo": 3, "generators": {
            "e": [[1, 1], [0, 1]], "f": [[1, 0], [1, 1]],
        }}
        problem = make_problem(graph_config, objective="path", target=[[1, 1], [0, 1]])
        graph = graph_for(problem)
        self.assertEqual(problem.target, (1, 1, 0, 1))
        self.assertEqual(problem.graph["n"], 4)
        experiment = Experiment.from_dict({"kind": "path"}, problem)
        result = execute(problem, experiment, RunConfig())
        self.assertEqual(result["path"], ["e"])
        self.assertTrue(result["shortest_path"])
        self.assertEqual(apply_move(graph, graph.identity, "e"), problem.target)
        with self.assertRaisesRegex(ValueError, "residues"):
            make_problem(graph_config, objective="path", target=[3, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "distinct-label"):
            probe_area(graph)
        with self.assertRaisesRegex(BackendUnavailable, "matrix"):
            multigpu_search(graph, start=graph.identity)

    def test_singular_generators_are_rejected_before_diameter_claims(self):
        for modulus, matrix in ((4, [[2, 0], [0, 1]]), (3, [[1, 1], [1, 1]])):
            with self.assertRaisesRegex(ValueError, "invertible"):
                GraphSpec("singular", {"g": matrix}, action="matrix", modulo=modulus)

    def test_rectangular_matrix_action_is_not_assumed_regular(self):
        graph = GraphSpec("vector-action", {"e": ((1, 1), (0, 1)), "f": ((1, 0), (1, 1))},
                          central_state=(1, 0), action="matrix", modulo=2)
        result = exact_bfs(graph)
        self.assertEqual(result.explored_states, 3)
        self.assertIsNone(result.diameter)
        self.assertEqual(exact_diameter(graph).diameter, 2)

    def test_matrix_replay_uses_exact_arithmetic_even_for_large_modulus(self):
        graph = GraphSpec("large-modulus", {"g": ((1, 2**31 - 1, 2**31 - 1), (0, 1, 0), (0, 0, 1))},
                          action="matrix", modulo=2**31)
        state = (2**31 - 1,) * 9
        result = apply_move(graph, state, "g")
        self.assertEqual(result[:3], (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "overflow"):
            cayleypy_graph(graph)

    def test_repeated_label_targets_preserve_multiplicity(self):
        config = {"family": "permutation", "n": 4, "generators": {"L": [1, 2, 3, 0]},
                  "central_state": [0, 0, 1, 1]}
        problem = make_problem(config, objective="path", target=[0, 1, 1, 0])
        self.assertTrue(execute(problem, Experiment("path", problem.target), RunConfig())["path_verified"])
        with self.assertRaisesRegex(ValueError, "multiset"):
            make_problem(config, objective="path", target=[0, 1, 2, 3])


@unittest.skipUnless(importlib.util.find_spec("cayleypy"), "CayleyPy is not installed")
class NativeCatalogTests(unittest.TestCase):
    def test_catalog_exposes_native_families(self):
        catalog = {item["constructor"]: item for item in graph_catalog()}
        for name in ("PermutationGroups.pancake", "MatrixGroups.heisenberg", "Puzzles.rubik_cube", "GapPuzzles.puzzle"):
            self.assertIn(name, catalog)
            self.assertIn("signature", catalog[name])

    def test_pancake_exact_and_native_search(self):
        problem = make_problem({"family": "cayleypy", "constructor": "PermutationGroups.pancake", "kwargs": {"n": 4}})
        graph = graph_for(problem)
        result = exact_diameter(graph)
        self.assertEqual(result.explored_states, 24)
        self.assertEqual(result.diameter, 4)
        target = (3, 2, 1, 0)
        beam = cayleypy_search(graph, start=graph.identity, target=target, beam_width=64, max_depth=5)
        self.assertTrue(beam.path_verified)
        self.assertFalse(beam.exact)
        self.assertEqual(replay_path(graph, beam.start, beam.path), target)
        native = cayleypy_graph(graph)
        self.assertTrue(callable(native.beam_search))
        self.assertTrue(callable(native.random_walks))

    def test_schreier_native_search(self):
        graph = graph_from_config({"family": "cayleypy", "constructor": "PermutationGroups.coxeter",
                                   "kwargs": {"n": 4}, "central_state": [0, 1, 0, 1]})
        target = (1, 0, 0, 1)
        beam = cayleypy_search(graph, start=graph.identity, target=target, beam_width=32, max_depth=4)
        self.assertEqual(beam.path, ("(0,1)",))
        self.assertTrue(beam.path_verified)

    def test_modular_matrix_native_and_independent_distances(self):
        graph = graph_from_config({"family": "cayleypy", "constructor": "MatrixGroups.special_linear_fundamental_roots",
                                   "kwargs": {"n": 2, "modulo": 3}})
        result = exact_diameter(graph)
        self.assertEqual(result.explored_states, 24)
        self.assertEqual(result.diameter, 4)
        # Independent enumeration identifies SL(2,3) by its determinant.
        matrices = {values for values in itertools.product(range(3), repeat=4)
                    if (values[0] * values[3] - values[1] * values[2]) % 3 == 1}
        self.assertEqual(set(result.distances), matrices)
        goal = (1, 1, 0, 1)
        beam = cayleypy_search(graph, start=graph.identity, target=goal, beam_width=32, max_depth=5)
        self.assertEqual(beam.path, ("e1",))
        self.assertTrue(beam.path_verified)
        self.assertFalse(beam.exact)

    def test_heisenberg_native_definition_covers_expected_finite_group(self):
        graph = graph_from_config({"family": "cayleypy", "constructor": "MatrixGroups.heisenberg",
                                   "kwargs": {"n": 3, "modulo": 3}})
        result = exact_diameter(graph)
        expected = {(1, a, c, 0, 1, b, 0, 0, 1) for a, b, c in itertools.product(range(3), repeat=3)}
        self.assertEqual(set(result.distances), expected)
        self.assertEqual(result.explored_states, 27)

    def test_resolved_catalog_replays_without_loading_cayleypy(self):
        problem = make_problem({"family": "cayleypy", "constructor": "PermutationGroups.pancake", "kwargs": {"n": 4}})
        saved = json.loads(json.dumps(problem.to_dict()))
        with patch("deepmath.search._load_optional", side_effect=AssertionError("must use frozen definition")):
            recovered = Problem.from_dict(saved)
            self.assertEqual(graph_for(recovered).to_dict(), graph_for(problem).to_dict())
            self.assertEqual(execute(recovered, Experiment("diameter"), RunConfig())["diameter"], 4)

    def test_native_modular_inverse_handles_nonunit_integer_determinants(self):
        # The inverse of [2] modulo 5 is [3], not the real inverse [0.5].
        # Over Z/6Z this second matrix has no unit entry in its first column,
        # so inversion also needs Bezout combinations rather than pivot swaps.
        for modulo, matrix in ((5, ((2,),)), (6, ((2, 3), (3, 2)))):
            graph = GraphSpec("modular-units", {"g": matrix}, action="matrix", modulo=modulo)
            target = apply_move(graph, graph.identity, "g")
            beam = cayleypy_search(graph, start=graph.identity, target=target, beam_width=8, max_depth=4)
            self.assertEqual(beam.path, ("g",))
            self.assertTrue(beam.path_verified)
            generator = cayleypy_graph(graph).definition.generators[0]
            self.assertTrue(generator.is_inverse_to(generator.inv))

    def test_catalog_central_state_never_truncates_floats_or_booleans(self):
        for center in ([0.9, 1.9, 2.9], [False, 1, 2], [[0.5, 1], [1, 0]]):
            with self.assertRaisesRegex(ValueError, "integers"):
                graph_from_config({"family": "cayleypy", "constructor": "PermutationGroups.coxeter",
                                   "kwargs": {"n": 3}, "central_state": center})

    def test_conjugacy_partitions_use_json_and_lx_catalog_keeps_directed_moves(self):
        graph = graph_from_config({"family": "cayleypy", "constructor": "PermutationGroups.conjugacy_classes",
                                   "kwargs": {"n": 3, "classes": [{"partition": [2], "count": None}]}})
        self.assertEqual(len(graph.generators), 3)
        self.assertEqual(exact_diameter(graph).diameter, 2)
        lx = graph_from_config({"family": "cayleypy", "constructor": "prepare_graph", "kwargs": {"name": "lx-4"}})
        self.assertEqual(len(lx.generators), 2)
        self.assertEqual(exact_diameter(lx).diameter, 6)

    def test_native_bfs_and_random_walks_access_full_graph_api(self):
        graph = graph_from_config({"family": "cayleypy", "constructor": "PermutationGroups.pancake", "kwargs": {"n": 4}})
        native = cayleypy_graph(graph)
        reference = exact_bfs(graph)
        result = native.bfs(max_diameter=5, max_layer_size_to_explore=100, max_layer_size_to_store=None)
        self.assertTrue(result.bfs_completed)
        self.assertEqual(tuple(result.layer_sizes), reference.layer_sizes)
        states, lengths = native.random_walks(width=4, length=5, mode="classic")
        # Random-walk labels are upper bounds, even when the native BFS agrees
        # with this independently enumerated small instance.
        for state, upper_bound in zip(states.tolist(), lengths.tolist()):
            self.assertLessEqual(reference.distances[tuple(state)], upper_bound)

    def test_config_never_evaluates_import_paths_or_arbitrary_files(self):
        for constructor, kwargs in (("os.system", {"command": "echo no"}),
                                    ("GapPuzzles.load_puzzle_from_file", {"file_name": "/etc/passwd"}),
                                    ("GapPuzzles.puzzle", {"puzzle_name": "../../../etc/passwd"})):
            with self.assertRaises(ValueError):
                graph_from_config({"family": "cayleypy", "constructor": constructor, "kwargs": kwargs})
        with self.assertRaisesRegex(ValueError, "modulo"):
            graph_from_config({"family": "cayleypy", "constructor": "MatrixGroups.heisenberg"})


if __name__ == "__main__":
    unittest.main()
