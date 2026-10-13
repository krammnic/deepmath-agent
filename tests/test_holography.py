"""Finite mathematical counterexamples and evidence-boundary checks."""

import itertools
import json
import unittest
from unittest.mock import patch

from cayleypy.research.holography import _cyclic_inversions, probe_area
from cayleypy.research.search import GraphSpec, lrx_graph, replay_path


class HolographyProbeTests(unittest.TestCase):
    def test_ordinary_inversion_lower_bound_fails_for_one_lrx_rotation(self):
        graph = lrx_graph(4)
        result = probe_area(graph)
        witness = result["lower_bound"]["counterexample"]
        self.assertEqual(result["evidence"], "finite_heuristic_probe")
        self.assertTrue(result["complete"])
        self.assertEqual(result["explored_states"], 24)
        self.assertFalse(result["lower_bound"]["holds_on_tested_states"])
        self.assertFalse(result["equality"]["holds_on_tested_states"])
        self.assertEqual(witness["shortest_word"], ["L"])
        self.assertEqual(witness["exact_distance"], 1)
        self.assertEqual(witness["statistic_value"], 3)
        self.assertEqual(list(replay_path(graph, graph.identity, witness["shortest_word"])), witness["state"])
        self.assertTrue(witness["path_verified"])
        self.assertTrue(witness["shortest_path"])
        self.assertEqual(result["universal_claim"], "not_established")

    def test_cyclic_variant_lower_bound_passes_lrx_four_but_equality_fails(self):
        graph = lrx_graph(4)
        result = probe_area(graph, statistic="cyclic_inversions")
        self.assertTrue(result["complete"])
        self.assertTrue(result["lower_bound"]["holds_on_tested_states"])
        self.assertTrue(result["lower_bound"]["holds_on_generated_component"])
        self.assertIsNone(result["lower_bound"]["counterexample"])
        self.assertFalse(result["equality"]["holds_on_tested_states"])
        example = result["equality"]["counterexample"]
        self.assertEqual(example["statistic_value"], 0)
        self.assertEqual(example["exact_distance"], 1)
        self.assertEqual(example["shortest_word"], ["L"])
        self.assertEqual(result["universal_claim"], "not_established")
        self.assertEqual(json.loads(json.dumps(result))["identity_value"], 0)

    def test_cyclic_recurrence_matches_independent_all_rotation_counts(self):
        for state in itertools.permutations(range(5)):
            rotations = [state[k:] + state[:k] for k in range(5)]
            expected = min(sum(row[i] > row[j] for i in range(5) for j in range(i + 1, 5))
                           for row in rotations)
            self.assertEqual(_cyclic_inversions(state), expected)

    def test_partial_bfs_can_already_refute_lower_bound(self):
        graph = lrx_graph(4)
        result = probe_area(graph, max_states=2)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["termination"], "max_states")
        self.assertEqual(result["explored_states"], 2)
        self.assertEqual(result["scope"], "discovered_states")
        self.assertEqual(result["lower_bound"]["outcome"], "counterexample_found")
        self.assertIsNone(result["lower_bound"]["holds_on_generated_component"])
        self.assertEqual(result["lower_bound"]["counterexample"]["shortest_word"], ["L"])

    def test_identity_only_probe_cannot_endorse_any_global_statement(self):
        result = probe_area(lrx_graph(4), statistic="cyclic_inversions", max_depth=0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["explored_states"], 1)
        self.assertEqual(result["universal_claim"], "not_established")
        for name in ("lower_bound", "equality"):
            self.assertTrue(result[name]["holds_on_tested_states"])
            self.assertIsNone(result[name]["holds_on_generated_component"])
            self.assertEqual(result[name]["outcome"], "no_counterexample_in_discovered_states")
            self.assertIsNone(result[name]["counterexample"])

    def test_complete_generated_subgroup_is_scoped_to_that_graph(self):
        graph = GraphSpec("identity-only", {"I": (0, 1, 2)}, expected_order=1)
        result = probe_area(graph)
        self.assertTrue(result["complete"])
        self.assertEqual(result["scope"], "generated_component")
        self.assertTrue(result["equality"]["holds_on_generated_component"])
        self.assertEqual(result["universal_claim"], "not_established")

    def test_resource_and_statistic_errors_fail_before_search(self):
        with patch("cayleypy.research.holography.exact_bfs") as search:
            for kwargs in ({"statistic": "paper_theorem"}, {"max_states": 100001},
                           {"max_states": 0}, {"max_states": True}, {"max_depth": -1}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        probe_area(lrx_graph(4), **kwargs)
            with self.assertRaisesRegex(ValueError, "smaller finite instance"):
                probe_area(lrx_graph(17))
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
