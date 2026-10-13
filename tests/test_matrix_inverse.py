"""Matrix group inverses work through the public CayleyPy graph API."""

import unittest

import numpy as np

import cayleypy as cp


class MatrixInverseTests(unittest.TestCase):
    def test_modular_inverses_use_the_ring_not_the_real_numbers(self):
        cases = [
            (5, [[2]], [[3]]),
            (6, [[2, 3], [3, 2]], [[2, 3], [3, 2]]),
            (5, [[0, 1], [1, 0]], [[0, 1], [1, 0]]),
        ]
        for modulo, matrix, inverse in cases:
            with self.subTest(modulo=modulo, matrix=matrix):
                generator = cp.MatrixGenerator.create(matrix, modulo)
                self.assertEqual(type(generator.inv), cp.MatrixGenerator)
                np.testing.assert_array_equal(generator.inv.matrix, inverse)
                self.assertTrue(generator.is_inverse_to(generator.inv))
                self.assertEqual(generator.inv.inv, generator)
                definition = cp.CayleyGraphDef.for_matrix_group(generators=[generator])
                self.assertEqual(definition.with_inverted_generators().with_inverted_generators(), definition)

    def test_noninvertible_modular_generators_are_rejected(self):
        for matrix in ([[2]], [[1, 1], [1, 1]]):
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "not invertible"):
                    cp.MatrixGenerator.create(matrix, modulo=6).inv

    def test_nonmodular_integer_inverse_behavior_is_preserved(self):
        for matrix, inverse in (
            ([[1, 1], [0, 1]], [[1, -1], [0, 1]]),
            ([[0, 1], [1, 0]], [[0, 1], [1, 0]]),
        ):
            with self.subTest(matrix=matrix):
                generator = cp.MatrixGenerator.create(matrix)
                self.assertEqual(generator.inv.modulo, 0)
                np.testing.assert_array_equal(generator.inv.matrix, inverse)
                self.assertTrue(generator.is_inverse_to(generator.inv))
                self.assertEqual(generator.inv.inv, generator)
        with self.assertRaisesRegex(AssertionError, "not invertible"):
            cp.MatrixGenerator.create([[2]]).inv
        with self.assertRaises(np.linalg.LinAlgError):
            cp.MatrixGenerator.create([[0]]).inv

    def test_public_cpu_beam_search_replays_with_modular_generators(self):
        for modulo, matrix in ((5, [[2]]), (6, [[2, 3], [3, 2]])):
            with self.subTest(modulo=modulo):
                generator = cp.MatrixGenerator.create(matrix, modulo)
                identity = np.eye(generator.n, dtype=np.int64)
                definition = cp.CayleyGraphDef.for_matrix_group(
                    generators=[generator], central_state=matrix, generator_names=["g"]
                )
                graph = cp.CayleyGraph(definition, device="cpu", random_seed=0, verbose=0)
                result = graph.beam_search(
                    start_state=identity, beam_mode="simple", beam_width=8,
                    max_steps=4, return_path=True,
                )
                self.assertTrue(result.path_found)
                self.assertEqual(result.path, [0])
                state = identity
                for index in result.path:
                    state = definition.generators[index].apply(state)
                np.testing.assert_array_equal(state, matrix)


if __name__ == "__main__":
    unittest.main()
