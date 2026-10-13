import tempfile
import unittest
from pathlib import Path

import numpy as np

from cayleypy.qk import (
    ZeroShotQKArtifact,
    aggregate_frozen_rsym,
    build_qk_verifier_prompt,
    fit_responsiveness_weights,
    head_votes_from_semantic_scores,
)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text))))}

    def decode(self, ids, **_kwargs):
        return "x" * len(ids)


class CoreTest(unittest.TestCase):
    def test_prompt_has_three_by_three_fixed_views(self):
        prompt, queries, keys, semantics = build_qk_verifier_prompt(
            "What is 2+2?",
            "It is 4.",
            CharacterTokenizer(),
            max_length=4096,
            mapping="a_correct",
        )
        self.assertEqual(len(queries), 3)
        self.assertEqual([len(option) for option in keys], [3, 3])
        self.assertEqual(semantics, [0, 1])
        self.assertEqual([prompt[index] for index in queries], ["V", "t", ":"])
        self.assertEqual([prompt[index] for index in keys[0]], ["t", ".", "\n"])

    def test_duplicate_views_are_not_counted_twice(self):
        scores = np.zeros((2, 1, 2, 3, 2), dtype=np.float64)
        scores[..., 0, 0] = 2.0
        scores[..., 0, 1] = 1.0
        scores[..., 1, :] = scores[..., 0, :]
        scores[..., 2, 0] = 0.0
        scores[..., 2, 1] = 3.0
        np.testing.assert_allclose(head_votes_from_semantic_scores(scores, 2), 0.0)

    def test_responsiveness_suppresses_static_heads(self):
        votes = np.array(
            [[1.0, 1.0, 0.5], [-1.0, 1.0, -0.5], [1.0, 1.0, 0.5], [-1.0, 1.0, -0.5]]
        )
        weights = fit_responsiveness_weights(votes)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertGreater(float(weights[0]), float(weights[2]))

    def test_frozen_symmetric_margin(self):
        one = np.array([[1.0, -1.0], [-1.0, 1.0]])
        two = np.array([[0.5, 0.5], [-0.5, -0.5]])
        result = aggregate_frozen_rsym(one, two, np.array([0.75, 0.25]), np.array([0.5, 0.5]))
        np.testing.assert_allclose(result, [0.5, -0.5])

    def test_artifact_enforces_zero_labels_and_zero_threshold(self):
        artifact = ZeroShotQKArtifact(
            schema_version=1,
            method="frozen-cf-qk-rsym",
            model_id="model",
            model_revision="revision",
            n_layers=2,
            n_query_heads=2,
            n_kv_heads=1,
            layer_start_fraction=0.5,
            head_indices=[2, 3],
            one_mapping_weights=[0.25, 0.75],
            two_mapping_weights=[0.5, 0.5],
            unlabeled_reference_size=4,
            reference_seed=1,
            prompt_mappings=["a_correct", "b_correct"],
            query_views=["verdict_start", "verdict_end", "verdict_colon"],
            key_views=["last_content", "terminal_punctuation", "end_of_line"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            artifact.save(path)
            self.assertEqual(ZeroShotQKArtifact.load(path), artifact)


if __name__ == "__main__":
    unittest.main()
