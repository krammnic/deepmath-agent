"""Exercise projection capture and artifact fitting on an actual local transformer."""

import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from deepmath.cli import main
from deepmath.qk import QKConfig, QKVerifier


INFERENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None
                          for name in ("qk_zero_shot", "transformers", "tokenizers", "torch"))


@unittest.skipUnless(INFERENCE_AVAILABLE, "Install .[research] for the local transformer integration test")
class QKInferenceTests(unittest.TestCase):
    def test_fit_reload_capture_and_score_without_network(self):
        import numpy as np
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from qk_zero_shot import ZeroShotQKArtifact

        examples = [
            ("What is two plus two?", "Adding two to two gives four."),
            ("Is every permutation an involution?", "A three-cycle is a counterexample."),
            ("What is one plus one?", "One plus one equals seven."),
            ("Does a path certify a lower bound?", "A path gives an upper bound on distance."),
        ]
        # Random weights test integration mechanics, never mathematical accuracy.
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng():
            root = Path(tmp)
            model_path, artifact_path = root / "model", root / "artifact.json"
            model_path.mkdir()
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.train_from_iterator(
                [" ".join(pair) for pair in examples],
                trainers.BpeTrainer(vocab_size=320, special_tokens=["[UNK]", "[PAD]", "[EOS]"],
                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
            )
            wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                             pad_token="[PAD]", eos_token="[EOS]")
            wrapped.save_pretrained(model_path)
            torch.manual_seed(17)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(wrapped), hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=1024, pad_token_id=wrapped.pad_token_id,
                eos_token_id=wrapped.eos_token_id,
            ))
            model.save_pretrained(model_path)
            del model
            reference = root / "reference.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["problem", "completion"])
                writer.writerows(examples)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["qk", "fit", "--reference-csv", str(reference),
                             "--model", str(model_path), "--output", str(artifact_path),
                             "--device", "cpu", "--dtype", "float32", "--max-length", "768"])
            self.assertEqual(code, 0)
            artifact = ZeroShotQKArtifact.load(artifact_path)
            self.assertEqual(artifact.gold_labels_used, 0)
            self.assertEqual(artifact.unlabeled_reference_size, len(examples))
            np.testing.assert_allclose(sum(artifact.one_mapping_weights), 1.0)
            with QKVerifier(QKConfig(artifact_path, device="cpu", dtype="float32", max_length=768)) as verifier:
                scores = verifier.score(examples[0][0], [pair[1] for pair in examples])
                repeated = verifier.score(examples[0][0], [pair[1] for pair in examples])
                self.assertEqual([s.margin for s in scores], [s.margin for s in repeated])
                self.assertTrue(all(s.status == "scored" and -1 <= s.margin <= 1 for s in scores))
                self.assertTrue(all(s.artifact_sha256 and not s.is_proof for s in scores))
                self.assertEqual(verifier._scorer.n_heads, 4)
                self.assertEqual(verifier._scorer.n_kv_heads, 2)
                hooks = verifier._scorer.capture.handles
                self.assertTrue(hooks)
            self.assertEqual(hooks, [])
            result_path = root / "scored.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["qk", "score", "--input-csv", str(reference),
                             "--artifact", str(artifact_path), "--output", str(result_path),
                             "--device", "cpu", "--dtype", "float32", "--max-length", "768"])
            self.assertEqual(code, 0)
            with result_path.open(newline="") as stream:
                scored = list(csv.DictReader(stream))
            self.assertEqual(len(scored), len(examples))
            self.assertTrue(all("qk_zero_shot_margin" in row for row in scored))


if __name__ == "__main__":
    unittest.main()
