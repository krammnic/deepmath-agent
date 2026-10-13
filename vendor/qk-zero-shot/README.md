# QK Zero-Shot

Calibration-free verification of candidate reasoning from a frozen causal
language model's attention query/key projections.

This repository contains code only. It intentionally excludes benchmark data,
model weights, generated completions, fitted artifacts, caches, and results.

## Method

For each problem and candidate solution, the verifier constructs two
counterbalanced prompts:

1. `A = correct`, `B = incorrect`;
2. `A = incorrect`, `B = correct`.

For every attention head in the latter half of the frozen verifier, it reads
three query positions around `Verdict:` (start, end, colon) and three key
positions for each semantic option (last content token, punctuation, line end).
This gives nine fixed QK views per semantic verdict.

Each view votes by the sign of
`QK(correct) - QK(incorrect)`. Fixed head weights are estimated once from an
**unlabeled** reference set: heads receive weight when their boundary views
agree and their verdict varies across examples; static option-biased heads are
suppressed. At inference, the one-mapping and counterbalanced two-mapping
margins are averaged. The decision rule is fixed:

```text
margin > 0  => correct
margin <= 0 => incorrect
```

The artifact schema enforces `gold_labels_used = 0` and
`decision_threshold = 0.0`. This is unlabeled batch-adaptive calibration-free
verification: the fixed weights use completions but no correctness labels.

## Install

```bash
python -m pip install -e .
```

The implementation expects decoder attention modules exposing `q_proj` and
`k_proj`, as in Qwen and Llama-family Hugging Face models. A CUDA GPU is
recommended.

## Input format

Both commands read CSV files with `problem` and `completion` columns:

```csv
problem,completion
"What is 2+2?","2+2=4, so the answer is 4."
```

No correctness-label column is read.

## Fit a label-free artifact

Use a sufficiently varied, unlabeled pool of base-model completions:

```bash
qk-zero-shot fit \
  --reference-csv unlabeled_reference.csv \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output artifacts/qwen3-8b-qk-zero-shot.json \
  --device cuda:0
```

The model revision, eligible head set, both fixed weight vectors, reference-set
size, prompt mappings, Q/K views, and zero decision threshold are recorded in
the JSON artifact.

## Score new candidates

```bash
qk-zero-shot score \
  --input-csv candidates.csv \
  --artifact artifacts/qwen3-8b-qk-zero-shot.json \
  --output scored.csv \
  --device cuda:0
```

The output adds:

- `qk_zero_shot_margin`, bounded in `[-1, 1]`;
- `qk_zero_shot_prediction`, either `correct` or `incorrect`.

## Python API

```python
from qk_zero_shot import ZeroShotQKArtifact, ZeroShotQKScorer

artifact = ZeroShotQKArtifact.load("artifacts/qwen3-8b-qk-zero-shot.json")
scorer = ZeroShotQKScorer(
    model_id=artifact.model_id,
    model_revision=artifact.model_revision,
    device="cuda:0",
    max_length=4096,
    layer_start_fraction=artifact.layer_start_fraction,
)
try:
    margins = scorer.rewards(problems, completions, artifact)
    predictions = margins > 0.0
finally:
    scorer.close()
```

## Tests

The pure prompt/aggregation tests do not load a model:

```bash
python -m pip install -e '.[test]'
pytest
```

## Scope

This repository exposes only the zero-label QK verifier. Supervised head
selection, fitted correctness thresholds, simulated annealing, benchmark
loaders, candidate generation, and experiment results are deliberately out of
scope.
