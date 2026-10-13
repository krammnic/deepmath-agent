# Run a research session

Prepare the environment once, choose a model, and set the scope of the investigation. DeepMath supports Python 3.10 or later. Research sessions also require access to a text model and a fitted reasoning-scoring artifact.

## Prepare the environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[research]'
deepmath doctor
```

The repository and built package contain pinned source snapshots. No additional repository checkout is needed. `python scripts/bootstrap.py --check` verifies bundled files offline. Revision, file integrity, and distribution terms are recorded in the [source lock](../dependencies.lock.json) and its source manifest.

For accelerated search across several GPUs, prepare a Linux machine with the requested NVIDIA devices, CUDA toolkit, compatible compiler, and NCCL. Use [the GPU configuration](../configs/multigpu.json). The installation includes the native search sources; first use may provision compiler dependencies, compile, and calibrate the runtime.

## Prepare reasoning scores

Fit the scorer on a varied reference set of mathematical problems and proposed solutions. The CSV needs `problem` and `completion` columns; correctness labels are not required. Preserve the dataset and its provenance, and keep the fitted artifact fixed during a research session.

```bash
mkdir -p artifacts
deepmath qk fit \
  --reference-csv unlabeled_reference.csv \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output artifacts/reasoning-scorer.json \
  --device cuda:0 --dtype bfloat16 \
  --max-length 4096 --micro-batch-size 2 --reference-seed 1729
```

This command uses a pinned model revision. Model loading can download weights and execute model-specific loading code, so use a revision you have reviewed. Select a device with sufficient memory; CPU scoring uses `device="cpu"` and `dtype="float32"`.

Scores prioritize proposed arguments. Their value is recorded with the artifact digest, model revision, and complete candidate text. Proof status depends on the mathematical evidence.

## Choose the session settings

[The research configuration](../configs/research.json) specifies three rounds, a bounded search budget, and required reasoning scores. It uses CPU search and CUDA scoring by default. Adjust the devices to your environment, then set:

```bash
export DEEPMATH_MODEL='your-model-id'
export OPENAI_API_KEY='your-key'
export DEEPMATH_QK_ARTIFACT='/absolute/path/to/artifacts/reasoning-scorer.json'
```

The `llm` section accepts a compatible endpoint and an environment-variable name for its API key. Use `role_models` to assign different models to `theorist`, `counterexample_hunter`, `holographer`, `critic`, and `synthesizer`. Each round makes up to five model requests; `max_completion_tokens` limits each response.

For a local model server, start from [the local configuration](../configs/local-llm.json). It targets `localhost:8000/v1` and allows sessions without a scoring artifact. Scores are recorded as unavailable in that configuration.

## Run and review

```bash
deepmath workspace init research
deepmath run examples/problems/lrx-n6-diameter.json \
  --config configs/research.json --workspace research --output runs/lrx-research
```

Choose a new output directory for every session. Reuse the workspace to carry verified run records, failed approaches, and open obligations into the next session on the same problem. Records from different problem definitions stay separate. The runner checks required scoring prerequisites before making model requests. Completed experiments remain available if a later step fails.

```bash
deepmath verify-run runs/lrx-research
deepmath replay runs/lrx-research
deepmath export runs/lrx-research --destination exports/lrx-research
```

Export prepares a local package for review. Inspect saved context or register an existing session with:

```bash
deepmath workspace context research examples/problems/lrx-n6-diameter.json
deepmath workspace add research runs/lrx-research
```

The [validation record](validation.md) describes the execution coverage established so far.

## Define another graph

Use `deepmath graphs` to inspect constructor names and parameters. A problem can select a catalog graph directly:

```json
{
  "id": "sl2-mod3",
  "title": "SL(2,3) diameter",
  "statement": "Compute the diameter with fundamental root generators and their inverses.",
  "graph": {
    "family": "cayleypy",
    "constructor": "MatrixGroups.special_linear_fundamental_roots",
    "kwargs": {"n": 2, "modulo": 3}
  },
  "objective": "diameter"
}
```

Inspect the resolved action with `deepmath graph problem.json`, then use `experiment` or `run`. Explicit `permutation` graphs accept named gather generators and a `central_state`, including repeated labels. Explicit `matrix` graphs accept named square matrix generators, a modulus, and an optional central state. Targets may use flat arrays or matrix rows. The run freezes the resolved definition for replay.

Diameter experiments share the state budget across all necessary roots for nonfree actions. Matrix experiments use exact arithmetic for verification and support CPU beam search; the native multi-GPU runtime accepts gather actions. For custom Python investigations, the installation provides the complete CayleyPy API, including BFS variants, random walks, datasets, predictors, models, and training utilities.
