# Run a research session

Prepare the environment once, choose a model, and set the scope of the investigation. DeepMath supports Python 3.10 or later. Research sessions also require access to a text model and a fitted reasoning-scoring artifact.

## Prepare the environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py
python -m pip install '.[qk]'
python -m pip install ./external/cayleypy ./external/qk-zero-shot
deepmath doctor
```

The setup uses pinned source revisions. Existing local changes are preserved, and `python scripts/bootstrap.py --check` checks the source state without changing it. Revision and license records are retained in the [source lock](../dependencies.lock.json).

For accelerated search across several GPUs, prepare a Linux machine with the requested NVIDIA devices, CUDA toolkit, compatible compiler, and NCCL. Install the accelerator support in the same environment:

```bash
python -m pip install ./external/MultiGPUBeamSearch/integrations/cayleypy_native
```

Use [the GPU configuration](../configs/multigpu.json) for these sessions. First use may compile and calibrate the search runtime.

## Prepare reasoning scores

Fit the scorer on a varied reference set of mathematical problems and proposed solutions. The CSV needs `problem` and `completion` columns; correctness labels are not required. Preserve the dataset and its provenance, and keep the fitted artifact fixed during a research session.

```bash
mkdir -p artifacts
qk-zero-shot fit \
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
deepmath run examples/problems/lrx-n6-diameter.json \
  --config configs/research.json --output runs/lrx-research
```

Choose a new output directory for every session. The runner checks required scoring prerequisites before making model requests. The report presents observations and proof obligations; the structured record retains proposals, reviews, and experiment details. Completed experiments remain available if a later step fails.

```bash
deepmath verify-run runs/lrx-research
deepmath replay runs/lrx-research
deepmath export runs/lrx-research --destination exports/lrx-research
```

Export prepares a local package for review. Use its findings and handoff to choose the next bounded investigation. The [validation record](validation.md) describes the execution coverage established so far.
