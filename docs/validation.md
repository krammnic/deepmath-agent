# Validation record — 13 October 2026

The automated suite passed **79 tests with zero skips** in the Python 3.13.12 test environment. Coverage includes the research loop, experiment replay, failure recovery, artifact integrity, source pinning, score handling, and exact directed and undirected distances. Package installation and access to the bundled research context also passed.

## Computational results

| Check | Result |
| --- | --- |
| Exhaustive LRX₆ search | All 720 states visited; diameter 15 |
| Exhaustive directed LX₆ search | All 720 states visited; diameter 18 |
| CPU beam search for the reflected LRX₈ witness | A 28-move word, independently replayed |
| Reflected-witness construction | Endpoints and word lengths checked for n=4…100 |
| Independent shortest-path checks | Exact witness distances confirmed for n=4…7 |
| Ordinary inversion lower-bound proposal on LRX₄ | Refuted: one left rotation has distance 1 and inversion count 3 |
| Cyclic-inversion proposal on LRX₄ | Lower-bound test passes all 24 states; equality fails |

The [saved LRX₈ search result](../examples/results/cayleypy-lrx8.json) contains the complete move word. That search used beam width 65,536 and depth cap 40. It establishes a verified upper bound for the pair. A smaller beam of 4,096 was inconclusive within the same search budget.

## Reproduce the checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/bootstrap.py --check

deepmath experiment examples/problems/lx-n6-diameter.json
deepmath experiment examples/problems/lrx-n8-witness.json \
  --backend cayleypy --beam-width 65536 --max-depth 40
```

The full test run used the prepared `.venv-search` environment. An environment without the optional search dependency explicitly skips its CPU integration test. Run inputs record package versions and hashes of the research runner's source files.

## Execution coverage

Model transport, role selection, feedback, and response handling were exercised with automated test fixtures. Live model requests, reasoning-model inference, and native GPU execution remain to be validated in their configured environments. The supplied LRX proof reports Lean verification; its source package has not been replayed in this workspace.
