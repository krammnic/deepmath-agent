# Validation record — 13 October 2026

Validation passed **109 research tests**, **387 native interface and model tests**, and **40 graph-library tests**, with no skips in those runs. Coverage includes real CPU search, QK artifact fitting and inference, research continuity, experiment replay, failure recovery, and evidence integrity. Validation uses Python 3.13.12 with `pip install '.[research]'`.

Installation checks exercise the complete Python API, 51 graph constructors, workspace creation, and the exact SL(2,3) diameter from outside the checkout. Native C++ and CUDA sources are included in the installed package for compilation on supported hardware.

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
| Pancake graph at n=4 | 24 states; exact diameter 4; native CPU beam path replayed |
| Binary adjacent-swap Schreier graph | Root eccentricity 3; all-roots diameter 4 |
| SL(2,3) | 24 states; exact diameter 4; native matrix beam path replayed |
| Heisenberg group modulo 3 | 27 states; native actions agree with exact integer replay |
| Modular matrix inverse edge cases | Native paths replay correctly modulo 5 and composite modulus 6 |

The [saved LRX₈ search result](../examples/results/cayleypy-lrx8.json) contains the complete move word. That search used beam width 65,536 and depth cap 40. It establishes a verified upper bound for the pair. A smaller beam of 4,096 was inconclusive within the same search budget.

## Reproduce the checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m pytest tests/native
python3 -m pytest src/cayleypy/graphs_lib_test.py

cayleypy-research experiment examples/problems/lx-n6-diameter.json
cayleypy-research experiment examples/problems/lrx-n8-witness.json \
  --backend cayleypy --beam-width 65536 --max-depth 40
```

The full test run used `.venv-integration`, with PyTorch 2.14.1, Transformers 4.57.6, and NumPy 2.5.4. Install the research extra to run inference and native Python integration tests; without these dependencies, affected tests report explicit skips. Run inputs retain source revisions, package versions, and runner source hashes.

## Execution coverage

QK fitting, artifact reload, grouped query/key capture, scoring, and hook cleanup were exercised on an actual small local transformer with random weights. This checks integration mechanics; it does not measure reasoning accuracy. No model weights were downloaded for this test.

Model transport, role selection, feedback, and response handling were exercised with automated test fixtures. Research workspace tests cover later-session context, errors, separate problem versions, independent archive verification, and preserved matrix/Schreier definitions. Live external LLM research, pretrained QK evaluation, and CUDA/NCCL execution still require validation in their configured environments. The supplied LRX proof reports Lean verification; its source package has not been replayed here.
