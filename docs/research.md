# Research program

CayleyPy Research explores open conjectures through a cycle of proposed arguments, exact experiments, counterexample search, and proof. Agents develop competing ideas, review the evidence, and refine the next experiment. QK scores help choose which ideas deserve closer examination; mathematical claims are settled by proofs and verified computations.

## Cayley graph conjectures

[CayleyPy Growth](https://arxiv.org/html/2509.19162v2), Sections 3.2, 3.12, and 4.1, provides starting questions on diameters and growth functions. Conjecture 11 predicts the directed LX diameter

\[
D_{LX}(n)=\begin{cases}(3n^2-8n+9)/4,&n\text{ odd},\\(3n^2-8n+12)/4,&n\text{ even}.\end{cases}
\]

Exact traversal can settle individual sizes and expose exceptions to a proposed formula. Search then supplies paths and difficult examples that can suggest a general construction. LX allows a left cyclic shift and a first-pair swap; LRX also allows the inverse shift. Each generator costs one move, so the two graphs have distinct distance functions.

The [LRX lower bound and exact reflected witness](lrx-case-study.md) provide a concrete result of agent-assisted research. The remaining diameter question asks for a path of at most \(\binom n2\) moves from the identity to every permutation, for every \(n\geq4\). A constructive proof must account for paid rotations, final orientation, termination, and all cases.

## Geometric representations

[CayleyPy-4: AI-Holography](https://arxiv.org/html/2603.22195v1), Section 2, Proposition 1, and Section 5, relates adjacent-swap distances to lattice-path areas. These representations suggest useful statistics for search and candidate inequalities for proof. Each proposed transfer to LRX must specify the encoding, the allowed moves, and how the statistic changes under those moves.

Small exact examples are particularly useful here. Section 6.1.3, Example 3, of the holography paper gives wrapped binary adjacent-swap distance 2 from `000111` to `101010`, while the minimum inversion area over cyclic shifts is 3. The path

```text
000111 → 100110 → 101010
```

uses swaps `(0,5)` and `(2,3)`. This counterexample rules out the naive wrapped-area equality and helps guide more precise formulations.

## Questions to pursue

- **LRX diameter:** turn short paths and structural patterns into an algorithm with a uniform cost bound.
- **LX diameter:** compare exact directed distances with Conjecture 11 and investigate any discrepancies.
- **Geometric bounds:** test area and projection inequalities on exact instances, then prove the surviving statements from their behavior under generators.

Every experiment should make its conclusion clear. A replayed path gives an upper bound for its endpoints; exhaustive traversal gives an exact finite diameter; a counterexample refutes the precise statement it violates. QK scores prioritize further work, while a general theorem requires an argument covering every value in its stated domain.

Paper versions and source references are recorded in the [research references](../references/sources.json).
