# LRX: sharp lower bound and exact witness

The research supplement *A sharp lower bound for the LRX diameter*, dated 25 September 2026, presents a proof developed through collaborating AI agents, together with a matching constructive witness and reported Lean verification. Its filename, fingerprint, and source details are recorded in the [source manifest](../references/sources.json).

## The result

For \(n\geq4\), the vertices of \(LRX_n\) are permutations of \(0,\ldots,n-1\). Each of the three generators costs one move:

\[
L(x_0,x_1,\ldots,x_{n-1})=(x_1,\ldots,x_{n-1},x_0),\qquad R=L^{-1},
\]
\[
X(x_0,x_1,x_2,\ldots)=(x_1,x_0,x_2,\ldots).
\]

For the identity and reflected permutation

\[
e=(0,1,\ldots,n-1),\qquad p=(1,0,n-1,n-2,\ldots,2),
\]

the supplement proves

\[
d(e,p)=\binom n2,\qquad \operatorname{diam}(LRX_n)\geq\binom n2.
\]

The exact witness supplies the matching path for this pair. The universal upper bound \(\operatorname{diam}(LRX_n)\leq\binom n2\) remains the open question in the supplement and the cited [research roadmap](https://github.com/TryDotAtwo/AutoMathLab/blob/0d3880fdb94851c026a7362857b9edc425ccd881/projects/lrx/ROADMAP.md). The theorem begins at \(n=4\); the diameters at \(n=2\) and \(n=3\) are 1 and 2, respectively.

## Constructing the witness

Section 15 gives a recurrence for a word reaching the reflected permutation. Words execute from left to right:

\[
E_0=X,\qquad E_{k+1}=XL E_k RX;
\]
\[
W_2=E_0,\quad W_3=E_1,\quad W_j=E_{j-2}L W_{j-2}\quad(j\geq4),\qquad o_j=\lfloor j/2\rfloor-1.
\]

Set \(a=\lfloor n/2\rfloor\) and \(b=n-a\). Choose \(T=L^a\) when \(o_a=o_b\), and \(T=R^a\) otherwise. The final word is

\[
W_a T W_b^{-1} RR.
\]

To invert a word, reverse its letters and exchange \(L\) with \(R\). Including the final two rotations, the construction uses exactly \(n(n-1)/2\) moves. Together with the lower-bound argument, it establishes the exact distance.

Construct and replay the witness at \(n=8\):

```sh
deepmath experiment examples/problems/lrx-n8-witness.json --kind witness
```

The resulting path reaches `(1,0,7,6,5,4,3,2)` in 28 moves.

## Verification and attribution

Sections 16 and 19 report compilation of 30 local Lean modules and the submission entry point with Lean 4.34.0, with an axiom audit free of `sorryAx`. The [local checks](../tests/test_lrx.py) independently replay the constructive word for \(4\leq n\leq100\), check its length and relabeling, and establish exact witness distances by breadth-first search for \(4\leq n\leq7\).

The supplement credits Astra Ultra material for the structural argument and reflection construction; Researcher A for structural and interval formalization; Researcher B for folded-coordinate transport and exact-witness composition; LRX_Math_Savant for the graph bridge; and the research team for review and coordination. The [source manifest](../references/sources.json) records the supplement's fingerprint and the compiler and source versions supporting its verification report.
