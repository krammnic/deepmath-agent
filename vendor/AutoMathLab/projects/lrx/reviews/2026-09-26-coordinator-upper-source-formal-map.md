# U19 → Lean: source-to-obligation map

**ID:** 2026-09-26-coordinator-upper-source-formal-map
**Author:** Codex coordinator (operator: Aleksei Makin).
**Date:** 26 September 2026.
**Claim:** partial. **Verification:** author-checked locator/source audit; no new kernel replay or full mathematical review.

The reviewed source is Sergey's 19 September manuscript, SHA-256 `8547daf9560d5e022f9b5234b3ab72f661e873d14e84eadaa703b66f03336d9d`, together with A's explicit §4/§7.2 patch. The source and patch are preserved externally and not republished here. Historical positive A/B reviews refer to this corrected chain, not every unchanged sentence. The previously published submission v2 records these distinctions.

The scientific claim and its complete Lean formalization are different statuses. This map identifies where existing modules fit and where the whole source obligation has not yet been discharged. A missing formal interface does not by itself refute the manuscript.

| Map node | Exact source locator | Existing formal component | Remaining obligation |
|---|---|---|---|
| MODEL | §1.1, §13; lines 39–74 | Represents / canonical / step_L / step_R / step_X | Local representation transitions exist; final graph-wide upper composition is open. |
| LIFT | §2.1, Lemma2; lines 322–406 | exists_majorized_short_lift (PR7); periodic extension A2883 | Energy minimum is not C-minimality. A2884 verifies the scoped lift; every downstream use must need only its proved properties. |
| PLAN | §3, Lemmas4–5; lines 505–577 | No complete interface pinned in this audit | Need constructible reachability/root data for every input; SCC structure must not become an unexplained assumption. |
| EXEC | §4, Lemma6 + A patch; lines 578–647 | cycle_service / ServiceTree.realize | Compiler has explicit Valid/target hypotheses. Existence, recursive selection and all-n budget are separate. A patch sums subtree winding and marks cycles before recursion. |
| COST | §5, Lemmas7–8; lines 648–762 | No complete bridge pinned in this audit | Do not count the scalar certificate as proof of the geometry producing its input. |
| COST | §6, connected branch; lines 763–874 | Conditional service baseline only | All cursor cases, chosen positive ports and actual service costs need common witnesses. |
| PLAN/COST | §7.1–7.2, Lemma10 + A patch; lines 875–942 | No full rank-transfer theorem pinned | Positive incident service ports required by the correction. External approaches may cross the cut without X; raw zero-port bound is not used. |
| PLAN/COST | §8–§10; lines 986–1386 | Local cutoff/tour lemmas do not cover this whole chain | Separate structural classification, rank moves, paid route and exceptional case coverage. |
| COST | §11.3, Lemma17; lines 1508–1584 | component_cutoff / half_size_cutoff / size_cutoff | Proves scalar implications t≤12, m≤1379, n≤2759 under explicit inequalities. Deriving them from LRX remains an obligation. |
| CAP | §12; lines 1585–1763 | tours16 / tours20; B cap1/2 author package2907 | Finite tours are scoped results. B538-module package has only partial independent replay2909; Admissible/ExternalRealization geometry remains open. |
| SMALL | §13 + §14 + §15; lines 1764–1779 | No all-small-n kernel theorem established by this locator audit | Source explicitly uses BFS for4…10. Its printed output is source evidence, not a new replay or a Lean theorem. |
| U-ALL/EQ | §13; lines 1764–1779 | No full unconditional Lean upper/equality pinned | Phase is part of the joint construction. Do not add an unpaid final rotation or infer all-input upper from the reflected witness. |

## Decisive integration conditions

1. The source selects token positions and the final physical cursor together. The lift, inverse-permutation convention and paid route must refer to the same target. Do not choose their witnesses independently.
2. `ServiceTree.realize` consumes `tree.Valid`; it does not construct such a tree from every permutation. Root/SCC geometry, recursive progress and budget are separate dependencies.
3. `UpperFiniteCutoff` explicitly assumes the scalar inequalities being bounded. `UpperExceptionalTours` verifies finite tour predicates. Neither supplies the geometric necessity of those assumptions.
4. Source §13 explicitly splits n=4…10 (BFS) from n≥11 (general construction). Hence the current SMALL obligation is now tied to this exact source, rather than guessed from an old dashboard. This audit does not rerun BFS or certify all small n in Lean.
5. The source's sole cited external structural result concerns shallow cycles. Its translation into the chosen formal model must be included or replaced by an internally proved equivalent; this audit does not certify that external reference anew.

## Existing checks and versions

PR7 contains the coordinator lift theorem; A2883 supplies its periodic extension and TL2884 reports independent replay. PR13 baseline has coordinator fresh replay in commit8ebba577 (21 modules plus fresh BlockExchange); later forest/residual modules are not part of that replay. B2907 reports ScalarCapChecked1/2 kernel success; TL2909 reports only a partial independent replay. This audit does not upgrade those statuses.

## Reproduce the locator checks

Run `python3 verify_locators.py --source /path/to/lrx_proof_human_readable_2026-09-19_RU.md --upper-root /path/to/PR13/source/formal --lift /path/to/UpperLiftSelection.lean` from the adjacent directory. Exact source and module hashes must match. PASS means file identity and locator range integrity only, not validity of the mathematical proof. No foreign manuscript files, binaries or credentials are included.

Next: A handles the exact lift/native phase interface; teamlead constructs PLAN; B's certificate receives a full independent replay and then a separate geometry-to-Admissible bridge. Coordinator owns source coverage and the small-n/integration packet in Issue16. The U-SOURCE obligation remains partial until every required source claim has a checked formal interface or an explicit unformalized entry.
