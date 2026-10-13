# Research workflow

A session begins with a precise conjecture: the graph, allowed moves, index convention, quantifiers, and any expected finite values. The researcher chooses the models and compute budget before the agents begin. These choices stay fixed throughout the session.

## Develop competing ideas

Three perspectives work in parallel. The theorist seeks structural lemmas and constructive arguments. The counterexample hunter challenges assumptions and looks for exceptional cases. The geometric researcher investigates representations that might explain distances or suggest useful bounds.

Each proposal states its claim, gives a rationale, and requests a small number of experiments. Reasoning scores help choose which proposals to investigate. When several ideas are selected, one exploration slot rotates through alternatives so that a low initial score does not close off a promising direction.

## Test consequences

Experiments can enumerate a finite graph, search for a path, replay a proposed word, construct the reflected LRX witness, or compare a candidate area statistic with exact distances. They run within the session's state, depth, and search-width limits.

Returned paths are checked move by move. Exhaustive search establishes the diameter of the generated Cayley graph using vertex transitivity. A bounded search records its explored range, and a failed heuristic search remains inconclusive. Area probes return concrete counterexamples when a proposed inequality fails.

The automated experiment domain is finite permutation graphs. LRX includes left and right rotation and exchange of the first two entries; directed LX omits right rotation. Custom generators use the convention `next[i] = state[generator[i]]`. Matrix groups and more general graph actions require additional experiment definitions.

## Critique and refine

A critic compares the arguments with the observed results, looking for unsupported steps, missed cases, and changes in quantifiers. A synthesizer turns that review into the next research direction and a list of proof obligations. Both the observations and critique return to the proposal agents in the following round.

Finite agreement supports further investigation. A general proof must address every case in its statement. Formal verification requires a concrete theorem, its source and dependencies, and a successful checker run. Sessions preserve proposed proofs and missing lemmas for that work; they do not automatically run a formal prover.

## Preserve and continue

Every session records inputs, model identities, usage, scores, experiments, and reviews. Checkpoints retain completed work if a later call fails. The report distinguishes successful completion, partial errors, and failed sessions, with links to the underlying evidence.

Use `verify-run` to check artifact integrity and `replay` to repeat the recorded experiments. A replay compares the evidence produced, including verified path lengths. Export creates a local review package containing the experiment record, review request, and handoff.

To continue, inspect the findings and unresolved obligations, revise the problem or budget, and start a new session. Preserve the previous run as the basis for comparison. For reproducible accelerator work, keep the runtime environment and hardware details alongside the research record.
