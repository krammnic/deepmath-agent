"""Finite tests of explicit area-inspired statistics against exact distances.

The statistics here are experimental transfers to permutation Cayley graphs,
including LRX. They are not the holography paper's area construction or a
theorem from that paper. Each probe tests two precisely stated hypotheses:
``H(s) <= d(e,s)`` and ``H(s) == d(e,s)`` on discovered states at one finite n.
Full enumeration covers the given generated component only; partial BFS still
supplies exact distances to discovered states and can yield counterexamples.
"""

from __future__ import annotations

from typing import Any

from .search import GraphSpec, InvalidWitnessError, State, exact_bfs, verify_path

MAX_PROBE_N = 16
MAX_PROBE_STATES = 100_000


def _inversions(state: State) -> int:
    """Ordinary inversion count, an area-inspired permutation statistic."""
    return sum(left > right for i, left in enumerate(state) for right in state[i + 1:])


def _cyclic_inversions(state: State) -> int:
    """Minimum ordinary inversion count over all cyclic position rotations.

    Rotating first label a to the last position changes inversions by
    ``n - 1 - 2*a`` for a permutation of ``0..n-1``. This recurrence avoids
    recomputing n separate quadratic inversion counts.
    """
    current = best = _inversions(state)
    n = len(state)
    for first in state[:-1]:
        current += n - 1 - 2 * first
        best = min(best, current)
    return best


def probe_area(
    graph: GraphSpec,
    *,
    statistic: str = "inversions",
    max_states: int = MAX_PROBE_STATES,
    max_depth: int = 100,
) -> dict[str, Any]:
    """Compare a named candidate statistic with independent exact BFS distances.

    The returned JSON-safe record reports both the candidate lower bound and
    equality hypothesis. For each failed relation it provides a violation at
    minimum discovered distance, together with its shortest generator word and
    independent word replay. Passing a finite test never establishes a family
    theorem. A bounded traversal's ``status='completed'`` means the probe ran;
    ``complete`` separately indicates whether the component was exhausted.

    Limits of n<=16 and 100,000 discovered states keep this diagnostic bounded.
    Computing either statistic costs O(n^2) per state. Counterexample word
    reconstruction may run at most two additional BFS traversals with the same
    bounds (one when both relations share a counterexample).
    """
    if not isinstance(graph, GraphSpec):
        raise TypeError("graph must be a deepmath.search.GraphSpec")
    if graph.n > MAX_PROBE_N:
        raise ValueError(f"Area probes support n <= {MAX_PROBE_N}; choose a smaller finite instance")
    if statistic not in {"inversions", "cyclic_inversions"}:
        raise ValueError("statistic must be 'inversions' or 'cyclic_inversions'")
    if type(max_states) is not int or not 1 <= max_states <= MAX_PROBE_STATES:
        raise ValueError(f"Area probe max_states must be an integer from 1 to {MAX_PROBE_STATES}")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("Area probe max_depth must be a nonnegative integer")

    evaluate = _inversions if statistic == "inversions" else _cyclic_inversions
    baseline = evaluate(graph.identity)
    if baseline != 0:
        raise ValueError("The candidate statistic must be zero at the identity")
    search = exact_bfs(graph, max_states=max_states, max_depth=max_depth)
    failures: dict[str, list[tuple[State, int, int]]] = {"lower_bound": [], "equality": []}
    max_statistic = 0
    # Keep only the minimum-distance example while counting all violations.
    counts = {name: 0 for name in failures}
    equal_count = 0
    for state, distance in search.distances.items():
        value = evaluate(state)
        max_statistic = max(max_statistic, value)
        equal_count += value == distance
        for name, failed in (("lower_bound", value > distance), ("equality", value != distance)):
            if failed:
                counts[name] += 1
                if not failures[name] or distance < failures[name][0][1]:
                    failures[name] = [(state, distance, value)]

    certificates: dict[State, dict[str, Any]] = {}

    def counterexample(name: str) -> dict[str, Any] | None:
        if not failures[name]:
            return None
        state, distance, value = failures[name][0]
        if state not in certificates:
            witness = exact_bfs(
                graph, target=state, max_states=max_states, max_depth=max_depth,
            )
            word = witness.path
            valid = word is not None and verify_path(graph, graph.identity, word, state)
            if not valid or not witness.shortest_path or witness.distance != distance:
                raise InvalidWitnessError("Area-probe counterexample shortest word failed independent replay")
            certificates[state] = {
                "state": list(state),
                "statistic_value": value,
                "exact_distance": distance,
                "shortest_word": list(word),
                "path_verified": True,
                "shortest_path": True,
                "distance_evidence": "python_exact_bfs",
                "minimality_scope": "minimum distance among discovered violating states",
            }
        return dict(certificates[state])

    relations = {}
    for name, statement in (("lower_bound", "H(s) <= d(e,s)"), ("equality", "H(s) == d(e,s)")):
        holds = counts[name] == 0
        relations[name] = {
            "statement": statement,
            "holds_on_tested_states": holds,
            "holds_on_generated_component": holds if search.complete else None,
            "outcome": (
                "counterexample_found" if not holds else
                "holds_on_generated_component" if search.complete else
                "no_counterexample_in_discovered_states"
            ),
            "violations": counts[name],
            "counterexample": counterexample(name),
        }

    return {
        "status": "completed",
        "evidence": "finite_heuristic_probe",
        "graph": graph.to_dict(),
        "statistic": statistic,
        "formula": (
            "H(s) = #{(i,j): 0 <= i < j < n and s[i] > s[j]}" if statistic == "inversions" else
            "H(s) = min_{0 <= k < n} inversions(rotate_left(s,k))"
        ),
        "identity_value": baseline,
        "interpretation": (
            "Experimental area-inspired permutation statistic; an LRX transfer hypothesis, "
            "not the holography paper's area construction or theorem."
        ),
        "scope": "generated_component" if search.complete else "discovered_states",
        "universal_claim": "not_established",
        "complete": search.complete,
        "explored_states": search.explored_states,
        "termination": search.termination,
        "max_statistic": max_statistic,
        "equality_states": equal_count,
        "budgets": {"max_states": max_states, "max_depth": max_depth},
        "search": search.to_dict(),
        **relations,
    }
