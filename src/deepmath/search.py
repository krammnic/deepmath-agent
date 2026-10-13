"""Permutation search with explicit evidence boundaries.

The stdlib BFS compares entire tuples, so visited-state equality has no hash
collision assumption. Optional beam backends produce replayable upper bounds;
neither a beam failure nor a QK score certifies a distance or a theorem.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import importlib
import math
import platform
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

State = tuple[int, ...]


class BackendUnavailable(RuntimeError):
    """An explicitly requested optional backend cannot run here."""


class InvalidWitnessError(RuntimeError):
    """A backend's alleged path failed independent verification."""


@dataclass(frozen=True)
class GraphSpec:
    """A finite permutation Cayley graph, restricted to its generated component.

    Every generator is a gather permutation: ``next[i] = state[generator[i]]``.
    States must have distinct labels ``0 .. n-1``. This makes each connected
    component vertex transitive, so an exhaustive root BFS determines that
    component's (possibly directed) diameter. ``expected_order`` is a checked
    assertion, never a substitute for exhaustiveness.
    """

    name: str
    generators: Mapping[str, Sequence[int]]
    expected_order: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("graph name must be nonempty")
        if not self.generators:
            raise ValueError("at least one generator is required")
        copied = {name: tuple(perm) for name, perm in self.generators.items()}
        n = len(next(iter(copied.values())))
        if n < 1:
            raise ValueError("permutations must be nonempty")
        for name, permutation in copied.items():
            if not isinstance(name, str) or not name:
                raise ValueError("generator names must be nonempty strings")
            _validate_permutation(permutation, n, "generator")
        if self.expected_order is not None:
            if type(self.expected_order) is not int or self.expected_order < 1:
                raise ValueError("expected_order must be a positive integer")
        object.__setattr__(self, "generators", MappingProxyType(copied))

    @property
    def n(self) -> int:
        return len(next(iter(self.generators.values())))

    @property
    def identity(self) -> State:
        return tuple(range(self.n))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "generators": {name: list(perm) for name, perm in self.generators.items()},
            "expected_order": self.expected_order,
            "scope": "generated_component",
        }


def _validate_permutation(values: Sequence[int], n: int, label: str) -> State:
    state = tuple(values)
    if (
        len(state) != n
        or any(type(value) is not int for value in state)
        or sorted(state) != list(range(n))
    ):
        raise ValueError(f"{label} must be a permutation of 0 through {n - 1}")
    return state


def _integer_budget(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def lrx_graph(n: int) -> GraphSpec:
    """LRX on S_n: rotate left/right and exchange positions 0 and 1.

    n=2 is supported directly, including the three coincident generators;
    upstream ``PermutationGroups.lrx`` currently requires n >= 3.
    """
    _integer_budget(n, "n", 2)
    return GraphSpec(
        name=f"lrx-{n}",
        generators={
            "L": tuple(range(1, n)) + (0,),
            "R": (n - 1,) + tuple(range(n - 1)),
            "X": (1, 0) + tuple(range(2, n)),
        },
        expected_order=math.factorial(n),
    )


def apply_move(graph: GraphSpec, state: Sequence[int], move: str) -> State:
    """Apply one named generator with validation at the public boundary."""
    checked = _validate_permutation(state, graph.n, "state")
    permutation = graph.generators[move]
    return tuple(checked[index] for index in permutation)


def replay_path(graph: GraphSpec, start: Sequence[int], path: Iterable[str]) -> State:
    """Replay a word without importing or trusting the search backend."""
    state = _validate_permutation(start, graph.n, "start")
    for move in path:
        permutation = graph.generators[move]
        state = tuple(state[index] for index in permutation)
    return state


def verify_path(
    graph: GraphSpec, start: Sequence[int], path: Iterable[str], target: Sequence[int]
) -> bool:
    try:
        checked_target = _validate_permutation(target, graph.n, "target")
        return replay_path(graph, start, path) == checked_target
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class SearchResult:
    backend: str
    graph: GraphSpec
    start: State
    target: State | None
    path: tuple[str, ...] | None
    path_verified: bool
    exact: bool
    complete: bool
    explored_states: int | None
    max_depth_reached: int | None
    layer_sizes: tuple[int, ...]
    diameter: int | None
    termination: str
    distances: Mapping[State, int] = field(default_factory=dict, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def path_found(self) -> bool:
        return self.path is not None

    @property
    def distance(self) -> int | None:
        """Exact distance for BFS, or a verified upper bound for beam search."""
        return len(self.path) if self.path is not None else None

    @property
    def shortest_path(self) -> bool:
        return self.exact and self.path_verified

    def to_dict(self, *, include_distances: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "graph": self.graph.to_dict(),
            "start": list(self.start),
            "target": list(self.target) if self.target is not None else None,
            "path": list(self.path) if self.path is not None else None,
            "path_found": self.path_found,
            "path_verified": self.path_verified,
            "path_length": self.distance,
            "shortest_path": self.shortest_path,
            "exact": self.exact,
            "complete": self.complete,
            "explored_states": self.explored_states,
            "max_depth_reached": self.max_depth_reached,
            "layer_sizes": list(self.layer_sizes),
            "diameter": self.diameter,
            "termination": self.termination,
            "metadata": dict(self.metadata),
        }
        if include_distances:
            payload["distances"] = [
                {"state": list(state), "distance": distance}
                for state, distance in self.distances.items()
            ]
        return payload


def exact_bfs(
    graph: GraphSpec,
    *,
    start: Sequence[int] | None = None,
    target: Sequence[int] | None = None,
    max_states: int = 100_000,
    max_depth: int | None = None,
) -> SearchResult:
    """Find a shortest path or enumerate a bounded exact neighborhood.

    With no target, return a farthest discovered vertex and its shortest word.
    A partial last layer is reported honestly when the hard state cap is hit.
    A depth limit may still certify completeness if every boundary edge closes
    within the visited set. Diameter is set only after all edges were checked.
    """
    _integer_budget(max_states, "max_states", 1)
    if max_depth is not None:
        _integer_budget(max_depth, "max_depth", 0)
    origin = graph.identity if start is None else _validate_permutation(start, graph.n, "start")
    goal = None if target is None else _validate_permutation(target, graph.n, "target")
    distances = {origin: 0}
    parents: dict[State, tuple[State, str]] = {}
    queue = deque([origin])
    layers = [1]
    farthest = origin
    depth_cut = False
    termination = "target_found" if origin == goal else "exhausted"
    complete = False

    if origin != goal:
        stop = False
        while queue and not stop:
            state = queue.popleft()
            depth = distances[state]
            for move, permutation in graph.generators.items():
                neighbor = tuple(state[index] for index in permutation)
                if neighbor in distances:
                    continue
                if max_depth is not None and depth >= max_depth:
                    depth_cut = True
                    continue
                if len(distances) >= max_states:
                    termination = "max_states"
                    stop = True
                    break
                next_depth = depth + 1
                distances[neighbor] = next_depth
                parents[neighbor] = (state, move)
                if next_depth == len(layers):
                    layers.append(0)
                layers[next_depth] += 1
                farthest = neighbor
                queue.append(neighbor)
                if neighbor == goal:
                    termination = "target_found"
                    stop = True
                    break
        if not stop:
            complete = not depth_cut
            termination = "exhausted" if complete else "max_depth"

    if complete and graph.expected_order is not None and len(distances) != graph.expected_order:
        raise ValueError(
            f"exhaustive component has {len(distances)} states; "
            f"expected_order asserted {graph.expected_order}"
        )

    witness = farthest if goal is None else goal
    path: tuple[str, ...] | None = None
    if witness in distances:
        backwards: list[str] = []
        cursor = witness
        while cursor != origin:
            cursor, move = parents[cursor]
            backwards.append(move)
        path = tuple(reversed(backwards))
    verified = path is not None and verify_path(graph, origin, path, witness)
    if path is not None and not verified:
        raise InvalidWitnessError("internal BFS witness failed replay")
    return SearchResult(
        backend="python_exact_bfs",
        graph=graph,
        start=origin,
        target=witness,
        path=path,
        path_verified=verified,
        exact=True,
        complete=complete,
        explored_states=len(distances),
        max_depth_reached=len(layers) - 1,
        layer_sizes=tuple(layers),
        diameter=len(layers) - 1 if complete else None,
        termination=termination,
        distances=MappingProxyType(distances),
        metadata={
            "scope": "generated_component",
            "distance_evidence": "exact_for_each_discovered_state",
            "max_states": max_states,
            "max_depth": max_depth,
            "target_requested": goal is not None,
        },
    )


def _load_optional(module: str, installation: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise BackendUnavailable(
            f"{module} is unavailable ({error}). Install {installation} in this Python environment."
        ) from error


def _cayleypy_graph(graph: GraphSpec, target: State, *, device: str, seed: int = 0, **kwargs: Any) -> Any:
    cp = _load_optional("cayleypy", "CayleyPy, e.g. pip install -e external/cayleypy")
    # An explicit graph definition also handles n=2 and arbitrary generator sets.
    definition = cp.CayleyGraphDef.create(
        generators=[list(perm) for perm in graph.generators.values()],
        generator_names=list(graph.generators),
        central_state=list(target),
        name=graph.name,
    )
    return cp.CayleyGraph(definition, device=device, random_seed=seed, verbose=0, **kwargs)


def _beam_result(
    graph: GraphSpec,
    start: State,
    target: State,
    raw: Any,
    *,
    backend: str,
    beam_width: int,
    max_depth: int,
    seed: int = 0,
) -> SearchResult:
    path = None
    if raw.path_found:
        if raw.path is None:
            raise InvalidWitnessError("backend claimed success without a path")
        moves = tuple(graph.generators)
        indices = tuple(raw.path)
        if any(type(index) is not int or not 0 <= index < len(moves) for index in indices):
            raise InvalidWitnessError("backend returned invalid generator indices")
        path = tuple(moves[index] for index in indices)
        if len(path) != raw.path_length or len(path) > max_depth:
            raise InvalidWitnessError("backend returned an inconsistent or over-budget path length")
        if not verify_path(graph, start, path, target):
            raise InvalidWitnessError("backend path did not replay to the requested target")
    metadata = {
        "beam_width": beam_width,
        "seed": seed,
        "max_depth": max_depth,
        "distance_evidence": "verified_upper_bound" if path is not None else "none",
        "failure_interpretation": "no conclusion about shortest distance or reachability",
        "upstream_metadata": getattr(raw, "native_metadata", {}),
    }
    return SearchResult(
        backend=backend,
        graph=graph,
        start=start,
        target=target,
        path=path,
        path_verified=path is not None,
        exact=False,
        complete=False,
        explored_states=None,
        max_depth_reached=len(path) if path is not None else None,
        layer_sizes=(),
        diameter=None,
        termination="path_found" if path is not None else "not_found_within_budget",
        metadata=metadata,
    )


def cayleypy_search(
    graph: GraphSpec,
    *,
    start: Sequence[int],
    target: Sequence[int] | None = None,
    beam_width: int = 1_000,
    max_depth: int = 100,
    device: str = "cpu",
    predictor: Any = None,
    use_child_scores: bool = False,
    seed: int = 0,
) -> SearchResult:
    """Run actual CayleyPy simple beam search, then independently replay it.

    An omitted predictor uses CayleyPy's Hamming heuristic. A predictor or
    per-generator Q model is a search heuristic, unrelated to LLM QK scoring.
    """
    _integer_budget(beam_width, "beam_width", 1)
    _integer_budget(max_depth, "max_depth", 0)
    _integer_budget(seed, "seed", 0)
    origin = _validate_permutation(start, graph.n, "start")
    goal = graph.identity if target is None else _validate_permutation(target, graph.n, "target")
    upstream_graph = _cayleypy_graph(graph, goal, device=device, seed=seed)
    raw = upstream_graph.beam_search(
        start_state=list(origin),
        beam_mode="simple",
        predictor=predictor,
        beam_width=beam_width,
        max_steps=max_depth,
        return_path=True,
        use_child_scores=use_child_scores,
    )
    return _beam_result(
        graph, origin, goal, raw,
        backend="cayleypy_beam", beam_width=beam_width, max_depth=max_depth, seed=seed,
    )


def multigpu_search(
    graph: GraphSpec,
    *,
    start: Sequence[int],
    target: Sequence[int] | None = None,
    beam_width: int = 100_000,
    max_depth: int = 100,
    num_gpus: int = 2,
    predictor: Any = None,
    use_child_scores: bool = False,
    native_options: Any = None,
    seed: int = 0,
) -> SearchResult:
    """Run MultiGPUBeamSearch's public, strict native dispatch (no fallback).

    This requires Linux, the requested NVIDIA GPUs, a CUDA-enabled PyTorch,
    CUDA toolkit/nvcc, NCCL and the upstream C++ build prerequisites. Upstream
    may fetch checksum-pinned CUTLASS and compile a runner on first use; pass
    prepared ``NativeOptions`` to control source, cache and timeout locations.
    Importing this module does not load PyTorch, fetch sources or build code.
    """
    _integer_budget(beam_width, "beam_width", 1)
    _integer_budget(max_depth, "max_depth", 0)
    _integer_budget(num_gpus, "num_gpus", 1)
    _integer_budget(seed, "seed", 0)
    origin = _validate_permutation(start, graph.n, "start")
    goal = graph.identity if target is None else _validate_permutation(target, graph.n, "target")
    if platform.system() != "Linux":
        raise BackendUnavailable("MultiGPUBeamSearch native execution requires Linux and NVIDIA CUDA GPUs")
    torch = _load_optional("torch", "CUDA-enabled PyTorch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise BackendUnavailable(f"MultiGPUBeamSearch requires {num_gpus} available NVIDIA CUDA GPU(s)")
    native = _load_optional(
        "multigpubeamsearch",
        "pip install -e external/MultiGPUBeamSearch/integrations/cayleypy_native",
    )
    if native_options is None:
        native_options = native.NativeOptions(num_gpus=num_gpus)
    else:
        if not isinstance(native_options, native.NativeOptions):
            raise TypeError("native_options must be multigpubeamsearch.NativeOptions")
        configured_count = (
            len(native_options.devices) if native_options.devices is not None else native_options.num_gpus
        )
        if configured_count != num_gpus:
            raise ValueError("native_options must select exactly num_gpus devices")
    upstream_graph = _cayleypy_graph(graph, goal, device="cuda", num_gpus=num_gpus, seed=seed)
    raw = native.beam_search(
        upstream_graph,
        backend="native",
        native_options=native_options,
        start_state=list(origin),
        beam_mode="simple",
        predictor=predictor,
        beam_width=beam_width,
        max_steps=max_depth,
        return_path=True,
        use_child_scores=use_child_scores,
    )
    return _beam_result(
        graph, origin, goal, raw,
        backend="multigpu_native_beam", beam_width=beam_width, max_depth=max_depth, seed=seed,
    )
