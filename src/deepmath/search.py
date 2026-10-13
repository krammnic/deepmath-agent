"""Native graph integration and search with explicit evidence boundaries.

The stdlib BFS compares entire tuples, so visited-state equality has no hash
collision assumption. Optional beam backends produce replayable upper bounds;
neither a beam failure nor a QK score certifies a distance or a theorem.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import importlib
import inspect
from functools import cached_property
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
    """An explicit gather or modular matrix action with independent replay.

    Gather actions may have repeated labels (Schreier graphs). Such actions
    need all-root searches for a diameter: root eccentricity is insufficient.
    Matrix actions use exact Python arithmetic and left multiplication modulo
    a common modulus; generators must be invertible over that finite ring.
    """

    name: str
    generators: Mapping[str, Sequence]
    expected_order: int | None = None
    central_state: Sequence[int] | None = None
    action: str = "permutation"
    modulo: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("graph name must be nonempty")
        if not self.generators or len(self.generators) > 4096:
            raise ValueError("graph needs 1..4096 generators")
        if any(not isinstance(name, str) or not name for name in self.generators):
            raise ValueError("generator names must be nonempty strings")
        if self.action == "permutation":
            copied = {name: tuple(perm) for name, perm in self.generators.items()}
            n = len(next(iter(copied.values())))
            if not 1 <= n <= 4096:
                raise ValueError("permutation state size must be in 1..4096")
            for perm in copied.values():
                _validate_permutation(perm, n, "generator")
            center = tuple(range(n)) if self.central_state is None else tuple(self.central_state)
            if len(center) != n or any(type(x) is not int or not 0 <= x < n for x in center):
                raise ValueError("central_state must have n integer labels in 0..n-1")
            if self.modulo is not None:
                raise ValueError("modulo only applies to matrix actions")
        elif self.action == "matrix":
            if type(self.modulo) is not int or not 2 <= self.modulo <= 2**31:
                raise ValueError("matrix actions require modulo in 2..2**31")
            copied = {name: tuple(tuple(row) for row in matrix) for name, matrix in self.generators.items()}
            dim = len(next(iter(copied.values())))
            if not 1 <= dim <= 32:
                raise ValueError("matrix dimension must be in 1..32")
            for matrix in copied.values():
                if len(matrix) != dim or any(len(row) != dim for row in matrix):
                    raise ValueError("matrix generators must be equally sized square matrices")
                if any(type(x) is not int for row in matrix for x in row):
                    raise ValueError("matrix entries must be integers")
                if math.gcd(_determinant(matrix), self.modulo) != 1:
                    raise ValueError("matrix generators must be invertible modulo the modulus")
            copied = {name: tuple(tuple(x % self.modulo for x in row) for row in matrix)
                      for name, matrix in copied.items()}
            center = (tuple(int(i == j) for i in range(dim) for j in range(dim))
                      if self.central_state is None else tuple(self.central_state))
            if not center or len(center) % dim or len(center) > 4096:
                raise ValueError("matrix central_state length must be a positive multiple of the matrix dimension")
            if any(type(x) is not int or not 0 <= x < self.modulo for x in center):
                raise ValueError("matrix state entries must be reduced residues modulo the modulus")
        else:
            raise ValueError("action must be permutation or matrix")
        if self.expected_order is not None:
            if type(self.expected_order) is not int or self.expected_order < 1:
                raise ValueError("expected_order must be a positive integer")
        object.__setattr__(self, "central_state", center)
        object.__setattr__(self, "generators", MappingProxyType(copied))

    @property
    def n(self) -> int:
        return len(self.central_state)

    @property
    def identity(self) -> State:
        """The designated root (not necessarily a group identity)."""
        return self.central_state

    @property
    def matrix_dimension(self) -> int:
        return len(next(iter(self.generators.values())))

    @property
    def vertex_transitive(self) -> bool:
        # A free group action is regular on each orbit. Repeated-label and
        # singular/rectangular matrix actions need independent all-root checks.
        return self.is_regular_state(self.identity)

    def is_regular_state(self, state: State) -> bool:
        if self.action == "permutation":
            return len(set(state)) == self.n
        dim = self.matrix_dimension
        return (self.n == dim * dim and math.gcd(_determinant(
            tuple(state[i*dim:(i+1)*dim] for i in range(dim))), self.modulo) == 1)

    def validate_state(self, values: Sequence[int], label: str = "state") -> State:
        state = tuple(values)
        if len(state) != self.n or any(type(x) is not int for x in state):
            raise ValueError(f"{label} must contain {self.n} integer entries")
        if self.action == "permutation":
            if Counter(state) != Counter(self.identity):
                raise ValueError(f"{label} must preserve the central_state label multiset")
        elif any(not 0 <= x < self.modulo for x in state):
            raise ValueError(f"{label} entries must be reduced residues modulo the modulus")
        return state

    def step(self, state: State, generator: Sequence) -> State:
        if self.action == "permutation":
            return tuple(state[index] for index in generator)
        dim = self.matrix_dimension
        columns = self.n // dim
        return tuple(sum(generator[i][k] * state[k * columns + j] for k in range(dim)) % self.modulo
                     for i in range(dim) for j in range(columns))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "n": self.n, "action": self.action,
            "generators": {name: (list(gen) if self.action == "permutation" else [list(row) for row in gen])
                           for name, gen in self.generators.items()},
            "central_state": list(self.identity), "modulo": self.modulo,
            "expected_order": self.expected_order, "vertex_transitive": self.vertex_transitive,
            "scope": "generated_component",
        }


def _determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Fraction-free elimination, avoiding floating-point invertibility tests."""
    rows = [list(row) for row in matrix]
    sign, previous = 1, 1
    for i in range(len(rows) - 1):
        if rows[i][i] == 0:
            pivot = next((j for j in range(i + 1, len(rows)) if rows[j][i]), None)
            if pivot is None:
                return 0
            rows[i], rows[pivot] = rows[pivot], rows[i]
            sign = -sign
        value = rows[i][i]
        for j in range(i + 1, len(rows)):
            for k in range(i + 1, len(rows)):
                rows[j][k] = (rows[j][k] * value - rows[j][i] * rows[i][k]) // previous
            rows[j][i] = 0
        previous = value
    return sign * rows[-1][-1]



def _modular_inverse(matrix: Sequence[Sequence[int]], modulo: int) -> tuple[tuple[int, ...], ...]:
    """Invert over Z/mZ, including composite moduli with no unit pivot entry.

    Bezout row combinations create a unit pivot when elementary row swaps
    alone are insufficient. All arithmetic is exact and reduced modulo m.
    """
    dim = len(matrix)
    rows = [[x % modulo for x in row] + [int(i == j) for j in range(dim)]
            for i, row in enumerate(matrix)]

    def bezout(a: int, b: int) -> tuple[int, int, int]:
        old_r, r, old_x, x, old_y, y = a, b, 1, 0, 0, 1
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_x, x = x, old_x - quotient * x
            old_y, y = y, old_y - quotient * y
        return old_r, old_x, old_y

    for column in range(dim):
        for other in range(column + 1, dim):
            if math.gcd(rows[column][column], modulo) == 1:
                break
            a, b = rows[column][column], rows[other][column]
            if not b:
                continue
            divisor, x, y = bezout(a, b)
            first, second = rows[column][:], rows[other][:]
            rows[column] = [(x * u + y * v) % modulo for u, v in zip(first, second)]
            rows[other] = [((-b // divisor) * u + (a // divisor) * v) % modulo
                           for u, v in zip(first, second)]
        if math.gcd(rows[column][column], modulo) != 1:
            raise ValueError("matrix is not invertible modulo the modulus")
        inverse = pow(rows[column][column], -1, modulo)
        rows[column] = [(x * inverse) % modulo for x in rows[column]]
        for other in range(dim):
            if other == column:
                continue
            factor = rows[other][column]
            rows[other] = [(u - factor * v) % modulo for u, v in zip(rows[other], rows[column])]
    return tuple(tuple(row[dim:]) for row in rows)

def _catalog_functions(cp: Any) -> dict[str, Any]:
    functions = {"prepare_graph": cp.prepare_graph}
    for group in ("PermutationGroups", "MatrixGroups", "Puzzles"):
        for name, value in vars(getattr(cp, group)).items():
            if not name.startswith("_") and isinstance(value, staticmethod):
                functions[f"{group}.{name}"] = getattr(getattr(cp, group), name)
    functions["GapPuzzles.puzzle"] = cp.GapPuzzles.puzzle
    return functions


def graph_catalog() -> list[dict[str, Any]]:
    """List the installed, reviewed native graph factories and their signatures."""
    cp = _load_optional("cayleypy", "python -m pip install '.[research]'")
    return [{"constructor": name, "signature": str(inspect.signature(factory)),
             "description": (inspect.getdoc(factory) or "").split("\n")[0]}
            for name, factory in sorted(_catalog_functions(cp).items())]


def graph_from_cayleypy(definition: Any, *, expected_order: int | None = None) -> GraphSpec:
    """Translate a native CayleyGraphDef to backend-independent exact actions."""
    names = definition.generator_names
    if len(set(names)) != len(names):
        raise ValueError("generator names must be unique for unambiguous replay")
    if definition.is_permutation_group():
        generators = {name: tuple(int(x) for x in gen) for name, gen in zip(names, definition.generators)}
        return GraphSpec(definition.name or "cayleypy", generators, expected_order,
                         tuple(int(x) for x in definition.central_state))
    moduli = {gen.modulo for gen in definition.generators}
    if len(moduli) != 1 or not next(iter(moduli)):
        raise ValueError("verified matrix experiments require a common finite modulo >=2; use the native API for unbounded integer groups")
    generators = {name: tuple(tuple(int(x) for x in row) for row in gen.matrix)
                  for name, gen in zip(names, definition.generators)}
    return GraphSpec(definition.name or "cayleypy-matrix", generators, expected_order,
                     tuple(int(x) for x in definition.central_state), "matrix", next(iter(moduli)))


def _flat_state(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("state must be an array of integers or matrix rows")
    return tuple(x for row in value for x in row) if value and isinstance(value[0], (list, tuple)) else tuple(value)


def graph_from_config(config: Mapping[str, Any], *, name: str = "custom") -> GraphSpec:
    """Resolve JSON graph definitions without evaluating code or import paths.

    Persisted ``resolved_definition`` freezes catalog/random constructor output
    so research reruns use exactly the same generators and designated root.
    """
    family = config.get("family")
    if family in {"lrx", "lx"}:
        n = config.get("n")
        if type(n) is not int or not 2 <= n <= 256:
            raise ValueError("graph.n must be an integer in 2..256")
        graph = lrx_graph(n)
        return graph if family == "lrx" else GraphSpec(f"lx-{n}", {k: v for k, v in graph.generators.items() if k != "R"}, graph.expected_order)
    if family in {"permutation", "matrix"}:
        generators = config.get("generators")
        if not isinstance(generators, dict):
            raise ValueError("generators must be an object of named generators")
        graph = GraphSpec(config.get("name", name), generators, config.get("expected_order"),
                         _flat_state(config.get("central_state")), family, config.get("modulo"))
        if config.get("n", graph.n) != graph.n:
            raise ValueError("graph.n must equal the flattened state size")
        return graph
    if family != "cayleypy":
        raise ValueError("graph.family must be lrx, lx, permutation, matrix, or cayleypy")
    if "resolved_definition" in config:
        resolved = config["resolved_definition"]
        if not isinstance(resolved, dict) or resolved.get("family") not in {"permutation", "matrix"}:
            raise ValueError("resolved_definition must be an explicit permutation or matrix graph")
        return graph_from_config(resolved, name=name)
    cp = _load_optional("cayleypy", "python -m pip install '.[research]'")
    constructor = config.get("constructor")
    factories = _catalog_functions(cp)
    if not isinstance(constructor, str) or constructor not in factories:
        raise ValueError("unknown CayleyPy constructor; list available constructors with deepmath graphs")
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, dict) or any(not isinstance(k, str) or k.startswith("_") for k in kwargs):
        raise ValueError("constructor kwargs must be a JSON object with public parameter names")
    if constructor.startswith("MatrixGroups.") and (type(kwargs.get("modulo")) is not int or kwargs["modulo"] < 2):
        raise ValueError("matrix research experiments require explicit modulo >=2")
    if constructor == "GapPuzzles.puzzle" and kwargs.get("puzzle_name") not in cp.GapPuzzles.list_puzzles():
        raise ValueError("puzzle_name must be a name from the bundled GapPuzzles catalog")
    kwargs = dict(kwargs)
    if constructor == "PermutationGroups.conjugacy_classes" or (constructor == "prepare_graph" and kwargs.get("name") == "conjugacy_class"):
        # JSON cannot represent tuple-keyed dictionaries. Accept native cycle
        # partitions as [{"partition": [3, 2], "count": null}, ...].
        classes = kwargs.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("conjugacy classes must be a list of partition/count objects")
        converted = {}
        for entry in classes:
            if not isinstance(entry, dict) or set(entry) != {"partition", "count"}:
                raise ValueError("each conjugacy class needs partition and count")
            partition, count = entry["partition"], entry["count"]
            if not isinstance(partition, list) or not partition or any(type(x) is not int or x < 1 for x in partition):
                raise ValueError("partition must contain positive integer cycle lengths")
            if count is not None and (type(count) is not int or not 1 <= count <= 4096):
                raise ValueError("count must be null or an integer in 1..4096")
            key = tuple(sorted(partition))
            if key in converted:
                raise ValueError("conjugacy partitions must be distinct")
            converted[key] = count
        kwargs["classes"] = converted
    try:
        # This pinned prepare_graph version routes 'lx-N' to LRX by mistake.
        # Resolve the named LX family explicitly so its directed moves survive.
        if constructor == "prepare_graph" and str(kwargs.get("name", "")).startswith("lx-"):
            definition = cp.PermutationGroups.lx(int(kwargs["name"][3:]))
        else:
            definition = factories[constructor](**kwargs)
        if config.get("central_state") is not None:
            center = _flat_state(config["central_state"])
            if any(type(x) is not int for x in center):
                raise ValueError("central_state entries must be integers")
            definition = definition.with_central_state(list(center))
        return graph_from_cayleypy(definition, expected_order=config.get("expected_order"))
    except (AssertionError, TypeError, KeyError) as error:
        raise ValueError(f"invalid {constructor} parameters: {error}") from error


def normalize_graph_config(config: dict, *, name: str) -> dict:
    graph = graph_from_config(config, name=name)
    if config.get("family") == "cayleypy":
        resolved = graph.to_dict()
        resolved["family"] = graph.action
        return {**config, "n": graph.n, "resolved_definition": resolved}
    return {**config, "n": graph.n}


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
    checked = graph.validate_state(state)
    permutation = graph.generators[move]
    return graph.step(checked, permutation)


def replay_path(graph: GraphSpec, start: Sequence[int], path: Iterable[str]) -> State:
    """Replay a word without importing or trusting the search backend."""
    state = graph.validate_state(start, "start")
    for move in path:
        permutation = graph.generators[move]
        state = graph.step(state, permutation)
    return state


def verify_path(
    graph: GraphSpec, start: Sequence[int], path: Iterable[str], target: Sequence[int]
) -> bool:
    try:
        checked_target = graph.validate_state(target, "target")
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
    within the visited set. Root eccentricity is a diameter only for regular
    actions; use exact_diameter for repeated-label or singular matrix actions.
    """
    _integer_budget(max_states, "max_states", 1)
    if max_depth is not None:
        _integer_budget(max_depth, "max_depth", 0)
    origin = graph.identity if start is None else graph.validate_state(start, "start")
    goal = None if target is None else graph.validate_state(target, "target")
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
                neighbor = graph.step(state, permutation)
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
        diameter=len(layers) - 1 if complete and graph.is_regular_state(origin) else None,
        termination=termination,
        distances=MappingProxyType(distances),
        metadata={
            "scope": "generated_component",
            "distance_evidence": "exact_for_each_discovered_state",
            "max_states": max_states,
            "max_depth": max_depth,
            "target_requested": goal is not None,
            "root_eccentricity": len(layers) - 1 if complete else None,
            "diameter_method": "regular_action_root_bfs" if graph.is_regular_state(origin) else "requires_all_roots",
        },
    )



def exact_diameter(graph: GraphSpec, *, max_states: int = 100_000,
                   max_depth: int | None = None) -> SearchResult:
    """Certify a finite orbit diameter, sharing the budget across root searches.

    Regular actions need one exhaustive traversal. Schreier and other nonfree
    actions require every root; ``max_states`` then caps total discovered-state
    visits across all traversals, not just the number of distinct vertices.
    Any returned word is a shortest word from the reported start. If the total
    budget is insufficient, its length is only a diameter lower bound.
    """
    first = exact_bfs(graph, max_states=max_states, max_depth=max_depth)
    if graph.vertex_transitive:
        return first
    total = first.explored_states
    best = first
    completed = int(first.complete)
    if first.complete:
        for root in first.distances:
            if root == first.start:
                continue
            remaining = max_states - total
            if remaining < 1:
                break
            result = exact_bfs(graph, start=root, max_states=remaining, max_depth=max_depth)
            total += result.explored_states
            if (result.distance or 0) > (best.distance or 0):
                best = result
            if not result.complete:
                break
            completed += 1
    complete = first.complete and completed == len(first.distances)
    metadata = {**best.metadata, "diameter_method": "all_roots_exact_bfs",
                "completed_roots": completed, "component_size": len(first.distances) if first.complete else None,
                "total_state_visits": total, "max_states": max_states,
                "diameter_lower_bound": best.distance,
                "component_exhausted": first.complete}
    return replace(best, complete=complete, diameter=best.distance if complete else None,
                   termination="exhausted_all_roots" if complete else "incomplete_diameter_search",
                   metadata=metadata)

def _load_optional(module: str, installation: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise BackendUnavailable(
            f"{module} is unavailable ({error}). Install {installation} in this Python environment."
        ) from error


def cayleypy_graph(graph: GraphSpec, *, central_state: Sequence[int] | None = None,
                   device: str = "cpu", seed: int = 0, **kwargs: Any) -> Any:
    """Return the native CayleyGraph, exposing its complete public research API.

    Native algorithms retain their own numerical/hash assumptions; only the
    explicit replay and exact experiment functions provide our certificates.
    """
    if graph.action == "matrix" and graph.matrix_dimension * (graph.modulo - 1)**2 > 2**63 - 1:
        raise ValueError("matrix modulus/dimension may overflow CayleyPy int64 multiplication; use the exact backend")
    cp = _load_optional("cayleypy", "python -m pip install '.[research]'")
    center = graph.identity if central_state is None else graph.validate_state(central_state)
    if graph.action == "permutation":
        definition = cp.CayleyGraphDef.create(
            generators=[list(perm) for perm in graph.generators.values()],
            generator_names=list(graph.generators), central_state=list(center), name=graph.name)
    else:
        # The overflow check above guarantees that native int64 multiplication
        # agrees with exact modular arithmetic for reduced state entries.
        class ModularGenerator(cp.MatrixGenerator):
            # CayleyPy's default .inv uses a real-valued matrix inverse. For
            # modular groups the inverse must instead be calculated in Z/mZ.
            @cached_property
            def inv(self):
                values = _modular_inverse(self.matrix.tolist(), self.modulo)
                native = cp.MatrixGenerator.create([list(row) for row in values], modulo=self.modulo)
                return ModularGenerator(native.matrix, native.modulo)

        generators = []
        for matrix in graph.generators.values():
            native = cp.MatrixGenerator.create([list(row) for row in matrix], modulo=graph.modulo)
            generators.append(ModularGenerator(native.matrix, native.modulo))
        definition = cp.CayleyGraphDef.for_matrix_group(
            generators=generators,
            generator_names=list(graph.generators), central_state=list(center), name=graph.name)
    return cp.CayleyGraph(definition, device=device, random_seed=seed, verbose=0, **kwargs)


def _cayleypy_graph(graph: GraphSpec, target: State, *, device: str, seed: int = 0, **kwargs: Any) -> Any:
    return cayleypy_graph(graph, central_state=target, device=device, seed=seed, **kwargs)


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
    origin = graph.validate_state(start, "start")
    goal = graph.identity if target is None else graph.validate_state(target, "target")
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
    origin = graph.validate_state(start, "start")
    goal = graph.identity if target is None else graph.validate_state(target, "target")
    if graph.action != "permutation":
        raise BackendUnavailable("MultiGPUBeamSearch currently supports permutation/gather actions only; use cayleypy or exact for matrix graphs")
    if platform.system() != "Linux":
        raise BackendUnavailable("MultiGPUBeamSearch native execution requires Linux and NVIDIA CUDA GPUs")
    torch = _load_optional("torch", "CUDA-enabled PyTorch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise BackendUnavailable(f"MultiGPUBeamSearch requires {num_gpus} available NVIDIA CUDA GPU(s)")
    native = _load_optional(
        "multigpubeamsearch",
        "python -m pip install '.[research]'",
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
