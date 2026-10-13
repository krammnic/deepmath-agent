"""Validated data at the boundary between LLM suggestions and executable tools."""

from dataclasses import asdict, dataclass, field
from typing import Any


def integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def text(value: Any, name: str, limit: int = 16000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value


def permutation(value: Any, n: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"target must contain exactly {n} labels")
    if any(type(x) is not int for x in value) or sorted(value) != list(range(n)):
        raise ValueError(f"target must be a permutation of 0..{n - 1}")
    return tuple(value)


def validate_target(value: Any, graph: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("target must be an array of integers or matrix rows")
    if value and isinstance(value[0], (list, tuple)):
        value = [x for row in value for x in row]
    return graph.validate_state(value, "target")


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    statement: str
    graph: dict
    objective: str
    target: tuple[int, ...] | None = None
    expected_diameter: int | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Problem":
        if not isinstance(data, dict):
            raise ValueError("problem must be a JSON object")
        graph = data["graph"]
        if not isinstance(graph, dict):
            raise ValueError("graph must be a JSON object")
        from .search import graph_from_config, normalize_graph_config
        graph = normalize_graph_config(graph, name=data.get("id", "custom"))
        spec = graph_from_config(graph)
        objective = data.get("objective", "diameter")
        if objective not in {"diameter", "path"}:
            raise ValueError("objective must be diameter or path")
        target = validate_target(data["target"], spec) if data.get("target") is not None else None
        if objective == "path" and target is None:
            raise ValueError("path objective requires a target")
        expected = data.get("expected_diameter")
        if expected is not None:
            integer(expected, "expected_diameter", 0, 10**9)
        sources = data.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
            raise ValueError("sources must be a list of strings")
        return cls(text(data["id"], "id", 128), text(data["title"], "title", 512),
                   text(data["statement"], "statement"), graph, objective, target, expected, sources)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    kind: str
    target: tuple[int, ...] | None = None
    word: tuple[str, ...] | None = None
    statistic: str | None = None

    @classmethod
    def from_dict(cls, data: dict, problem: Problem) -> "Experiment":
        if not isinstance(data, dict) or set(data) - {"kind", "target", "word", "statistic"}:
            raise ValueError("experiment accepts only kind, target, word and statistic")
        kind = data.get("kind")
        if kind not in {"diameter", "path", "witness", "word", "area_probe"}:
            raise ValueError("unrecognized experiment kind")
        from .search import graph_from_config
        target = validate_target(data["target"], graph_from_config(problem.graph)) if data.get("target") is not None else problem.target
        if kind in {"path", "word"} and target is None:
            raise ValueError("path/word experiments require a target")
        if kind == "witness" and (problem.graph["family"] != "lrx" or problem.graph["n"] < 4):
            raise ValueError("reflected witness requires LRX with n>=4")
        word = data.get("word")
        if kind == "word":
            if not isinstance(word, list) or len(word) > 100000 or any(not isinstance(m, str) for m in word):
                raise ValueError("word must be a list of at most 100000 generator names")
        elif word is not None:
            raise ValueError("word only applies to word experiments")
        statistic = data.get("statistic")
        if kind == "area_probe":
            statistic = statistic or "inversions"
            if statistic not in {"inversions", "cyclic_inversions"}:
                raise ValueError("area_probe statistic must be inversions or cyclic_inversions")
        elif statistic is not None:
            raise ValueError("statistic only applies to area_probe experiments")
        return cls(kind, target, tuple(word) if word is not None else None, statistic)


@dataclass(frozen=True)
class Idea:
    id: str
    role: str
    title: str
    rationale: str
    claim: str
    experiments: tuple[Experiment, ...]

    @classmethod
    def from_dict(cls, data: dict, *, idea_id: str, role: str, problem: Problem) -> "Idea":
        experiments = data.get("experiments")
        if not isinstance(experiments, list) or not 1 <= len(experiments) <= 3:
            raise ValueError("an idea must request 1..3 experiments")
        return cls(idea_id, role, text(data.get("title"), "title", 512),
                   text(data.get("rationale"), "rationale"), text(data.get("claim"), "claim"),
                   tuple(Experiment.from_dict(x, problem) for x in experiments))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 2
    top_k: int = 2
    max_states: int = 100000
    max_depth: int = 100
    beam_width: int = 4096
    backend: str = "exact"
    device: str = "cpu"
    num_gpus: int = 1
    seed: int = 0
    require_qk: bool = False

    def __post_init__(self) -> None:
        for name, low, high in [("rounds", 1, 20), ("top_k", 1, 3), ("max_states", 1, 5000000),
                                 ("max_depth", 1, 100000), ("beam_width", 1, 10000000), ("num_gpus", 1, 128)]:
            integer(getattr(self, name), name, low, high)
        integer(self.seed, "seed", 0, 2**32 - 1)
        if self.backend not in {"exact", "cayleypy", "multigpu"}:
            raise ValueError("backend must be exact, cayleypy or multigpu")
        if type(self.require_qk) is not bool:
            raise ValueError("require_qk must be a boolean")
