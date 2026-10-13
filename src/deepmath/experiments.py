"""A bounded experiment registry; agent prose and source snippets never execute."""

from .lrx import reflected_witness, witness_word
from .models import Experiment, Problem, RunConfig
from .search import (GraphSpec, cayleypy_search, exact_bfs, lrx_graph,
                     multigpu_search, verify_path)


def graph_for(problem: Problem) -> GraphSpec:
    graph = problem.graph
    if graph["family"] == "permutation":
        return GraphSpec(problem.id, graph["generators"])
    spec = lrx_graph(graph["n"])
    if graph["family"] == "lx":
        return GraphSpec(f"lx-{graph['n']}", {k: v for k, v in spec.generators.items() if k != "R"}, spec.expected_order)
    return spec


def execute(problem: Problem, experiment: Experiment, config: RunConfig) -> dict:
    graph = graph_for(problem)
    if experiment.kind == "area_probe":
        from .holography import probe_area
        return probe_area(graph, statistic=experiment.statistic or "inversions",
                          max_states=min(config.max_states, 100000), max_depth=config.max_depth)
    if experiment.kind in {"witness", "word"}:
        target = reflected_witness(graph.n) if experiment.kind == "witness" else experiment.target
        word = witness_word(graph.n) if experiment.kind == "witness" else experiment.word
        valid = verify_path(graph, graph.identity, word, target)
        return {"kind": experiment.kind, "status": "completed", "graph": graph.to_dict(),
                "start": list(graph.identity), "target": list(target), "path": list(word),
                "path_verified": valid, "path_length": len(word), "exact": False,
                "complete": False, "diameter": None,
                "evidence": "verified_path_upper_bound" if valid else "invalid_word",
                "interpretation": "Word replay establishes an upper bound only; the paper's lower bound is separate evidence."}
    if experiment.kind == "diameter":
        # Beam searches cannot certify a diameter. Always use bounded exact BFS.
        result = exact_bfs(graph, max_states=config.max_states, max_depth=config.max_depth)
    elif config.backend == "exact":
        result = exact_bfs(graph, target=experiment.target, max_states=config.max_states, max_depth=config.max_depth)
    else:
        search = cayleypy_search if config.backend == "cayleypy" else multigpu_search
        options = {"beam_width": config.beam_width, "max_depth": config.max_depth, "seed": config.seed}
        if config.backend == "multigpu":
            options["num_gpus"] = config.num_gpus
        else:
            options["device"] = config.device
        result = search(graph, start=graph.identity, target=experiment.target, **options)
    payload = {"kind": experiment.kind, "status": "completed", **result.to_dict()}
    expected = problem.expected_diameter
    if experiment.kind == "diameter" and expected is not None:
        if result.diameter is not None:
            payload["expected_diameter_check"] = "matches" if result.diameter == expected else "refuted_at_this_n"
        elif result.exact and result.distance is not None and result.distance > expected:
            payload["expected_diameter_check"] = "refuted_at_this_n"
        else:
            payload["expected_diameter_check"] = "inconclusive"
        payload["expected_diameter"] = expected
    payload["evidence"] = ("exact_finite_diameter" if result.diameter is not None else
                           "exact_distance" if result.shortest_path else
                           "verified_path_upper_bound" if result.path_verified else "inconclusive")
    return payload
