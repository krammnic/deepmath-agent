"""Command-line research runner. All commands write only explicit local outputs."""

import argparse
import importlib.util
import json
import os
import re
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .engine import run
from .experiments import execute
from .ledger import export_research, verify_run
from .models import Experiment, Problem, RunConfig
from .providers import ChatCompletionsProvider
from .qk import DisabledQKVerifier, QKConfig, QKVerifier


def load_json(path):
    return json.loads(Path(path).read_text())


def expand_environment(value):
    if isinstance(value, dict):
        return {k: expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError(f"Set {name} referenced by the run configuration")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    return value


def provenance(config: dict) -> dict:
    root = Path(__file__).resolve().parents[3]
    if not (root / "references/sources.json").is_file():
        root = Path(sys.prefix) / "share" / "cayleypy" / "research"
    record = {"llm": config.get("llm", {}), "qk": config.get("qk", {}), "research_context": []}
    for name in ["references/sources.json"]:
        path = root / name
        if path.is_file():
            record[name] = load_json(path)
    for name in ["docs/research.md", "docs/lrx-case-study.md"]:
        path = root / name
        if path.is_file():
            record["research_context"].append({"source": name, "text": path.read_text()[:24000]})
    return record


def default_problem(n: int) -> Problem:
    return Problem.from_dict({"id": f"lrx-diameter-n{n}", "title": f"LRX diameter at n={n}",
        "statement": "Test the diameter candidate n(n-1)/2 at this finite n. Universal upper bound remains a proof obligation.",
        "graph": {"family": "lrx", "n": n}, "objective": "diameter",
        "expected_diameter": n * (n - 1) // 2,
        "sources": ["https://arxiv.org/abs/2509.19162", "LRX_PROOF_EN_20260925.pdf"]})


def output_path(value: str | None) -> Path:
    if value:
        return Path(value)
    return Path("runs") / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S-%fZ")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cayleypy-research", description="Multiagent Cayley graph conjecture research")
    commands = result.add_subparsers(dest="command", required=True)
    research = commands.add_parser("run", help="Run live LLM agents and configured QK/search backends")
    research.add_argument("problem")
    research.add_argument("--config", required=True)
    research.add_argument("--output")
    research.add_argument("--workspace", type=Path, help="Retain research memory and continue from earlier findings")
    exp = commands.add_parser("experiment", help="Run one bounded experiment without LLM calls")
    exp.add_argument("problem")
    exp.add_argument("--backend", choices=["exact", "cayleypy", "multigpu"], default="exact")
    exp.add_argument("--max-states", type=int, default=100000)
    exp.add_argument("--max-depth", type=int, default=100)
    exp.add_argument("--beam-width", type=int, default=4096)
    exp.add_argument("--device", default="cpu")
    exp.add_argument("--num-gpus", type=int, default=1)
    exp.add_argument("--kind", choices=["diameter", "path", "witness", "area_probe"])
    exp.add_argument("--statistic", choices=["inversions", "cyclic_inversions"], default="inversions")
    for name, help_text in [("verify-run", "Check artifact hashes and event-chain consistency"),
                            ("replay", "Repeat recorded experiments with the recorded budgets")]:
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("run_dir")
    export = commands.add_parser("export", help="Export a research evidence package locally")
    export.add_argument("run_dir")
    export.add_argument("--destination", required=True)
    workspace = commands.add_parser("workspace", help="Manage a continuing research investigation")
    workspace_commands = workspace.add_subparsers(dest="workspace_command", required=True)
    init = workspace_commands.add_parser("init", help="Initialize a research workspace")
    init.add_argument("path", type=Path)
    context = workspace_commands.add_parser("context", help="Inspect relevant findings and open obligations")
    context.add_argument("path", type=Path)
    context.add_argument("problem", type=Path)
    register = workspace_commands.add_parser("add", help="Register a verified run in a workspace")
    register.add_argument("path", type=Path)
    register.add_argument("run_dir", type=Path)
    commands.add_parser("qk", help="Fit or apply the reasoning scorer (cayleypy-research qk --help)")
    commands.add_parser("graphs", help="List available graph constructors and their parameters")
    graph = commands.add_parser("graph", help="Resolve and inspect a problem's exact graph definition")
    graph.add_argument("problem", type=Path)
    commands.add_parser("doctor", help="Report optional dependency availability without downloading models")
    return result


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "qk":
            try:
                from cayleypy.qk.cli import parser as qk_parser
            except ImportError as exc:
                raise RuntimeError("Install the research environment with python -m pip install '.[research]'") from exc
            command = qk_parser()
            command.prog = "cayleypy-research qk"
            qk_args = command.parse_args(arguments[1:])
            qk_args.func(qk_args)
            return 0
        args = parser().parse_args(arguments)
        if args.command in {"graphs", "graph"}:
            from .search import graph_catalog, graph_from_config
            if args.command == "graphs":
                result = graph_catalog()
            else:
                problem = Problem.from_dict(load_json(args.problem))
                result = graph_from_config(problem.graph).to_dict()
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "workspace":
            from .workspace import ResearchWorkspace
            workspace = ResearchWorkspace(args.path)
            if args.workspace_command == "init":
                result = workspace.initialize()
            elif args.workspace_command == "context":
                result = workspace.context(Problem.from_dict(load_json(args.problem)))
            else:
                result = workspace.publish_run(args.run_dir)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "doctor":
            modules = {}
            for name in ["cayleypy", "cayleypy.native", "cayleypy.qk", "torch", "transformers"]:
                spec = importlib.util.find_spec(name)
                modules[name] = {"available": spec is not None, "path": spec.origin if spec else None}
            print(json.dumps({"python": sys.version.split()[0], "modules": modules,
                "note": "Presence does not confirm working GPU, model access, fitted QK artifact, or API credentials."}, indent=2))
            return 0
        if args.command == "run":
            problem = Problem.from_dict(load_json(args.problem))
            settings = expand_environment(load_json(args.config))
            if set(settings) - {"run", "llm", "qk"}:
                raise ValueError("configuration accepts only run, llm and qk sections")
            config = RunConfig(**settings.get("run", {}))
            provider = ChatCompletionsProvider(**settings["llm"])
            verifier = QKVerifier(QKConfig(**settings["qk"])) if settings.get("qk") else DisabledQKVerifier("No QK artifact configured")
            destination = output_path(args.output)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Choose a new run output directory: {destination}")
            source_record = provenance(settings)
            source_record["invocation_id"] = str(uuid.uuid4())
            workspace = None
            if getattr(args, "workspace", None):
                from .workspace import ResearchWorkspace
                workspace = ResearchWorkspace(args.workspace)
                workspace.initialize()
                source_record["research_memory"] = workspace.context(problem)
            try:
                with verifier:
                    summary = run(problem, config, provider, verifier, destination, source_record)
            except Exception:
                if workspace is not None and (destination / "manifest.json").is_file():
                    try:
                        finalized = load_json(destination / "summary.json")
                        if finalized.get("provenance", {}).get("invocation_id") == source_record["invocation_id"]:
                            workspace.publish_run(destination)
                    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as registration_error:
                        print(f"cayleypy-research: workspace registration also failed: {registration_error}; "
                              f"run evidence remains at {destination}", file=sys.stderr)
                raise
            if workspace is not None:
                workspace.publish_run(destination)
            print(json.dumps({"status": summary["status"], "mode": summary["mode"], "output": str(destination.resolve()),
                              "claim_status": summary["claim_status"],
                              **({"workspace": str(args.workspace.resolve())} if workspace is not None else {})}, indent=2))
            return 0 if summary["status"] == "completed" else 2
        if args.command == "experiment":
            problem = Problem.from_dict(load_json(args.problem))
            config = RunConfig(backend=args.backend, max_states=args.max_states, max_depth=args.max_depth,
                               beam_width=args.beam_width, device=args.device, num_gpus=args.num_gpus)
            kind = args.kind or problem.objective
            experiment = Experiment.from_dict({"kind": kind, "target": problem.target,
                **({"statistic": args.statistic} if kind == "area_probe" else {})}, problem)
            print(json.dumps(execute(problem, experiment, config), indent=2))
            return 0
        root = Path(args.run_dir)
        if args.command == "verify-run":
            print(json.dumps(verify_run(root), indent=2))
        elif args.command == "export":
            print(export_research(root, Path(args.destination)).resolve())
        elif args.command == "replay":
            verify_run(root)
            summary = load_json(root / "summary.json")
            problem = Problem.from_dict(summary["problem"])
            config = RunConfig(**summary["config"])
            checks = []
            for record in summary["rounds"]:
                for old in record["observations"]:
                    if old["result"].get("status") != "completed":
                        checks.append({"idea_id": old["idea_id"], "match": False, "reason": "Original experiment had no completed result"})
                        continue
                    new = execute(problem, Experiment.from_dict(old["experiment"], problem), config)
                    keys = ["evidence", "path_verified", "diameter", "complete", "expected_diameter_check"]
                    if old["result"].get("path_verified"):
                        keys += ["path_length"]
                    if old["result"].get("exact"):
                        keys += ["path_length", "explored_states", "layer_sizes"]
                    if old["result"].get("evidence") == "finite_heuristic_probe":
                        keys += ["lower_bound", "equality", "explored_states", "statistic"]
                    checks.append({"idea_id": old["idea_id"], "match": all(new.get(k) == old["result"].get(k) for k in keys)})
            matched = bool(checks) and all(c["match"] for c in checks)
            print(json.dumps({"matched": matched, "checks": checks,
                              "scope": "Re-execution with the same implementation; no new general theorem or independent proof"}, indent=2))
            return 0 if matched else 2
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        print(f"cayleypy-research: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
