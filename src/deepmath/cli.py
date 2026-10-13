"""Command-line research runner. All commands write only explicit local outputs."""

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .engine import run
from .experiments import execute
from .ledger import export_automathlab, verify_run
from .models import Experiment, Problem, RunConfig
from .providers import ChatCompletionsProvider, ScriptedProvider
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
    root = Path(__file__).resolve().parents[2]
    if not (root / "references/sources.json").is_file():
        root = Path(sys.prefix) / "share" / "deepmath"
    record = {"llm": config.get("llm", {}), "qk": config.get("qk", {}), "research_context": []}
    for name in ["dependencies.lock.json", "references/sources.json"]:
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
    result = argparse.ArgumentParser(prog="deepmath", description="Multiagent Cayley graph conjecture research")
    commands = result.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Scripted offline agents with real exact computations; no LLM/QK inference")
    demo.add_argument("--n", type=int, default=6)
    demo.add_argument("--rounds", type=int, default=2)
    demo.add_argument("--output")
    research = commands.add_parser("run", help="Run live LLM agents and configured QK/search backends")
    research.add_argument("problem")
    research.add_argument("--config", required=True)
    research.add_argument("--output")
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
    export = commands.add_parser("export", help="Stage an AutoMathLab-format research package locally")
    export.add_argument("run_dir")
    export.add_argument("--destination", required=True)
    commands.add_parser("doctor", help="Report optional dependency availability without downloading models")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            modules = {name: importlib.util.find_spec(name) is not None for name in ["cayleypy", "multigpubeamsearch", "qk_zero_shot", "torch", "transformers"]}
            print(json.dumps({"python": sys.version.split()[0], "modules": modules,
                "note": "Presence does not confirm working GPU, model access, fitted QK artifact, or API credentials."}, indent=2))
            return 0
        if args.command in {"run", "demo"}:
            if args.command == "demo":
                if not 4 <= args.n <= 8:
                    raise ValueError("offline LRX demo supports n=4..8; use experiment/run for explicit larger budgets")
                problem = default_problem(args.n)
                config = RunConfig(rounds=args.rounds, top_k=3)
                provider, verifier, settings = ScriptedProvider(), DisabledQKVerifier("Offline demonstration: no QK model was run"), {}
            else:
                problem = Problem.from_dict(load_json(args.problem))
                settings = expand_environment(load_json(args.config))
                if set(settings) - {"run", "llm", "qk"}:
                    raise ValueError("configuration accepts only run, llm and qk sections")
                config = RunConfig(**settings.get("run", {}))
                provider = ChatCompletionsProvider(**settings["llm"])
                verifier = QKVerifier(QKConfig(**settings["qk"])) if settings.get("qk") else DisabledQKVerifier("No QK artifact configured")
            destination = output_path(args.output)
            with verifier:
                summary = run(problem, config, provider, verifier, destination, provenance(settings))
            print(json.dumps({"status": summary["status"], "mode": summary["mode"], "output": str(destination.resolve()),
                              "claim_status": summary["claim_status"]}, indent=2))
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
            print(export_automathlab(root, Path(args.destination)).resolve())
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
        print(f"deepmath: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
