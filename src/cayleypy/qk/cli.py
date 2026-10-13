"""Fit and apply CayleyPy reasoning score artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .core import FrozenQKRewardScorer, QKRewardArtifact, make_reward_artifact


def read_pairs(path: Path, problem_column: str, completion_column: str) -> tuple[list[str], list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    missing = {problem_column, completion_column} - set(rows[0])
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return (
        [str(row[problem_column]) for row in rows],
        [str(row[completion_column]) for row in rows],
        rows,
    )


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--problem-column", default="problem")
    parser.add_argument("--completion-column", default="completion")


def fit(args: argparse.Namespace) -> None:
    problems, completions, _ = read_pairs(
        args.reference_csv,
        args.problem_column,
        args.completion_column,
    )
    if len(problems) < 2:
        raise ValueError("The unlabeled reference set must contain at least two rows")
    scorer = FrozenQKRewardScorer(
        model_id=args.model,
        model_revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        micro_batch_size=args.micro_batch_size,
        layer_start_fraction=args.layer_start_fraction,
    )
    try:
        artifact = make_reward_artifact(
            scorer,
            problems,
            completions,
            reference_seed=args.reference_seed,
        )
        artifact.save(args.output)
    finally:
        scorer.close()
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "model_id": artifact.model_id,
                "model_revision": artifact.model_revision,
                "unlabeled_reference_size": artifact.unlabeled_reference_size,
                "gold_labels_used": artifact.gold_labels_used,
                "decision_threshold": artifact.decision_threshold,
            },
            indent=2,
        )
    )


def score(args: argparse.Namespace) -> None:
    artifact = QKRewardArtifact.load(args.artifact)
    problems, completions, rows = read_pairs(
        args.input_csv,
        args.problem_column,
        args.completion_column,
    )
    output_fields = {"qk_zero_shot_margin", "qk_zero_shot_prediction"}
    if output_fields & set(rows[0]):
        raise ValueError("Input CSV already contains QK output columns")
    scorer = FrozenQKRewardScorer(
        model_id=artifact.model_id,
        model_revision=artifact.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        micro_batch_size=args.micro_batch_size,
        layer_start_fraction=artifact.layer_start_fraction,
    )
    try:
        margins = scorer.rewards(problems, completions, artifact)
    finally:
        scorer.close()
    output_rows: list[dict[str, object]] = []
    for row, margin in zip(rows, margins):
        output_rows.append(
            {
                **row,
                "qk_zero_shot_margin": float(margin),
                "qk_zero_shot_prediction": "correct" if float(margin) > 0.0 else "incorrect",
            }
        )
    fields = list(rows[0]) + ["qk_zero_shot_margin", "qk_zero_shot_prediction"]
    atomic_csv(args.output, output_rows, fields)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(output_rows),
                "gold_labels_used": 0,
                "decision_threshold": 0.0,
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cayleypy-research qk")
    commands = root.add_subparsers(dest="command", required=True)

    fit_parser = commands.add_parser("fit", help="fit fixed weights on unlabeled completions")
    fit_parser.add_argument("--reference-csv", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--model", required=True)
    fit_parser.add_argument("--revision")
    fit_parser.add_argument("--reference-seed", type=int, default=1729)
    fit_parser.add_argument("--layer-start-fraction", type=float, default=0.5)
    add_shared_arguments(fit_parser)
    fit_parser.set_defaults(func=fit)

    score_parser = commands.add_parser("score", help="score problem/completion pairs")
    score_parser.add_argument("--input-csv", type=Path, required=True)
    score_parser.add_argument("--artifact", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    add_shared_arguments(score_parser)
    score_parser.set_defaults(func=score)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
