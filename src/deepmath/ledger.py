"""Local provenance and AutoMathLab-compatible experiment export."""

import hashlib
import importlib.metadata
import json
import platform
import shlex
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    staging.replace(path)


def environment() -> dict:
    packages = {}
    for name in ["deepmath-agent", "cayleypy", "qk-zero-shot", "multigpubeamsearch", "torch", "transformers", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    package_dir = Path(__file__).resolve().parent
    source_hashes = {path.name: sha256(path) for path in sorted(package_dir.glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "runner_sources": source_hashes,
            "runner_source_sha256": hashlib.sha256(canonical(source_hashes)).hexdigest()}


def verification_status(summary: dict) -> str:
    """Classify recorded finite evidence, independently of run/review success."""
    for record in summary.get("rounds", []):
        for observation in record.get("observations", []):
            result = observation.get("result", {})
            if result.get("status") != "completed":
                continue
            evidence = result.get("evidence")
            if evidence == "exact_finite_diameter" and result.get("exact") and result.get("complete") and type(result.get("diameter")) is int:
                return "author-checked"
            if evidence == "exact_distance" and result.get("exact") and result.get("path_verified"):
                return "author-checked"
            if evidence == "verified_path_upper_bound" and result.get("path_verified"):
                return "author-checked"
            if (evidence == "finite_heuristic_probe" and result.get("search", {}).get("exact")
                    and type(result.get("explored_states")) is int and result["explored_states"] >= 1):
                return "author-checked"
    return "not-reviewed"


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.previous_hash = "0" * 64
        self.sequence = 0

    def event(self, kind: str, data: dict) -> None:
        record = {"sequence": self.sequence, "time": now(), "kind": kind, "data": data, "previous_hash": self.previous_hash}
        record["hash"] = hashlib.sha256(canonical(record)).hexdigest()
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.previous_hash = record["hash"]
        self.sequence += 1

    def save(self, filename: str, data) -> None:
        write_json(self.root / filename, data)

    def finalize(self, summary: dict) -> None:
        summary["verification_status"] = verification_status(summary)
        self.save("summary.json", summary)
        lines = [f"# {summary['problem']['title']}", "", f"Run: `{self.root.name}`", "",
                 f"Mode: {summary['mode']}. Run status: {summary.get('status', 'unknown')}.", "",
                 f"Claim status: {summary.get('claim_status', 'open')}. Verification: {summary['verification_status']}.", "",
                 "Verification status applies only to successful finite factual experiments. QK margins and agent reviews do not certify mathematical truth. No universal theorem or new Lean replay is claimed.", ""]
        if summary.get("errors"):
            lines.extend(["## Errors", ""])
            for error in summary["errors"]:
                context = f"round {error['round']}, " if "round" in error else ""
                lines.append(f"- {context}{error.get('stage', 'run')}: {error['type']}: {error['message']}")
            lines.append("")
        elif summary.get("error"):
            lines.extend([f"Error: {summary['error']['type']}: {summary['error']['message']}", ""])
        lines.extend(["## Observations", ""])
        for round_data in summary["rounds"]:
            for observation in round_data["observations"]:
                r = observation["result"]
                lines.append(f"- {observation['idea_id']}: status={r.get('status', 'unknown')}; {r.get('evidence', 'inconclusive')}; "
                             f"diameter={r.get('diameter')}, path length={r.get('path_length')}, "
                             f"complete={r.get('complete', False)}, "
                             f"expected diameter check={r.get('expected_diameter_check', 'not requested')}.")
                if r.get("evidence") == "finite_heuristic_probe":
                    lines.append(f"  Statistic={r.get('statistic')}; checked states={r.get('explored_states')}; "
                                 f"lower-bound check={r.get('lower_bound', {}).get('outcome', 'unknown')}; "
                                 f"equality check={r.get('equality', {}).get('outcome', 'unknown')}; "
                                 "universal claim remains unestablished.")
        lines.extend(["", "## Agent synthesis (unverified prose)", "", summary.get("synthesis", {}).get("summary", "No synthesis available."), ""])
        (self.root / "report.md").write_text("\n".join(lines))
        manifest = {"schema_version": 1, "id": self.root.name, "author": "DeepMath research runner",
                    "created": now(), "claim_ids": [summary["problem"]["id"]], "artifacts": []}
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                manifest["artifacts"].append({"path_or_url": path.name, "size_bytes": path.stat().st_size,
                    "sha256": sha256(path), "license": "Local research artifact; review permissions before publishing",
                    "verify_command": f"deepmath verify-run {shlex.quote(str(self.root))}"})
        self.save("manifest.json", manifest)


def verify_run(root: Path) -> dict:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest.get("artifacts", [])
    names = [item["path_or_url"] for item in entries]
    if any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", "..", "manifest.json"} for name in names):
        raise ValueError("manifest artifact paths must be plain filenames")
    if not {"events.jsonl", "summary.json", "report.md", "input.json"}.issubset(names) or len(set(names)) != len(names):
        raise ValueError("manifest is missing required artifacts or contains duplicates")
    for item in entries:
        path = (root / item["path_or_url"]).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError("manifest has an invalid artifact path")
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"artifact digest mismatch: {path.name}")
    previous = "0" * 64
    records = (root / "events.jsonl").read_text().splitlines()
    for index, line in enumerate(records):
        record = json.loads(line)
        digest = record.pop("hash")
        if record["sequence"] != index or record["previous_hash"] != previous or hashlib.sha256(canonical(record)).hexdigest() != digest:
            raise ValueError(f"event chain mismatch at {index}")
        previous = digest
    return {"valid": True, "artifacts": len(entries), "events": len(records),
            "scope": "file integrity and event consistency; not mathematical proof or a digital signature"}


def copy_verified_run(root: Path, destination: Path) -> dict:
    """Copy only manifested files, then verify the independent archived copy."""
    root, destination = Path(root).resolve(), Path(destination)
    verify_run(root)
    manifest = json.loads((root / "manifest.json").read_text())
    names = [item["path_or_url"] for item in manifest["artifacts"]]
    if any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", "..", "manifest.json"} for name in names):
        raise ValueError("manifest artifact paths must be plain filenames")
    destination.mkdir(parents=True, exist_ok=False)
    for filename in names + ["manifest.json"]:
        (destination / filename).write_bytes((root / filename).read_bytes())
    verify_run(destination)
    return json.loads((destination / "summary.json").read_text())


def export_automathlab(root: Path, destination: Path) -> Path:
    root = Path(root).resolve()
    destination = Path(destination)
    verify_run(root)
    summary = json.loads((root / "summary.json").read_text())
    topic = summary["problem"]["graph"]["family"]
    if topic not in {"lrx", "lx", "permutation", "matrix", "cayleypy"}:
        raise ValueError("unsupported graph family in research export")
    status = summary.get("status", "unknown")
    verification = verification_status(summary)
    error = summary.get("error")
    error_text = f"Error: {error['type']}: {error['message']}\n\n" if error else ""
    if not error and summary.get("errors"):
        error_text = "Recorded errors:\n\n" + "\n".join(
            f"- {item.get('stage', 'run')}: {item['type']}: {item['message']}" for item in summary["errors"]
        ) + "\n\n"
    # Create a self-contained staging directory; never edit/publish upstream.
    destination.mkdir(parents=True, exist_ok=False)
    project = destination / "projects" / topic
    for kind in ["experiments", "reviews", "handoffs"]:
        (project / kind).mkdir(parents=True)
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in root.name) or "run"
    artifact_dir = project / "experiments" / name
    copy_verified_run(root, artifact_dir)
    (project / "experiments" / f"{name}.md").write_text(
        f"# {summary['problem']['title']}\n\nID: {name}\n\nAuthor: DeepMath research runner\n\n"
        f"Created: {summary['created']}\n\nRun status: {status}\n\nClaim status: {summary.get('claim_status', 'open')}\n\n"
        f"Verification status: {verification}\n\n{error_text}"
        f"## Precise statement\n\n{summary['problem']['statement']}\n\n"
        f"## Evidence\n\nSee [{name}/report.md]({name}/report.md), input.json, summary.json and manifest.json.\n\n"
        "Finite experiments are bounded by the recorded graph and resource budget. Agent prose and QK scores are unverified.\n\n"
        f"## Reproduce\n\nRun `deepmath replay {shlex.quote(str(artifact_dir))}` to repeat recorded experiments.\n")
    (project / "reviews" / f"{name}.md").write_text(
        f"# Review requested: {name}\n\nOutcome: not-reviewed\n\nIndependent reproduction and any formal proof replay remain pending.\n"
        "Agent critiques are recorded in summary.json; they are not independent human or kernel verification.\n")
    (project / "handoffs" / f"{name}.md").write_text(
        f"# Handoff: {name}\n\nRun status: {status}. Verification status: {verification}.\n\n{error_text}"
        "Review retained experiment artifacts and recorded errors, then close the proof obligations "
        "in the synthesis before changing any theorem status. No running processes are transferred.\n")
    return project
