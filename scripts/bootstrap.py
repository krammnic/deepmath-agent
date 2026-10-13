#!/usr/bin/env python3
"""Verify bundled research sources offline; no downloads, builds, or writes.

All four source trees are included in a normal clone. Install the research
runtime with ``python -m pip install '.[research]'``. Both the default command
and ``--check`` validate the exact files recorded in the source manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


class BootstrapError(RuntimeError):
    """Bundled sources do not match their recorded snapshot."""


@dataclass(frozen=True)
class Repository:
    name: str
    url: str
    commit: str
    path: str
    source_manifest: str = "vendor/source-manifest.json"


def _document(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BootstrapError(f"Expected an object in {path}")
    return document


def load_lock(path: Path) -> list[Repository]:
    document = _document(path)
    if document.get("schema_version") != 2:
        raise BootstrapError("Unsupported dependency lock schema")
    rows = document.get("repositories")
    if not isinstance(rows, list) or not rows:
        raise BootstrapError("Dependency lock must contain a nonempty repositories array")
    repositories = []
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BootstrapError("Each dependency must be an object")
        fields = {key: row.get(key) for key in ("name", "url", "commit", "path", "source_manifest")}
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise BootstrapError("Each dependency needs string name, url, commit, path, and source_manifest")
        repository = Repository(**fields)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", repository.name):
            raise BootstrapError(f"Invalid repository name: {repository.name!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", repository.commit):
            raise BootstrapError(f"{repository.name}: commit must be a full lowercase 40-character SHA")
        if repository.path != f"vendor/{repository.name}" or repository.source_manifest != "vendor/source-manifest.json":
            raise BootstrapError(f"{repository.name}: source paths must remain inside vendor")
        if repository.name in names:
            raise BootstrapError(f"Duplicate dependency: {repository.name}")
        names.add(repository.name)
        repositories.append(repository)
    return repositories


def _safe_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (not parts or PurePosixPath(relative).is_absolute() or ".." in parts or "\\" in relative
            or ":" in relative or relative != PurePosixPath(relative).as_posix()):
        raise BootstrapError(f"Invalid source path: {relative!r}")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BootstrapError(f"Refusing symlink in bundled sources: {candidate}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BootstrapError(f"Cannot read bundled source {path}: {exc}") from exc
    return digest.hexdigest()


def check_repository(root: Path, repository: Repository) -> None:
    """Check every bundled byte, reject missing/extra files, and never repair."""
    destination = _safe_file(root, repository.path)
    manifest_path = _safe_file(root, repository.source_manifest)
    manifest = _document(manifest_path)
    if manifest.get("schema_version") != 1:
        raise BootstrapError("Unsupported source manifest schema")
    record = manifest.get("repositories", {}).get(repository.name)
    if not isinstance(record, dict) or record.get("commit") != repository.commit:
        raise BootstrapError(f"{repository.name}: source manifest revision differs from dependency lock")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise BootstrapError(f"{repository.name}: empty or invalid source manifest")
    for relative, expected in files.items():
        if not isinstance(expected, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256"))):
            raise BootstrapError(f"{repository.name}: invalid hash for {relative}")
        path = _safe_file(destination, relative)
        if not path.is_file():
            raise BootstrapError(f"{repository.name}: missing bundled source {relative}")
        if path.stat().st_size != expected.get("size") or _sha256(path) != expected["sha256"]:
            raise BootstrapError(f"{repository.name}: modified bundled source {relative}; left untouched")
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise BootstrapError(f"{repository.name}: symlink in bundled source {path}")
        relative = path.relative_to(destination)
        # Imports may create bytecode; it is never part of the source snapshot.
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            actual.add(relative.as_posix())
    extra = actual - files.keys()
    if extra:
        raise BootstrapError(f"{repository.name}: unexpected bundled files: {', '.join(sorted(extra)[:5])}; left untouched")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lock", type=Path, help="Lock path; default ROOT/dependencies.lock.json")
    parser.add_argument("--check", action="store_true", help="Verify bundled sources offline (also the default)")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    lock_path = args.lock or root / "dependencies.lock.json"
    try:
        repositories = load_lock(lock_path)
        manifest_path = _safe_file(root, "vendor/source-manifest.json")
        expected = _document(lock_path).get("source_manifest_sha256")
        if not isinstance(expected, str) or _sha256(manifest_path) != expected:
            raise BootstrapError("Source manifest SHA256 differs from dependency lock")
        for repository in repositories:
            check_repository(root, repository)
            print(f"verified {repository.name} {repository.commit}")
    except BootstrapError as exc:
        print(f"bootstrap: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
