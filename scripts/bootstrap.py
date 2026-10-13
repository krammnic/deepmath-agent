#!/usr/bin/env python3
"""Fetch immutable dependency sources without installing or executing them.

Existing directories are only inspected; mismatched, modified, untracked, or
non-repository destinations require the owner to resolve them separately.
``--check`` is read-only and suppresses Git's optional index writes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class BootstrapError(RuntimeError):
    """The requested pin cannot be established without changing existing work."""


@dataclass(frozen=True)
class Repository:
    name: str
    url: str
    commit: str
    path: str


def load_lock(path: Path) -> list[Repository]:
    """Load full commit hashes and constrained external checkout destinations."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"Cannot read dependency lock {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise BootstrapError("Unsupported dependency lock schema")
    rows = document.get("repositories")
    if not isinstance(rows, list) or not rows:
        raise BootstrapError("Dependency lock must contain a nonempty repositories array")
    repositories = []
    names: set[str] = set()
    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BootstrapError("Each dependency lock entry must be an object")
        fields = {key: row.get(key) for key in ("name", "url", "commit", "path")}
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise BootstrapError("Each dependency needs string name, url, commit, and path fields")
        repository = Repository(**fields)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", repository.name):
            raise BootstrapError(f"Invalid repository name: {repository.name!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", repository.commit):
            raise BootstrapError(f"{repository.name}: commit must be a full lowercase 40-character SHA")
        if repository.path != f"external/{repository.name}":
            raise BootstrapError(f"{repository.name}: path must be external/{repository.name}")
        if repository.url.startswith("-") or "\n" in repository.url or "\r" in repository.url:
            raise BootstrapError(f"{repository.name}: invalid repository URL")
        if repository.name in names or repository.path in paths:
            raise BootstrapError(f"Duplicate dependency: {repository.name}")
        names.add(repository.name)
        paths.add(repository.path)
        repositories.append(repository)
    return repositories


def _git(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    command = ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false", *arguments]
    try:
        process = subprocess.run(
            command, cwd=cwd, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise BootstrapError(f"Cannot run Git: {exc}") from exc
    if process.returncode:
        raise BootstrapError(process.stderr.strip() or f"Git failed with exit code {process.returncode}")
    return process.stdout.strip()


def _destination(root: Path, repository: Repository) -> Path:
    destination = root / repository.path
    # A symlink could redirect downloads or checks outside the specified root.
    for path in (root / "external", destination):
        if path.is_symlink():
            raise BootstrapError(f"Refusing symlink dependency destination: {path}")
    return destination


def _canonical_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git")


def check_repository(root: Path, repository: Repository) -> None:
    """Check checkout identity, origin, exact revision, and clean source state."""
    destination = _destination(root, repository)
    if not destination.is_dir():
        raise BootstrapError(f"{repository.name}: missing checkout at {destination}")
    top = Path(_git(["rev-parse", "--show-toplevel"], cwd=destination)).resolve()
    if top != destination.resolve():
        raise BootstrapError(f"{repository.name}: destination is not an independent Git checkout")
    actual = _git(["rev-parse", "HEAD"], cwd=destination)
    if actual != repository.commit:
        raise BootstrapError(
            f"{repository.name}: expected {repository.commit}, found {actual}; "
            "existing checkout was left untouched"
        )
    origin = _git(["remote", "get-url", "origin"], cwd=destination)
    if _canonical_url(origin) != _canonical_url(repository.url):
        raise BootstrapError(f"{repository.name}: origin does not match dependency lock; checkout left untouched")
    dirty = _git(["status", "--porcelain", "--untracked-files=all"], cwd=destination)
    if dirty:
        raise BootstrapError(f"{repository.name}: checkout contains local changes or untracked files; left untouched")


def sync_repository(root: Path, repository: Repository) -> bool:
    """Fetch a missing checkout at the pin; return true only when one was created.

    Source is fetched and checked out in a temporary sibling directory. The
    final directory is reserved exclusively before moving the completed clone,
    so an existing directory is never replaced, even if it appears mid-fetch.
    No package install, source build, submodule, or repository script is run.
    """
    destination = _destination(root, repository)
    if destination.exists():
        check_repository(root, repository)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{repository.name}-", dir=destination.parent))
    try:
        _git(["init", "--quiet"], cwd=staging)
        _git(["remote", "add", "origin", repository.url], cwd=staging)
        _git(["fetch", "--quiet", "--depth=1", "origin", repository.commit], cwd=staging)
        _git(["checkout", "--quiet", "--detach", repository.commit], cwd=staging)
        if _git(["rev-parse", "HEAD"], cwd=staging) != repository.commit:
            raise BootstrapError(f"{repository.name}: fetched commit differs from the lock")
        # mkdir fails instead of overwriting a destination created concurrently.
        destination.mkdir()
        for child in staging.iterdir():
            shutil.move(str(child), str(destination / child.name))
        check_repository(root, repository)
    except OSError as exc:
        raise BootstrapError(
            f"{repository.name}: unable to publish checkout: {exc}. "
            "Inspect any incomplete destination before retrying."
        ) from exc
    finally:
        shutil.rmtree(staging)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lock", type=Path, help="Lock path; default ROOT/dependencies.lock.json")
    parser.add_argument("--check", action="store_true", help="Only verify pins and clean state; no downloads or writes")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        repositories = load_lock(args.lock or root / "dependencies.lock.json")
        # Inspect existing destinations before performing any network operation.
        errors = []
        for repository in repositories:
            try:
                destination = _destination(root, repository)
                if args.check or destination.exists():
                    check_repository(root, repository)
            except BootstrapError as exc:
                errors.append(str(exc))
        if errors:
            raise BootstrapError("\n".join(errors))
        for repository in repositories:
            created = False if args.check else sync_repository(root, repository)
            print(f"{'cloned' if created else 'verified'} {repository.name} {repository.commit}")
    except BootstrapError as exc:
        print(f"bootstrap: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
