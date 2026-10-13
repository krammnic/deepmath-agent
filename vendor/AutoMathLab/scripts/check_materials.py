"""Structural checks only; no mathematical verification or execution of submissions."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
errors = []
for name in filter(None, paths):
    path = root / name
    if path.is_symlink():
        errors.append(f"{name}: symlinks are not accepted")
        continue
    if not path.is_file():
        errors.append(f"{name}: tracked file missing")
        continue
    if path.stat().st_size > 10 * 1024 * 1024:
        errors.append(f"{name}: use an artifact manifest for files over 10 MiB")
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            errors.append(f"{name}: {exc}")
for name in ["README.md", "INDEX.md", "TASKS.md", "CONTRIBUTING.md", "AGENTS.md"]:
    if not (root / name).is_file():
        errors.append(f"missing {name}")
if errors:
    raise SystemExit("\n".join(errors))
print("Structure OK. Mathematical claims were not checked.")
