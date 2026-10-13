"""Exercise pinning and preservation with local Git repositories; no network."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "deepmath_bootstrap", Path(__file__).resolve().parents[1] / "scripts/bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bootstrap
_spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "project"
        self.root.mkdir()
        self.source = Path(self.directory.name) / "source"
        self.source.mkdir()
        self.git("init", "--quiet")
        (self.source / "value.txt").write_text("pinned bytes\n")
        self.git("add", "value.txt")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-m", "fixture")
        self.commit = self.git("rev-parse", "HEAD")
        self.repo = bootstrap.Repository("sample", self.source.as_uri(), self.commit, "external/sample")
        self.destination = self.root / self.repo.path
        self.lock = self.root / "dependencies.lock.json"
        self.lock.write_text(json.dumps({"schema_version": 1, "repositories": [self.repo.__dict__]}))

    def git(self, *args, cwd=None):
        result = subprocess.run(
            ["git", "-c", f"core.hooksPath={os.devnull}", *args],
            cwd=cwd or self.source, text=True, capture_output=True, check=True,
        )
        return result.stdout.strip()

    def test_clone_fetches_exact_pin_even_when_origin_head_moves(self):
        (self.source / "value.txt").write_text("newer bytes\n")
        self.git("add", "value.txt")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-m", "newer")
        self.assertTrue(bootstrap.sync_repository(self.root, self.repo))
        self.assertEqual((self.destination / "value.txt").read_text(), "pinned bytes\n")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.destination), self.commit)
        self.assertFalse(bootstrap.sync_repository(self.root, self.repo))

    def test_dirty_checkout_is_never_reset(self):
        bootstrap.sync_repository(self.root, self.repo)
        (self.destination / "value.txt").write_text("user work\n")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "local changes"):
            bootstrap.sync_repository(self.root, self.repo)
        self.assertEqual((self.destination / "value.txt").read_text(), "user work\n")

    def test_untracked_work_is_preserved(self):
        bootstrap.sync_repository(self.root, self.repo)
        (self.destination / "notes.txt").write_text("untracked work")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "untracked"):
            bootstrap.check_repository(self.root, self.repo)
        self.assertEqual((self.destination / "notes.txt").read_text(), "untracked work")

    def test_mismatched_revision_does_not_checkout_pin(self):
        bootstrap.sync_repository(self.root, self.repo)
        other = bootstrap.Repository(self.repo.name, self.repo.url, "0" * 40, self.repo.path)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "expected"):
            bootstrap.sync_repository(self.root, other)
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.destination), self.commit)

    def test_wrong_origin_fails(self):
        bootstrap.sync_repository(self.root, self.repo)
        self.git("remote", "set-url", "origin", "https://example.invalid/wrong.git", cwd=self.destination)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "origin"):
            bootstrap.check_repository(self.root, self.repo)

    def test_existing_non_repository_is_not_overwritten(self):
        self.destination.mkdir(parents=True)
        (self.destination / "mine").write_text("keep")
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.sync_repository(self.root, self.repo)
        self.assertEqual((self.destination / "mine").read_text(), "keep")

    def test_symlink_destination_is_rejected(self):
        self.destination.parent.mkdir()
        self.destination.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "symlink"):
            bootstrap.sync_repository(self.root, self.repo)

    def test_check_missing_checkout_does_not_create_directories(self):
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(bootstrap.main(["--root", str(self.root), "--check"]), 1)
        self.assertFalse((self.root / "external").exists())

    def test_check_existing_checkout_does_not_change_git_index(self):
        bootstrap.sync_repository(self.root, self.repo)
        index = self.destination / ".git/index"
        before = (index.read_bytes(), index.stat().st_mtime_ns)
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(bootstrap.main(["--root", str(self.root), "--check"]), 0)
        self.assertEqual((index.read_bytes(), index.stat().st_mtime_ns), before)

    def test_invalid_lock_cannot_escape_external_or_use_moving_revision(self):
        for field, value in (("path", "../elsewhere"), ("commit", "main"), ("name", "../sample")):
            with self.subTest(field=field):
                record = {**self.repo.__dict__, field: value}
                self.lock.write_text(json.dumps({"schema_version": 1, "repositories": [record]}))
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap.load_lock(self.lock)

    def test_failed_fetch_cleans_only_its_staging_directory(self):
        bad = bootstrap.Repository(self.repo.name, self.repo.url, "0" * 40, self.repo.path)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.sync_repository(self.root, bad)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_existing_error_prevents_other_downloads(self):
        bootstrap.sync_repository(self.root, self.repo)
        (self.destination / "value.txt").write_text("user work")
        with patch.object(bootstrap, "sync_repository") as sync:
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(bootstrap.main(["--root", str(self.root)]), 1)
            sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
