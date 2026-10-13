"""Exercise offline snapshot verification, tamper detection, and preservation."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

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
        self.root = Path(self.directory.name)
        self.repo = bootstrap.Repository("sample", "https://example.invalid/sample.git", "a" * 40, "vendor/sample")
        self.destination = self.root / self.repo.path
        self.destination.mkdir(parents=True)
        self.source = self.destination / "source.py"
        self.source.write_bytes(b"print('pinned bytes')\n")
        self.manifest = self.root / self.repo.source_manifest
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "repositories": {"sample": {"commit": self.repo.commit, "files": {
                "source.py": {"sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                              "size": self.source.stat().st_size, "mode": "100644"}
            }, "excluded": []}}
        }))
        self.lock = self.root / "dependencies.lock.json"
        self.lock.write_text(json.dumps({
            "schema_version": 2, "repositories": [self.repo.__dict__],
            "source_manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        }))

    def check_main(self, *args):
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            return bootstrap.main(["--root", str(self.root), *args])

    def test_verification_is_offline_and_read_only(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.check_main("--check"), 0)
        self.assertEqual(self.check_main(), 0)
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / "external").exists())

    def test_modified_source_is_never_repaired(self):
        self.source.write_bytes(b"researcher changes\n")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "modified"):
            bootstrap.check_repository(self.root, self.repo)
        self.assertEqual(self.source.read_bytes(), b"researcher changes\n")

    def test_same_size_modification_is_detected(self):
        self.source.write_bytes(self.source.read_bytes().replace(b"pinned", b"edited"))
        self.assertEqual(self.check_main(), 1)

    def test_missing_file_is_not_downloaded_or_created(self):
        self.source.unlink()
        self.assertEqual(self.check_main("--check"), 1)
        self.assertFalse(self.source.exists())

    def test_unknown_source_is_preserved_and_rejected(self):
        extra = self.destination / "local.py"
        extra.write_text("local experiment")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "unexpected"):
            bootstrap.check_repository(self.root, self.repo)
        self.assertEqual(extra.read_text(), "local experiment")

    def test_import_bytecode_does_not_count_as_source(self):
        cache = self.destination / "__pycache__"
        cache.mkdir()
        (cache / "source.cpython-313.pyc").write_bytes(b"bytecode")
        bootstrap.check_repository(self.root, self.repo)

    def test_manifest_tampering_is_detected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        self.assertEqual(self.check_main(), 1)

    def test_revision_mismatch_is_rejected(self):
        wrong = bootstrap.Repository(self.repo.name, self.repo.url, "b" * 40, self.repo.path)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "revision"):
            bootstrap.check_repository(self.root, wrong)

    def test_symlink_file_is_rejected(self):
        outside = self.root / "outside.py"
        outside.write_bytes(self.source.read_bytes())
        self.source.unlink()
        self.source.symlink_to(outside)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "symlink"):
            bootstrap.check_repository(self.root, self.repo)

    def test_symlink_parent_is_rejected(self):
        self.source.unlink()
        self.destination.rmdir()
        outside = self.root / "outside"
        outside.mkdir()
        self.destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "symlink"):
            bootstrap.check_repository(self.root, self.repo)

    def test_lock_cannot_escape_vendor_or_use_moving_revision(self):
        for field, value in (("path", "../elsewhere"), ("commit", "main"),
                             ("name", "../sample"), ("source_manifest", "../source.json")):
            with self.subTest(field=field):
                document = json.loads(self.lock.read_text())
                document["repositories"] = [{**self.repo.__dict__, field: value}]
                self.lock.write_text(json.dumps(document))
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap.load_lock(self.lock)

    def test_manifest_path_traversal_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data["repositories"]["sample"]["files"]["../escape.py"] = data["repositories"]["sample"]["files"]["source.py"]
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(bootstrap.BootstrapError, "Invalid source path"):
            bootstrap.check_repository(self.root, self.repo)


if __name__ == "__main__":
    unittest.main()
