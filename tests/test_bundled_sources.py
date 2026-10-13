"""The distribution's source inventory includes working APIs and build assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BundledSourcesTests(unittest.TestCase):
    def test_every_recorded_snapshot_file_is_present_and_unmodified(self):
        manifest = json.loads((ROOT / "vendor/source-manifest.json").read_text())
        lock = json.loads((ROOT / "dependencies.lock.json").read_text())
        self.assertEqual(hashlib.sha256((ROOT / "vendor/source-manifest.json").read_bytes()).hexdigest(),
                         lock["source_manifest_sha256"])
        self.assertEqual(set(manifest["repositories"]), {r["name"] for r in lock["repositories"]})
        for repository in lock["repositories"]:
            record = manifest["repositories"][repository["name"]]
            self.assertEqual(record["commit"], repository["commit"])
            for relative, expected in record["files"].items():
                path = ROOT / repository["path"] / relative
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"], str(path))
            self.assertEqual(len(record["files"]), repository["bundled_files"])
            self.assertEqual(len(record["excluded"]), repository["excluded_files"])
            for excluded in record["excluded"]:
                self.assertTrue(excluded["path"].startswith("test_results/"))
                self.assertEqual(len(excluded["sha256"]), 64)
                self.assertFalse((ROOT / repository["path"] / excluded["path"]).exists())

    def test_required_public_apis_native_sources_and_templates_are_bundled(self):
        for relative in (
            "cayleypy/cayleypy/__init__.py", "cayleypy/cayleypy/cayley_graph.py",
            "cayleypy/cayleypy/cayley_graph_def.py", "cayleypy/LICENSE",
            "qk-zero-shot/src/qk_zero_shot/core.py", "qk-zero-shot/src/qk_zero_shot/cli.py",
            "MultiGPUBeamSearch/CMakeLists.txt", "MultiGPUBeamSearch/tools/production_runner.cu",
            "MultiGPUBeamSearch/third_party/nlohmann/json.hpp", "MultiGPUBeamSearch/LICENSE",
            "MultiGPUBeamSearch/integrations/cayleypy_native/multigpubeamsearch/__init__.py",
            "MultiGPUBeamSearch/integrations/cayleypy_native/cayleypy_native/__init__.py",
            "AutoMathLab/templates/hypothesis.md", "AutoMathLab/templates/experiment.md",
            "AutoMathLab/templates/review.md", "AutoMathLab/scripts/check_materials.py",
        ):
            self.assertTrue((ROOT / "vendor" / relative).is_file(), relative)

    def test_snapshots_have_no_nested_git_directories(self):
        self.assertEqual(list((ROOT / "vendor").rglob(".git")), [])


if __name__ == "__main__":
    unittest.main()
