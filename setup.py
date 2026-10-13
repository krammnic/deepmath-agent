"""Build a single distribution containing the complete pinned research sources."""
from pathlib import Path
import hashlib
import json
import shutil
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent
SOURCES = ROOT / "vendor"
PACKAGE_ROOTS = {
    "deepmath": "src/deepmath",
    "deepmath_vendor": "vendor",
    "cayleypy": "vendor/cayleypy/cayleypy",
    "qk_zero_shot": "vendor/qk-zero-shot/src/qk_zero_shot",
    "multigpubeamsearch": "vendor/MultiGPUBeamSearch/integrations/cayleypy_native/multigpubeamsearch",
    "cayleypy_native": "vendor/MultiGPUBeamSearch/integrations/cayleypy_native/cayleypy_native",
}
packages = []
for package, folder in PACKAGE_ROOTS.items():
    packages.append(package)
    if package != "deepmath_vendor":
        packages.extend(f"{package}.{child}" for child in find_packages(folder))


class BuildResearchSources(build_py):
    """Use the recorded inventory so local caches never enter distributions."""

    def run(self):
        manifest_path = SOURCES / "source-manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        lock = json.loads((ROOT / "dependencies.lock.json").read_text())
        if hashlib.sha256(manifest_bytes).hexdigest() != lock["source_manifest_sha256"]:
            raise RuntimeError("Bundled source manifest does not match dependency lock")
        manifest = json.loads(manifest_bytes)
        # Validate all snapshot bytes before copying any upstream implementation.
        for name, record in manifest["repositories"].items():
            for relative, expected in record["files"].items():
                source = SOURCES / name / relative
                if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
                    raise RuntimeError(f"Bundled source integrity check failed: {name}/{relative}")
        super().run()
        archive = Path(self.build_lib) / "deepmath_vendor"
        self._copy(manifest_path, archive / "source-manifest.json")
        self._copy(ROOT / "dependencies.lock.json", archive / "dependencies.lock.json")
        for name, record in manifest["repositories"].items():
            for relative in record["files"]:
                self._copy(SOURCES / name / relative, archive / name / relative)
        # Upstream discovery supports this location without fetching native code.
        native = "MultiGPUBeamSearch"
        target = Path(self.build_lib) / "multigpubeamsearch" / "_native_source"
        native_manifest = {}
        roots = {"cuda", "src", "tools", "tests", "cmake", "configs", "schemas", "third_party"}
        for relative, expected in manifest["repositories"][native]["files"].items():
            if relative in {"CMakeLists.txt", "LICENSE"} or Path(relative).parts[0] in roots:
                self._copy(SOURCES / native / relative, target / relative)
                native_manifest[relative] = expected["sha256"]
        (target / "source_manifest.json").write_text(json.dumps(native_manifest, indent=2, sort_keys=True) + "\n")
        # Runtime puzzle tables are addressed relative to the CayleyPy package.
        for relative in manifest["repositories"]["cayleypy"]["files"]:
            if relative.startswith(("cayleypy/data/", "cayleypy/puzzles/gap_files/")):
                self._copy(SOURCES / "cayleypy" / relative, Path(self.build_lib) / relative)

    @staticmethod
    def _copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copymode(source, target)


setup(packages=packages, package_dir=PACKAGE_ROOTS, include_package_data=False,
      cmdclass={"build_py": BuildResearchSources})
