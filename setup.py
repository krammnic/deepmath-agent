"""Include native search sources in installed research environments."""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildResearch(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "native"
        target = Path(self.build_lib) / "cayleypy" / "native" / "_source"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".DS_Store", "build", "CMakeFiles", "CMakeCache.txt",
        ))


setup(cmdclass={"build_py": BuildResearch})
