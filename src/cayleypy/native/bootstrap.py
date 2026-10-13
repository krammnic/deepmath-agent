"""Locate native sources in a CayleyPy checkout or installed distribution."""
from dataclasses import replace
from pathlib import Path

from .errors import NativeUnavailable


_REQUIRED_FILES = (
    "CMakeLists.txt",
    "tools/production_runner.cu",
    "tools/export_stream1_mlp.py",
    "tools/stream1_mlp_libtorch_backend.hpp",
)


def native_source_dir():
    """Return the editable native source tree without downloading code."""
    module = Path(__file__).resolve()
    checkout = module.parents[3] / "native"
    installed = module.parent / "_source"
    for candidate in (checkout, installed):
        if all((candidate / relative).is_file() for relative in _REQUIRED_FILES):
            return candidate
    raise NativeUnavailable(
        "CayleyPy native sources are missing; use a complete checkout or reinstall CayleyPy"
    )


def prepare_sources(options):
    if options.runner_path is not None:
        return options
    if options.source_dir is None:
        options = replace(options, source_dir=native_source_dir())
    if options.cutlass_dir is None:
        from .sources import setup_sources
        options = setup_sources(options=options)
    return options
