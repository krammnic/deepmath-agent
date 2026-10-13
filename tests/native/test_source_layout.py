"""Native source discovery follows the project layout without network access."""
from pathlib import Path

import pytest

from cayleypy.native import bootstrap, NativeOptions, NativeUnavailable


def source_tree(directory):
    for relative in bootstrap._REQUIRED_FILES:
        file = directory / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("editable native source")
    return directory


def test_checkout_sources_are_editable_and_need_no_inventory(tmp_path, monkeypatch):
    module = tmp_path / "src/cayleypy/native/bootstrap.py"
    monkeypatch.setattr(bootstrap, "__file__", str(module))
    native = source_tree(tmp_path / "native")
    assert bootstrap.native_source_dir() == native
    (native / "tools/production_runner.cu").write_text("research changes")
    assert bootstrap.native_source_dir() == native


def test_installed_package_contains_native_sources(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/cayleypy/native"
    monkeypatch.setattr(bootstrap, "__file__", str(package / "bootstrap.py"))
    native = source_tree(package / "_source")
    assert bootstrap.native_source_dir() == native


def test_missing_sources_fail_without_fetching_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "__file__", str(tmp_path / "site-packages/cayleypy/native/bootstrap.py"))
    with pytest.raises(NativeUnavailable, match="sources are missing"):
        bootstrap.native_source_dir()
    assert not list(tmp_path.iterdir())


def test_configured_toolchain_reuses_native_tree_without_setup(tmp_path, monkeypatch):
    from cayleypy.native import sources
    monkeypatch.setattr(bootstrap, "__file__", str(tmp_path / "src/cayleypy/native/bootstrap.py"))
    native = source_tree(tmp_path / "native")
    monkeypatch.setattr(sources, "setup_sources", lambda **_: pytest.fail("already configured toolchain"))
    options = NativeOptions(cutlass_dir=tmp_path / "toolchain")
    result = bootstrap.prepare_sources(options)
    assert result.source_dir == native
    assert options.source_dir is None
