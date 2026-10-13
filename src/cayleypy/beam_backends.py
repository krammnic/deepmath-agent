"""Backend selection for the common CayleyGraph beam-search API."""
from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from .algo.beam_search import BeamSearchAlgorithm


def _torch_search(graph, **kwargs):
    return BeamSearchAlgorithm(graph).search(**kwargs)


_LOCK = RLock()
_BACKENDS: dict[str, Callable[..., Any]] = {"torch": _torch_search}
_DEFAULT_BACKEND = "torch"
_LAZY_BACKENDS = frozenset({"native", "auto"})


def _name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("beam-search backend must be a nonempty name")
    return name


def register_beam_search_backend(name: str, callback: Callable[..., Any]) -> None:
    """Register a callback accepting ``(graph, **search_options)``.

    Registering does not select a new default or replace another callback.
    ``native`` and ``auto`` may be registered to configure their session options;
    without registration they use CayleyPy's lazy native dispatch.
    """
    name = _name(name)
    if not callable(callback):
        raise TypeError("beam-search backend callback must be callable")
    with _LOCK:
        if name in _BACKENDS:
            raise ValueError(f"beam-search backend is already registered: {name}")
        _BACKENDS[name] = callback


def unregister_beam_search_backend(name: str) -> None:
    """Remove a registration; native/auto resume their built-in dispatch."""
    name = _name(name)
    with _LOCK:
        if name == "torch":
            raise ValueError("the torch beam-search backend cannot be unregistered")
        if name not in _BACKENDS:
            raise ValueError(f"unknown beam-search backend: {name}")
        if name == _DEFAULT_BACKEND and name not in _LAZY_BACKENDS:
            raise ValueError("select another default before unregistering its beam-search backend")
        del _BACKENDS[name]


def get_beam_search_backend(name: str) -> Callable[..., Any]:
    """Return an explicitly registered callback, including the torch backend."""
    name = _name(name)
    with _LOCK:
        if name not in _BACKENDS:
            raise ValueError(f"unknown registered beam-search backend: {name}")
        return _BACKENDS[name]


def get_default_beam_search_backend() -> str:
    """Return the selected backend; new processes default to torch."""
    with _LOCK:
        return _DEFAULT_BACKEND


def set_default_beam_search_backend(name: str) -> str:
    """Select a registered or built-in backend and return the previous default."""
    global _DEFAULT_BACKEND
    name = _name(name)
    with _LOCK:
        if name not in _BACKENDS and name not in _LAZY_BACKENDS:
            raise ValueError(f"unknown beam-search backend: {name}")
        previous = _DEFAULT_BACKEND
        _DEFAULT_BACKEND = name
        return previous


def beam_search(graph, *, backend=None, **kwargs):
    """Dispatch one search without modifying the graph class or its methods."""
    with _LOCK:
        name = _DEFAULT_BACKEND if backend is None else _name(backend)
        callback = _BACKENDS.get(name)
    if callback is not None:
        return callback(graph, **kwargs)
    if name in _LAZY_BACKENDS:
        from .native import beam_search as native_search
        return native_search(graph, backend=name, **kwargs)
    raise ValueError(f"unknown beam-search backend: {name}")
