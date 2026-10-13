"""Score research ideas with fitted query/key attention weights.

``ZeroShotQKScorer.rewards`` applies an artifact fitted on unlabeled completions.
The returned margin is a frozen query/key voting signal, not a probability,
mathematical verification, or a proof. Positive margins predict ``correct``;
zero and negative margins predict ``incorrect`` under the scoring protocol.

The model weights and accelerator are loaded on the first nonempty scoring
call. Decoder attention must expose ``q_proj`` and ``k_proj`` (for example
Qwen/Llama). The model loader uses ``trust_remote_code=True``: use a reviewed
model at an immutable revision. Candidates may be truncated in the middle to
``max_length``; a margin remains a ranking aid even for a complete proof.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

QK_METHOD = "frozen-cf-qk-rsym"


class QKUnavailableError(RuntimeError):
    """A requested QK backend or fitted artifact is unavailable.

    Callers may record this failure and continue without QK only when their run
    policy permits it. Never substitute a heuristic number for a missing score.
    """


@dataclass(frozen=True)
class QKConfig:
    """Configuration for an already fitted, zero-label QK artifact.

    Fit the artifact with ``cayleypy-research qk fit`` first.
    The model, revision, and selected layers come from that artifact, ensuring
    inference uses the same model and head pool that produced the weights.
    Set ``device='cpu', dtype='float32'`` explicitly for CPU inference.
    """

    artifact_path: str | Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_length: int = 4096
    micro_batch_size: int = 2

    def __post_init__(self) -> None:
        if not str(self.artifact_path).strip():
            raise ValueError("artifact_path must identify a fitted QK artifact")
        if not self.device.strip():
            raise ValueError("device must be nonempty")
        if self.dtype not in {"float16", "fp16", "bfloat16", "bf16", "float32"}:
            raise ValueError("Unsupported QK dtype")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length < 128:
            raise ValueError("max_length must be an integer >= 128")
        if isinstance(self.micro_batch_size, bool) or not isinstance(self.micro_batch_size, int) or self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be a positive integer")


@dataclass(frozen=True)
class QKScore:
    """A QK margin and provenance, or an explicit absence of a score.

    ``prediction`` names the QK classifier output, not mathematical
    truth. ``is_proof`` is always false and survives JSON serialization through
    :meth:`to_dict`. ``artifact_sha256`` identifies the fitted head weights.
    """

    margin: float | None
    prediction: Literal["correct", "incorrect", "unavailable"]
    status: Literal["scored", "unavailable"]
    method: str = QK_METHOD
    model_id: str | None = None
    model_revision: str | None = None
    artifact_sha256: str | None = None
    reason: str | None = None
    is_proof: Literal[False] = field(default=False, init=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without dropping the evidence limitation."""
        return asdict(self)


def _check_inputs(problem: str, candidates: Sequence[str]) -> list[str]:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a nonempty string")
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of candidate strings, not a string")
    texts = list(candidates)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each candidate must be a nonempty string")
    return texts


class QKVerifier:
    """Score candidate ideas or solutions using the real optional QK backend.

    Example::

        with QKVerifier(QKConfig("artifacts/qk.json")) as verifier:
            scores = verifier.score(problem, [idea_a, idea_b])

    The backend is persistent across calls to amortize model loading. Use this
    object from one worker at a time because projection capture hooks mutate
    per-inference state. Model/download/device errors remain explicit errors;
    they are never silently converted into negative scores.
    """

    def __init__(self, config: QKConfig):
        self.config = config
        self._artifact: Any = None
        self._scorer: Any = None
        self._artifact_sha256: str | None = None
        self._closed = False

    def _load(self) -> None:
        if self._closed:
            raise RuntimeError("QKVerifier is closed")
        if self._scorer is not None:
            return
        path = Path(self.config.artifact_path).expanduser()
        if not path.is_file():
            raise QKUnavailableError(
                f"QK artifact not found: {path}. Fit an unlabeled reference pool "
                "with `cayleypy-research qk fit` before enabling QK scoring."
            )
        try:
            backend = import_module("cayleypy.qk")
        except ImportError as exc:
            raise QKUnavailableError(
                "QK scoring requires the scorer's inference dependencies. "
                "Install them with `python -m pip install '.[research]'`."
            ) from exc
        artifact_bytes = path.read_bytes()
        artifact = backend.ZeroShotQKArtifact.load(path)
        # Loading validates the scoring method, fixed views, zero labels,
        # zero threshold, and weight vectors before any model is downloaded.
        artifact.validate()
        if path.read_bytes() != artifact_bytes:
            raise ValueError("QK artifact changed while it was being loaded; retry with a stable file")
        try:
            scorer = backend.ZeroShotQKScorer(
                model_id=artifact.model_id,
                model_revision=artifact.model_revision,
                device=self.config.device,
                dtype=self.config.dtype,
                max_length=self.config.max_length,
                micro_batch_size=self.config.micro_batch_size,
                layer_start_fraction=artifact.layer_start_fraction,
            )
        except ImportError as exc:
            raise QKUnavailableError(
                "QK model initialization is missing an optional inference dependency: "
                f"{exc}. Install the research requirements for this device."
            ) from exc
        self._artifact = artifact
        self._scorer = scorer
        self._artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

    def score(self, problem: str, candidates: Sequence[str]) -> list[QKScore]:
        """Score candidates in their original order against the same problem.

        The fitted artifact enforces a fixed zero threshold; this adapter
        neither fits a correctness threshold nor maps the margin to confidence.
        Empty batches avoid model initialization. Malformed/nonfinite backend
        outputs raise errors so invalid evidence cannot enter a research ledger.
        """
        texts = _check_inputs(problem, candidates)
        if self._closed:
            raise RuntimeError("QKVerifier is closed")
        if not texts:
            return []
        self._load()
        margins = self._scorer.rewards([problem] * len(texts), texts, self._artifact)
        raw_values = list(margins)
        if len(raw_values) != len(texts):
            raise ValueError("QK backend returned a different number of scores than candidates")
        result: list[QKScore] = []
        for raw_margin in raw_values:
            margin = float(raw_margin)
            if not math.isfinite(margin) or not -1.0 - 1e-8 <= margin <= 1.0 + 1e-8:
                raise ValueError(f"QK backend returned an invalid margin: {margin!r}")
            # Only remove floating-point drift at the documented boundaries.
            margin = max(-1.0, min(1.0, margin))
            result.append(QKScore(
                margin=margin,
                prediction="correct" if margin > 0.0 else "incorrect",
                status="scored",
                model_id=self._artifact.model_id,
                model_revision=getattr(self._scorer, "model_revision", self._artifact.model_revision),
                artifact_sha256=self._artifact_sha256,
            ))
        return result

    def prepare(self) -> None:
        """Load and validate the artifact/model before candidate generation.

        A run requiring QK can call this before any paid language-model call,
        so missing artifacts, dependencies, or devices fail early. This method
        may download the configured model using the configured model loader.
        """
        self._load()

    def score_one(self, problem: str, candidate: str) -> QKScore:
        """Convenience wrapper for one candidate, retaining the same semantics."""
        return self.score(problem, [candidate])[0]

    def close(self) -> None:
        """Remove projection hooks and release model references."""
        if self._closed:
            return
        self._closed = True
        scorer, self._scorer = self._scorer, None
        self._artifact = None
        if scorer is not None:
            scorer.close()

    def __enter__(self) -> QKVerifier:
        if self._closed:
            raise RuntimeError("QKVerifier is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class DisabledQKVerifier:
    """Explicit no-QK mode for reproducible offline runs and missing artifacts.

    This mode produces ``margin=None`` and ``prediction='unavailable'``. It
    does not simulate internal transformer attention or confidence scores.
    """

    def __init__(self, reason: str = "QK scoring disabled; no fitted artifact configured"):
        if not reason.strip():
            raise ValueError("A disabled QK verifier needs an explanatory reason")
        self.reason = reason

    def score(self, problem: str, candidates: Sequence[str]) -> list[QKScore]:
        return [QKScore(None, "unavailable", "unavailable", reason=self.reason)
                for _ in _check_inputs(problem, candidates)]

    def score_one(self, problem: str, candidate: str) -> QKScore:
        return self.score(problem, [candidate])[0]

    def prepare(self) -> None:
        """Do nothing; callers requiring QK must reject disabled mode explicitly."""
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> DisabledQKVerifier:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
