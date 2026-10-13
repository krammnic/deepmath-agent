"""CayleyPy reasoning scores from frozen, responsive-symmetric QK projections.

The scorer reads only fixed query/key positions from a frozen decoder. Fixed
responsiveness weights are estimated once from unlabeled completions; gold
answers and correctness labels are neither accepted nor stored.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROMPT_MAPPINGS = ("a_correct", "b_correct")
QUERY_VIEWS = ("verdict_start", "verdict_end", "verdict_colon")
KEY_VIEWS = ("last_content", "terminal_punctuation", "end_of_line")
EPS = 1e-8


def _option_line(prefix: str, letter: str, text: str) -> tuple[str, dict[str, int]]:
    line_prefix = f"{letter}. "
    sentence = f"{text}."
    line = line_prefix + sentence + "\n"
    start = len(prefix)
    return line, {
        "last_content": start + len(line_prefix) + len(text) - 1,
        "terminal_punctuation": start + len(line_prefix) + len(sentence) - 1,
        "end_of_line": start + len(line) - 1,
    }


def _token_count(tokenizer: Any, text: str, add_special_tokens: bool = True) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def _truncate_middle(tokenizer: Any, text: str, max_tokens: int) -> str:
    text = str(text)
    if max_tokens <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    if max_tokens <= 16:
        return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)
    marker = "\n...[truncated]...\n"
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    room = max_tokens - len(marker_ids)
    if room <= 0:
        return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)
    head = max(1, room // 3)
    tail = max(1, room - head)
    return tokenizer.decode(
        ids[:head] + marker_ids + ids[-tail:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def build_qk_verifier_prompt(
    problem: str,
    completion: str,
    tokenizer: Any,
    max_length: int,
    mapping: str,
) -> tuple[str, list[int], list[list[int]], list[int]]:
    """Return prompt and character anchors for one fixed A/B mapping.

    The final tuple contains semantic identities for surface options A/B:
    0 means ``correct`` and 1 means ``incorrect``.
    """
    if mapping not in PROMPT_MAPPINGS:
        raise ValueError(f"Unknown mapping {mapping!r}")
    prefix = (
        "You are verifying whether a candidate reasoning chain is correct.\n"
        f"Problem:\n{problem!s}\n\nCandidate solution:\n"
    )
    verdict_intro = "\n\nChoose the correct verdict:\n"
    correct = "The candidate solution is correct"
    incorrect = "The candidate solution contains an error"
    if mapping == "a_correct":
        option_texts, semantics = (correct, incorrect), [0, 1]
    else:
        option_texts, semantics = (incorrect, correct), [1, 0]

    # Determine the candidate budget using the complete, immutable scaffold.
    option_prefix = prefix + str(completion) + verdict_intro
    option_a, _ = _option_line(option_prefix, "A", option_texts[0])
    option_b, _ = _option_line(option_prefix + option_a, "B", option_texts[1])
    scaffold = prefix + verdict_intro + option_a + option_b + "\nVerdict:"
    solution = _truncate_middle(
        tokenizer,
        str(completion),
        max(0, max_length - _token_count(tokenizer, scaffold) - 4),
    )

    option_prefix = prefix + solution + verdict_intro
    option_a, a_positions = _option_line(option_prefix, "A", option_texts[0])
    option_b_prefix = option_prefix + option_a
    option_b, b_positions = _option_line(option_b_prefix, "B", option_texts[1])
    before_verdict = option_b_prefix + option_b + "\n"
    verdict_start = len(before_verdict)
    verdict_end = verdict_start + len("Verdict") - 1
    text = before_verdict + "Verdict:"
    final_tokens = _token_count(tokenizer, text, add_special_tokens=True)
    if final_tokens > max_length:
        raise ValueError(
            f"Verifier scaffold/problem needs {final_tokens} tokens, exceeding "
            f"the pinned max_length={max_length}; refusing to truncate verdict anchors"
        )
    queries = [verdict_start, verdict_end, verdict_end + 1]
    keys = [
        [a_positions[name] for name in KEY_VIEWS],
        [b_positions[name] for name in KEY_VIEWS],
    ]
    return text, queries, keys, semantics


def _char_to_token_positions(offsets: Sequence[Sequence[int]], chars: Sequence[int]) -> list[int]:
    positions: list[int] = []
    valid = [(i, int(a), int(b)) for i, (a, b) in enumerate(offsets) if int(b) > 0]
    for char in chars:
        chosen: int | None = None
        for i, start, end in valid:
            if start <= int(char) < end:
                chosen = i
                break
        if chosen is None:
            after = [i for i, start, _end in valid if start >= int(char)]
            chosen = after[0] if after else valid[-1][0]
        positions.append(int(chosen))
    return positions


def _unique_view_mask(scores: np.ndarray) -> np.ndarray:
    """Mask tokenizer-duplicate Q/K views.

    Args:
        scores: ``[N, heads, mappings, views, 2]``.
    Returns:
        Boolean mask ``[N, mappings, views]``.
    """
    values = np.asarray(scores)
    n, _heads, mappings, views, _candidates = values.shape
    mask = np.ones((n, mappings, views), dtype=bool)
    for row in range(n):
        for mapping in range(mappings):
            for view in range(1, views):
                current = values[row, :, mapping, view, :]
                for prior in range(view):
                    if mask[row, mapping, prior] and np.array_equal(
                        current,
                        values[row, :, mapping, prior, :],
                        equal_nan=True,
                    ):
                        mask[row, mapping, view] = False
                        break
    return mask


def head_votes_from_semantic_scores(
    semantic_scores: np.ndarray,
    mapping_count: int,
) -> np.ndarray:
    """Convert QK contrasts to per-example, per-head votes in ``[-1, 1]``."""
    scores = np.asarray(semantic_scores, dtype=np.float64)
    if scores.ndim != 5 or scores.shape[-1] != 2:
        raise ValueError(f"Expected [N,H,M,V,2], got {scores.shape}")
    if mapping_count not in (1, 2) or scores.shape[2] < mapping_count:
        raise ValueError("mapping_count must be 1 or 2 and available in the score tensor")
    selected = scores[:, :, :mapping_count]
    mask = _unique_view_mask(selected)
    votes = np.sign(selected[..., 0] - selected[..., 1])
    masked = votes * mask[:, None, :, :]
    count = mask.sum(axis=(1, 2), dtype=np.float64)[:, None]
    return np.divide(
        masked.sum(axis=(2, 3), dtype=np.float64),
        count,
        out=np.zeros(masked.shape[:2], dtype=np.float64),
        where=count > 0,
    )


def fit_responsiveness_weights(head_votes: np.ndarray) -> np.ndarray:
    """Fit label-free fixed head weights from an unlabeled completion pool."""
    votes = np.asarray(head_votes, dtype=np.float64)
    if votes.ndim != 2 or len(votes) < 2:
        raise ValueError("Need an [examples, heads] vote matrix with at least two examples")
    responsive = np.abs(votes) > EPS
    responsive_count = responsive.sum(axis=0)
    positive_count = ((votes > 0.0) & responsive).sum(axis=0)
    positive_rate = np.divide(
        positive_count,
        responsive_count,
        out=np.full(votes.shape[1], 0.5, dtype=np.float64),
        where=responsive_count > 0,
    )
    variability = 4.0 * positive_rate * (1.0 - positive_rate)
    coverage = responsive.mean(axis=0)
    boundary_agreement = np.abs(votes).mean(axis=0)
    weights = variability * coverage * boundary_agreement
    if float(weights.sum()) <= EPS:
        weights = np.ones(votes.shape[1], dtype=np.float64)
    return weights / weights.sum()


def aggregate_frozen_rsym(
    one_mapping_votes: np.ndarray,
    two_mapping_votes: np.ndarray,
    one_weights: np.ndarray,
    two_weights: np.ndarray,
) -> np.ndarray:
    """Return the frozen CF-QK-RSym margin in ``[-1, 1]``."""
    one = np.asarray(one_mapping_votes, dtype=np.float64)
    two = np.asarray(two_mapping_votes, dtype=np.float64)
    w1 = np.asarray(one_weights, dtype=np.float64)
    w2 = np.asarray(two_weights, dtype=np.float64)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("One- and two-mapping vote matrices must have identical [N,H] shapes")
    if w1.shape != (one.shape[1],) or w2.shape != (one.shape[1],):
        raise ValueError("Weight vectors do not match the vote head dimension")
    return 0.5 * (one @ w1 + two @ w2)


@dataclass
class QKRewardArtifact:
    schema_version: int
    method: str
    model_id: str
    model_revision: str | None
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    layer_start_fraction: float
    head_indices: list[int]
    one_mapping_weights: list[float]
    two_mapping_weights: list[float]
    unlabeled_reference_size: int
    reference_seed: int
    prompt_mappings: list[str]
    query_views: list[str]
    key_views: list[str]
    gold_labels_used: int = 0
    decision_threshold: float = 0.0

    def validate(self) -> None:
        if self.schema_version != 1 or self.method != "frozen-cf-qk-rsym":
            raise ValueError("Unsupported QK reward artifact")
        if not self.model_id or self.n_layers < 1 or self.n_query_heads < 1 or self.n_kv_heads < 1:
            raise ValueError("Artifact model metadata is incomplete")
        if not 0.0 <= self.layer_start_fraction < 1.0:
            raise ValueError("Artifact layer_start_fraction must be in [0, 1)")
        if self.unlabeled_reference_size < 2:
            raise ValueError("Artifact needs at least two unlabeled reference examples")
        if tuple(self.prompt_mappings) != PROMPT_MAPPINGS:
            raise ValueError("Artifact prompt mappings differ from the fixed protocol")
        if tuple(self.query_views) != QUERY_VIEWS or tuple(self.key_views) != KEY_VIEWS:
            raise ValueError("Artifact Q/K views differ from the fixed protocol")
        n = len(self.head_indices)
        if n == 0 or len(self.one_mapping_weights) != n or len(self.two_mapping_weights) != n:
            raise ValueError("Artifact head indices and weights have inconsistent lengths")
        if self.gold_labels_used != 0 or self.decision_threshold != 0.0:
            raise ValueError("QK reward artifact is not calibration-free")
        for values in (self.one_mapping_weights, self.two_mapping_weights):
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError("Artifact weights must be finite and non-negative")
            if not np.isclose(array.sum(), 1.0, atol=1e-6):
                raise ValueError("Artifact weights must sum to one")

    def save(self, path: os.PathLike[str] | str) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, allow_nan=False))
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> QKRewardArtifact:
        artifact = cls(**json.loads(Path(path).read_text()))
        artifact.validate()
        return artifact


class _BatchedProjectionCapture:
    """Capture Q/K projections at per-example token positions."""

    def __init__(self, model: Any, layer_indices: Sequence[int]):
        import torch

        self.torch = torch
        self.model = model
        self.layers = self._find_layers(model)
        self.layer_indices = list(map(int, layer_indices))
        self.q_positions: Any | None = None
        self.k_positions: Any | None = None
        self.cache: dict[tuple[str, int], Any] = {}
        self.handles: list[Any] = []
        for local, global_layer in enumerate(self.layer_indices):
            q_proj, k_proj = self._get_qk_modules(self.layers[global_layer])
            self.handles.append(q_proj.register_forward_hook(self._hook("q", local)))
            self.handles.append(k_proj.register_forward_hook(self._hook("k", local)))

    @staticmethod
    def _find_layers(model: Any) -> Sequence[Any]:
        candidates = [getattr(model, "model", None), model]
        for base in candidates:
            if base is None:
                continue
            if hasattr(base, "layers"):
                return base.layers
            if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
                return base.decoder.layers
            if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
                return base.transformer.h
        raise ValueError("Could not locate decoder layers")

    @staticmethod
    def _get_qk_modules(layer: Any) -> tuple[Any, Any]:
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attention", None)
            or getattr(layer, "attn", None)
        )
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "k_proj"):
            raise ValueError("Attention layer has no q_proj/k_proj modules")
        return attn.q_proj, attn.k_proj

    def _hook(self, kind: str, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            values = output[0] if isinstance(output, tuple) else output
            positions = self.q_positions if kind == "q" else self.k_positions
            if positions is None:
                raise RuntimeError("Projection hook fired without configured positions")
            batch = self.torch.arange(values.shape[0], device=values.device)[:, None]
            self.cache[(kind, layer)] = values[batch, positions].detach()

        return hook

    def capture(
        self,
        backbone: Any,
        encoded: dict[str, Any],
        q_positions: Any,
        k_positions: Any,
    ) -> dict[tuple[str, int], Any]:
        self.cache = {}
        self.q_positions = q_positions
        self.k_positions = k_positions
        with self.torch.inference_mode():
            output = backbone(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=False,
            )
        del output
        return self.cache

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FrozenQKRewardScorer:
    """GPU scorer for a fixed zero-label CF-QK-RSym reward."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        max_length: int = 2048,
        micro_batch_size: int = 2,
        layer_start_fraction: float = 0.5,
        model_revision: str | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        requested_revision = model_revision
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.micro_batch_size = int(micro_batch_size)
        self.layer_start_fraction = float(layer_start_fraction)
        if self.max_length < 128 or self.micro_batch_size < 1:
            raise ValueError("max_length must be >=128 and micro_batch_size must be positive")
        if not 0.0 <= self.layer_start_fraction < 1.0:
            raise ValueError("layer_start_fraction must be in [0, 1)")
        config = AutoConfig.from_pretrained(
            model_id,
            revision=requested_revision,
            trust_remote_code=True,
        )
        self.model_revision = getattr(config, "_commit_hash", None) or requested_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=self.model_revision,
            trust_remote_code=True,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("QK reward requires a fast tokenizer with offset mappings")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.n_layers = int(config.num_hidden_layers)
        self.n_heads = int(config.num_attention_heads)
        self.n_kv_heads = int(getattr(config, "num_key_value_heads", self.n_heads))
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.n_heads))
        if self.n_heads % self.n_kv_heads:
            raise ValueError("Query head count must be divisible by KV head count")
        first = math.floor(self.layer_start_fraction * self.n_layers)
        self.layer_indices = list(range(first, self.n_layers))
        self.head_indices = [
            layer * self.n_heads + head
            for layer in self.layer_indices
            for head in range(self.n_heads)
        ]
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "revision": self.model_revision,
            "trust_remote_code": True,
            "dtype": dtype_map[dtype],
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except (ImportError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.backbone = getattr(self.model, "model", self.model)
        self.capture = _BatchedProjectionCapture(self.model, self.layer_indices)
        self.kv_for_query = torch.arange(self.n_heads, device=self.device) // (
            self.n_heads // self.n_kv_heads
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "n_layers": self.n_layers,
            "n_query_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "layer_start_fraction": self.layer_start_fraction,
            "head_indices": self.head_indices,
        }

    def _mapping_scores(
        self,
        problems: Sequence[str],
        completions: Sequence[str],
        mapping: str,
    ) -> np.ndarray:
        texts: list[str] = []
        query_chars: list[list[int]] = []
        key_chars: list[list[int]] = []
        semantics: list[list[int]] = []
        for problem, completion in zip(problems, completions):
            text, queries, option_keys, semantic = build_qk_verifier_prompt(
                problem,
                completion,
                self.tokenizer,
                self.max_length,
                mapping,
            )
            texts.append(text)
            query_chars.append(queries)
            key_chars.append([position for option in option_keys for position in option])
            semantics.append(semantic)

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping").tolist()
        query_positions = self.torch.as_tensor(
            [_char_to_token_positions(row, chars) for row, chars in zip(offsets, query_chars)],
            device=self.device,
            dtype=self.torch.long,
        )
        key_positions = self.torch.as_tensor(
            [_char_to_token_positions(row, chars) for row, chars in zip(offsets, key_chars)],
            device=self.device,
            dtype=self.torch.long,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        cache = self.capture.capture(
            self.backbone,
            encoded,
            query_positions,
            key_positions,
        )

        batch = len(texts)
        per_layer: list[Any] = []
        for local, _global_layer in enumerate(self.layer_indices):
            q_raw = cache[("q", local)]
            k_raw = cache[("k", local)]
            q = q_raw.view(batch, 3, self.n_heads, self.head_dim).float()
            k = k_raw.view(batch, 2, 3, self.n_kv_heads, self.head_dim).float()
            k = k[:, :, :, self.kv_for_query, :]
            # [batch, head, query-view, key-view, surface-option]
            surface = self.torch.einsum("bqhd,bokhd->bhqko", q, k)
            semantic = self.torch.empty_like(surface)
            if semantics[0] == [0, 1]:
                semantic.copy_(surface)
            else:
                semantic[..., 0] = surface[..., 1]
                semantic[..., 1] = surface[..., 0]
            per_layer.append(semantic.reshape(batch, self.n_heads, 9, 2).cpu())
        del cache, encoded
        return self.torch.cat(per_layer, dim=1).numpy()

    def semantic_scores(
        self,
        problems: Sequence[str],
        completions: Sequence[str],
    ) -> np.ndarray:
        if len(problems) != len(completions):
            raise ValueError("problems and completions must have the same length")
        chunks: list[np.ndarray] = []
        for start in range(0, len(problems), self.micro_batch_size):
            stop = min(len(problems), start + self.micro_batch_size)
            mapping_scores = [
                self._mapping_scores(problems[start:stop], completions[start:stop], mapping)
                for mapping in PROMPT_MAPPINGS
            ]
            chunks.append(np.stack(mapping_scores, axis=2))
        if not chunks:
            return np.empty((0, len(self.head_indices), 2, 9, 2), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def head_votes(
        self,
        problems: Sequence[str],
        completions: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.semantic_scores(problems, completions)
        return (
            head_votes_from_semantic_scores(scores, mapping_count=1),
            head_votes_from_semantic_scores(scores, mapping_count=2),
        )

    def rewards(
        self,
        problems: Sequence[str],
        completions: Sequence[str],
        artifact: QKRewardArtifact,
    ) -> np.ndarray:
        artifact.validate()
        if artifact.model_id != self.model_id:
            raise ValueError(f"Artifact model {artifact.model_id} != scorer model {self.model_id}")
        if artifact.model_revision and artifact.model_revision != self.model_revision:
            raise ValueError(
                f"Artifact revision {artifact.model_revision} != scorer revision {self.model_revision}"
            )
        if artifact.head_indices != self.head_indices:
            raise ValueError("Artifact and scorer head pools differ")
        one, two = self.head_votes(problems, completions)
        return aggregate_frozen_rsym(
            one,
            two,
            np.asarray(artifact.one_mapping_weights),
            np.asarray(artifact.two_mapping_weights),
        )

    def close(self) -> None:
        self.capture.close()


def make_reward_artifact(
    scorer: FrozenQKRewardScorer,
    problems: Sequence[str],
    completions: Sequence[str],
    reference_seed: int,
) -> QKRewardArtifact:
    """Fit and return a frozen zero-label reward artifact."""
    one, two = scorer.head_votes(problems, completions)
    metadata = scorer.metadata
    return QKRewardArtifact(
        schema_version=1,
        method="frozen-cf-qk-rsym",
        model_id=metadata["model_id"],
        model_revision=metadata["model_revision"],
        n_layers=metadata["n_layers"],
        n_query_heads=metadata["n_query_heads"],
        n_kv_heads=metadata["n_kv_heads"],
        layer_start_fraction=metadata["layer_start_fraction"],
        head_indices=metadata["head_indices"],
        one_mapping_weights=fit_responsiveness_weights(one).tolist(),
        two_mapping_weights=fit_responsiveness_weights(two).tolist(),
        unlabeled_reference_size=len(problems),
        reference_seed=int(reference_seed),
        prompt_mappings=list(PROMPT_MAPPINGS),
        query_views=list(QUERY_VIEWS),
        key_views=list(KEY_VIEWS),
    )
