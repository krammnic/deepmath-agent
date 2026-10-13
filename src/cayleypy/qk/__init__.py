"""Calibration-free QK reasoning scores for CayleyPy research."""

from .core import (
    KEY_VIEWS,
    PROMPT_MAPPINGS,
    QUERY_VIEWS,
    FrozenQKRewardScorer,
    QKRewardArtifact,
    aggregate_frozen_rsym,
    build_qk_verifier_prompt,
    fit_responsiveness_weights,
    head_votes_from_semantic_scores,
    make_reward_artifact,
)

ZeroShotQKScorer = FrozenQKRewardScorer
ZeroShotQKArtifact = QKRewardArtifact
fit_zero_shot_artifact = make_reward_artifact

__all__ = [
    "KEY_VIEWS",
    "PROMPT_MAPPINGS",
    "QUERY_VIEWS",
    "ZeroShotQKArtifact",
    "ZeroShotQKScorer",
    "aggregate_frozen_rsym",
    "build_qk_verifier_prompt",
    "fit_responsiveness_weights",
    "fit_zero_shot_artifact",
    "head_votes_from_semantic_scores",
]
