from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str
    loader: str = "demo"
    interaction_path: str | None = None
    item_feature_paths: dict[str, str] = Field(default_factory=dict)
    derived_feature_dir: str | None = None
    item_metadata_path: str | None = None
    title_path: str | None = None
    tags_path: str | None = None
    comments_path: str | None = None
    engagement_path: str | None = None
    timestamp_column: str = "timestamp"
    user_column: str = "user"
    item_column: str = "item"
    popularity_column: str = "interaction_count"
    feature_dim: int = 64
    max_comments_per_item: int = 5
    evaluation_fraction: float = 0.2
    negatives_per_user: int = 99
    min_user_train_interactions: int = 3
    max_eval_users: int | None = 1000


class ModelConfig(BaseModel):
    trend_window_hours: int = 72
    top_trend_fraction: float = 0.1
    emerging_fraction: float = 0.0
    min_trend_items: int = 20
    emerging_min_recent_interactions: int = 2
    emerging_smoothing: float = 1.0
    cold_item_max_interactions: int = 10
    similarity_metric: str = "cosine"
    modality_weights: dict[str, float] = Field(default_factory=dict)
    boost_weight: float = 0.35
    tag_boost_weight: float = 0.2
    emerging_boost_weight: float = 0.0
    virality_prior_smoothing: float = 5.0
    personalization_strength: float = 1.0
    base_history_weight: float = 0.7
    base_popularity_weight: float = 0.3
    trainable_ranker_enabled: bool = False
    trainable_model_type: str = "pairwise_linear"
    trainable_max_cases: int = 5000
    trainable_negatives_per_case: int = 10
    trainable_epochs: int = 20
    trainable_learning_rate: float = 0.05
    trainable_l2: float = 1e-4
    trainable_blend_weight: float = 0.0
    trainable_cold_case_weight: float = 1.0
    trainable_hgb_max_iter: int = 200
    trainable_hgb_learning_rate: float = 0.05
    trainable_hgb_max_depth: int | None = 6
    trainable_hgb_min_samples_leaf: int = 20
    trainable_hard_negative_ratio: float = 0.0
    trainable_hard_negative_pool_size: int = 50
    evaluate_trend_aware: bool = True
    evaluate_ablation_no_personalization: bool = True
    evaluate_baseline_no_boost: bool = True
    evaluate_baseline_popularity: bool = True
    evaluate_baseline_content_nn: bool = True


class ExperimentConfig(BaseModel):
    experiment_name: str
    seed: int = 7
    top_k: int = 10
    data_settings_path: str
    boost_settings_path: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_bundle(path: str | Path) -> tuple[ExperimentConfig, DataConfig, ModelConfig]:
    experiment = ExperimentConfig.model_validate(load_yaml(path))
    base_dir = Path(path).parent.parent
    data = DataConfig.model_validate(load_yaml(base_dir.parent / experiment.data_settings_path))
    model = ModelConfig.model_validate(load_yaml(base_dir.parent / experiment.boost_settings_path))
    return experiment, data, model
