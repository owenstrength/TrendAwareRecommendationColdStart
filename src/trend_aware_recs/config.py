from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str
    interaction_path: str
    item_feature_paths: dict[str, str]
    item_metadata_path: str
    timestamp_column: str
    user_column: str
    item_column: str
    popularity_column: str


class ModelConfig(BaseModel):
    trend_window_hours: int = 72
    top_trend_fraction: float = 0.1
    cold_item_max_interactions: int = 10
    similarity_metric: str = "cosine"
    modality_weights: dict[str, float] = Field(default_factory=dict)
    boost_weight: float = 0.35
    virality_prior_smoothing: float = 5.0


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
