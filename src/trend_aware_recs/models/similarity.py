from __future__ import annotations

import math

import numpy as np

from trend_aware_recs.data.schema import ItemFeatures


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(left, right) / denominator)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return vector
    return vector / norm


def average_item_features(item_id: str, source_items: list[ItemFeatures]) -> ItemFeatures:
    """Aggregate a list of items into a single centroid representation."""
    if not source_items:
        raise ValueError("source_items must not be empty")

    visual = _normalize(np.mean([item.visual for item in source_items], axis=0))
    text = _normalize(np.mean([item.text for item in source_items], axis=0))
    audio = _normalize(np.mean([item.audio for item in source_items], axis=0))
    return ItemFeatures(
        item_id=item_id,
        visual=visual,
        text=text,
        audio=audio,
        interaction_count=int(round(np.mean([item.interaction_count for item in source_items]))),
    )


def weighted_average_item_features(
    item_id: str,
    source_items: list[ItemFeatures],
    weights: list[float],
) -> ItemFeatures:
    """Aggregate items into a single weighted centroid representation."""
    if not source_items:
        raise ValueError("source_items must not be empty")
    if len(source_items) != len(weights):
        raise ValueError("source_items and weights must have the same length")

    arr = np.asarray(weights, dtype=float)
    if arr.sum() <= 0:
        arr = np.ones(len(source_items), dtype=float)
    arr = arr / arr.sum()

    visual = _normalize(np.average([item.visual for item in source_items], axis=0, weights=arr))
    text = _normalize(np.average([item.text for item in source_items], axis=0, weights=arr))
    audio = _normalize(np.average([item.audio for item in source_items], axis=0, weights=arr))
    return ItemFeatures(
        item_id=item_id,
        visual=visual,
        text=text,
        audio=audio,
        interaction_count=int(round(np.average([item.interaction_count for item in source_items], weights=arr))),
    )


def multimodal_similarity(
    item: ItemFeatures,
    trend_items: list[ItemFeatures],
    modality_weights: dict[str, float],
) -> float:
    if not trend_items:
        return 0.0

    weights = modality_weights or {"visual": 1 / 3, "text": 1 / 3, "audio": 1 / 3}
    scores = []
    for trend_item in trend_items:
        score = (
            weights.get("visual", 0.0) * cosine_similarity(item.visual, trend_item.visual)
            + weights.get("text", 0.0) * cosine_similarity(item.text, trend_item.text)
            + weights.get("audio", 0.0) * cosine_similarity(item.audio, trend_item.audio)
        )
        scores.append(score)
    return float(np.mean(scores))
