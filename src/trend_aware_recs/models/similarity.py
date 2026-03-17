from __future__ import annotations

import math

import numpy as np

from trend_aware_recs.data.schema import ItemFeatures


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(left, right) / denominator)


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
