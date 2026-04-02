from __future__ import annotations

from dataclasses import dataclass

from trend_aware_recs.data.schema import CandidateScore, ItemFeatures
from trend_aware_recs.models.personalization import compute_user_virality_affinity
from trend_aware_recs.models.similarity import multimodal_similarity


@dataclass(frozen=True)
class RankedScore:
    item_id: str
    base_score: float
    similarity_score: float
    boost: float
    final_score: float


def rerank_with_trend_boost(
    user_id: str,
    candidates: list[CandidateScore],
    interactions,
    items: dict[str, ItemFeatures],
    trending_item_ids: list[str],
    cold_item_max_interactions: int,
    modality_weights: dict[str, float],
    boost_weight: float,
    smoothing: float,
    disable_personalization: bool = False,
) -> list[RankedScore]:
    trend_items = [items[item_id] for item_id in trending_item_ids if item_id in items]
    affinity = (
        1.0
        if disable_personalization
        else compute_user_virality_affinity(user_id, interactions, trending_item_ids, smoothing)
    )

    scored: list[RankedScore] = []
    for candidate in candidates:
        item = items[candidate.item_id]
        similarity = multimodal_similarity(item, trend_items, modality_weights)
        cold_factor = 1.0 if item.interaction_count <= cold_item_max_interactions else 0.0
        boost = boost_weight * affinity * similarity * cold_factor
        scored.append(
            RankedScore(
                item_id=candidate.item_id,
                base_score=candidate.base_score,
                similarity_score=similarity,
                boost=boost,
                final_score=candidate.base_score + boost,
            )
        )

    return sorted(scored, key=lambda score: score.final_score, reverse=True)
