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
    trend_item_features: list[ItemFeatures] | None = None,
    precomputed_affinity: float | None = None,
    item_interaction_counts: dict[str, int] | None = None,
    extra_candidate_boosts: dict[str, float] | None = None,
    personalization_strength: float = 1.0,
) -> list[RankedScore]:
    trend_items = (
        trend_item_features
        if trend_item_features is not None
        else [items[item_id] for item_id in trending_item_ids if item_id in items]
    )
    raw_affinity = (
        1.0
        if disable_personalization
        else (
            precomputed_affinity
            if precomputed_affinity is not None
            else compute_user_virality_affinity(user_id, interactions, trending_item_ids, smoothing)
        )
    )
    affinity = 1.0 - personalization_strength + personalization_strength * raw_affinity

    scored: list[RankedScore] = []
    for candidate in candidates:
        item = items[candidate.item_id]
        if not trend_items:
            similarity = 0.0
        elif len(trend_items) == 1:
            similarity = max(0.0, multimodal_similarity(item, trend_items, modality_weights))
        else:
            similarity = max(
                0.0,
                max(
                    multimodal_similarity(item, [trend_item], modality_weights)
                    for trend_item in trend_items
                ),
            )
        interaction_count = (
            item_interaction_counts.get(candidate.item_id, item.interaction_count)
            if item_interaction_counts is not None
            else item.interaction_count
        )
        cold_factor = 1.0 if interaction_count <= cold_item_max_interactions else 0.0
        extra_boost = 0.0 if extra_candidate_boosts is None else extra_candidate_boosts.get(candidate.item_id, 0.0)
        boost = boost_weight * affinity * similarity * cold_factor + extra_boost * cold_factor
        scored.append(
            RankedScore(
                item_id=candidate.item_id,
                base_score=candidate.base_score,
                similarity_score=similarity,
                boost=boost,
                final_score=candidate.base_score + boost,
            )
        )

    return sorted(scored, key=lambda score: (-score.final_score, score.item_id))
