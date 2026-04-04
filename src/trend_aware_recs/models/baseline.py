"""Baseline recommenders for comparison against the trend-aware boost.

The proposal requires comparing against at least:
  (i)  a base recommender with no boost (identity – just return candidates as-is),
  (ii) a popularity-based heuristic, and
  (iii) a content-only nearest-neighbour baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures
from trend_aware_recs.models.similarity import average_item_features, cosine_similarity, multimodal_similarity


@dataclass(frozen=True)
class BaselineScore:
    """A candidate item with its baseline ranking score."""

    item_id: str
    base_score: float
    method_score: float
    final_score: float


def build_base_candidates(
    candidate_item_ids: list[str],
    history_items: list[ItemFeatures],
    items: dict[str, ItemFeatures],
    popularity_counts: dict[str, int],
    modality_weights: dict[str, float] | None = None,
    history_weight: float = 0.7,
    popularity_weight: float = 0.3,
) -> list[CandidateScore]:
    """Build a lightweight hybrid base score for each candidate item."""
    weights = modality_weights or {"visual": 1 / 3, "text": 1 / 3, "audio": 1 / 3}
    history_profile = average_item_features("user_profile", history_items) if history_items else None
    max_popularity = max(popularity_counts.values(), default=0)
    max_popularity_log = math.log1p(max_popularity) if max_popularity > 0 else 1.0

    candidates: list[CandidateScore] = []
    for item_id in candidate_item_ids:
        item = items.get(item_id)
        if item is None:
            continue

        history_similarity = 0.0
        if history_profile is not None:
            history_similarity = max(
                0.0,
                multimodal_similarity(item, [history_profile], weights),
            )

        popularity = 0.0
        if max_popularity > 0:
            popularity = math.log1p(popularity_counts.get(item_id, 0)) / max_popularity_log

        base_score = history_weight * history_similarity + popularity_weight * popularity
        candidates.append(CandidateScore(item_id=item_id, base_score=base_score))

    return candidates


def no_boost_baseline(candidates: list[CandidateScore]) -> list[BaselineScore]:
    """Return candidates ranked purely by their base score (no boost)."""
    ranked = sorted(candidates, key=lambda c: (-c.base_score, c.item_id))
    return [
        BaselineScore(
            item_id=c.item_id,
            base_score=c.base_score,
            method_score=0.0,
            final_score=c.base_score,
        )
        for c in ranked
    ]


def popularity_rerank(
    candidates: list[CandidateScore],
    items: dict[str, ItemFeatures],
    boost_weight: float = 0.35,
    popularity_counts: dict[str, int] | None = None,
) -> list[BaselineScore]:
    """Re-rank candidates by mixing base score with normalised popularity (interaction count).

    This is the popularity-only heuristic baseline: it uses raw interaction counts
    rather than multimodal content signals.
    """
    counts = np.array(
        [
            (
                popularity_counts.get(c.item_id, items[c.item_id].interaction_count)
                if popularity_counts is not None
                else items[c.item_id].interaction_count
            )
            for c in candidates
            if c.item_id in items
        ],
        dtype=float,
    )
    max_count = float(counts.max()) if counts.size > 0 and counts.max() > 0 else 1.0

    scored: list[BaselineScore] = []
    for c in candidates:
        item = items.get(c.item_id)
        raw_popularity = 0.0
        if item is not None:
            raw_popularity = (
                popularity_counts.get(c.item_id, item.interaction_count)
                if popularity_counts is not None
                else item.interaction_count
            )
        norm_pop = (raw_popularity / max_count) if max_count > 0 else 0.0
        final = c.base_score + boost_weight * norm_pop
        scored.append(
            BaselineScore(
                item_id=c.item_id,
                base_score=c.base_score,
                method_score=norm_pop,
                final_score=final,
            )
        )
    return sorted(scored, key=lambda s: (-s.final_score, s.item_id))


def content_nn_rerank(
    user_id: str,
    candidates: list[CandidateScore],
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    modality_weights: dict[str, float] | None = None,
    boost_weight: float = 0.35,
    history_items: list[ItemFeatures] | None = None,
) -> list[BaselineScore]:
    """Re-rank candidates by content similarity to the user's interaction history.

    This is the content-only nearest-neighbour baseline: it leverages multimodal
    content signals but *not* trend information or personalized scaling.
    """
    weights = modality_weights or {"visual": 1 / 3, "text": 1 / 3, "audio": 1 / 3}

    # Collect items the user has already interacted with.
    resolved_history_items = history_items
    if resolved_history_items is None:
        seen_ids = {i.item_id for i in interactions if i.user_id == user_id}
        resolved_history_items = [items[iid] for iid in seen_ids if iid in items]

    scored: list[BaselineScore] = []
    for c in candidates:
        item = items.get(c.item_id)
        if item is None or not resolved_history_items:
            sim = 0.0
        else:
            sims = []
            for hist_item in resolved_history_items:
                s = (
                    weights.get("visual", 0.0) * cosine_similarity(item.visual, hist_item.visual)
                    + weights.get("text", 0.0) * cosine_similarity(item.text, hist_item.text)
                    + weights.get("audio", 0.0) * cosine_similarity(item.audio, hist_item.audio)
                )
                sims.append(s)
            sim = float(np.mean(sims))
        final = c.base_score + boost_weight * sim
        scored.append(
            BaselineScore(
                item_id=c.item_id,
                base_score=c.base_score,
                method_score=sim,
                final_score=final,
            )
        )
    return sorted(scored, key=lambda s: (-s.final_score, s.item_id))
