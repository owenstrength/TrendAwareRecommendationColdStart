"""Baseline recommenders for comparison against the trend-aware boost.

The proposal requires comparing against at least:
  (i)  a base recommender with no boost (identity – just return candidates as-is),
  (ii) a popularity-based heuristic, and
  (iii) a content-only nearest-neighbour baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures
from trend_aware_recs.models.similarity import cosine_similarity


@dataclass(frozen=True)
class BaselineScore:
    """A candidate item with its baseline ranking score."""

    item_id: str
    base_score: float
    method_score: float
    final_score: float


def no_boost_baseline(candidates: list[CandidateScore]) -> list[BaselineScore]:
    """Return candidates ranked purely by their base score (no boost)."""
    ranked = sorted(candidates, key=lambda c: c.base_score, reverse=True)
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
) -> list[BaselineScore]:
    """Re-rank candidates by mixing base score with normalised popularity (interaction count).

    This is the popularity-only heuristic baseline: it uses raw interaction counts
    rather than multimodal content signals.
    """
    counts = np.array(
        [items[c.item_id].interaction_count for c in candidates if c.item_id in items],
        dtype=float,
    )
    max_count = float(counts.max()) if counts.size > 0 and counts.max() > 0 else 1.0

    scored: list[BaselineScore] = []
    for c in candidates:
        item = items.get(c.item_id)
        norm_pop = (item.interaction_count / max_count) if item is not None else 0.0
        final = c.base_score + boost_weight * norm_pop
        scored.append(
            BaselineScore(
                item_id=c.item_id,
                base_score=c.base_score,
                method_score=norm_pop,
                final_score=final,
            )
        )
    return sorted(scored, key=lambda s: s.final_score, reverse=True)


def content_nn_rerank(
    user_id: str,
    candidates: list[CandidateScore],
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    modality_weights: dict[str, float] | None = None,
    boost_weight: float = 0.35,
) -> list[BaselineScore]:
    """Re-rank candidates by content similarity to the user's interaction history.

    This is the content-only nearest-neighbour baseline: it leverages multimodal
    content signals but *not* trend information or personalized scaling.
    """
    weights = modality_weights or {"visual": 1 / 3, "text": 1 / 3, "audio": 1 / 3}

    # Collect items the user has already interacted with.
    seen_ids = {i.item_id for i in interactions if i.user_id == user_id}
    history_items = [items[iid] for iid in seen_ids if iid in items]

    scored: list[BaselineScore] = []
    for c in candidates:
        item = items.get(c.item_id)
        if item is None or not history_items:
            sim = 0.0
        else:
            sims = []
            for hist_item in history_items:
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
    return sorted(scored, key=lambda s: s.final_score, reverse=True)
