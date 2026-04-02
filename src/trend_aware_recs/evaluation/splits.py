"""Utilities for creating cold-start vs. warm evaluation splits.

The proposal requires evaluation metrics broken down by item type
(cold-start vs. warm) to verify that the trend boost helps new items
without hurting overall quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trend_aware_recs.data.schema import ItemFeatures
from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k


@dataclass
class SplitMetrics:
    """Evaluation metrics for a single split (cold-start or warm)."""

    split: str
    hit_rate: float
    ndcg: float
    num_items: int


@dataclass
class UserMetrics:
    """All metric splits for a single user."""

    user_id: str
    overall: SplitMetrics
    cold: SplitMetrics
    warm: SplitMetrics
    extra: dict[str, float] = field(default_factory=dict)


def partition_items(
    item_ids: list[str],
    items: dict[str, ItemFeatures],
    cold_item_max_interactions: int,
) -> tuple[list[str], list[str]]:
    """Split *item_ids* into (cold_ids, warm_ids) based on interaction count."""
    cold = [
        iid
        for iid in item_ids
        if iid in items and items[iid].interaction_count <= cold_item_max_interactions
    ]
    warm = [
        iid
        for iid in item_ids
        if iid in items and items[iid].interaction_count > cold_item_max_interactions
    ]
    return cold, warm


def compute_split_metrics(
    ranked_items: list[str],
    relevant_items: set[str],
    items: dict[str, ItemFeatures],
    cold_item_max_interactions: int,
    k: int,
    user_id: str,
) -> UserMetrics:
    """Compute hit-rate@k and NDCG@k broken down by cold-start and warm splits."""
    cold_ranked, warm_ranked = partition_items(ranked_items, items, cold_item_max_interactions)
    cold_relevant = {
        iid
        for iid in relevant_items
        if iid in items and items[iid].interaction_count <= cold_item_max_interactions
    }
    warm_relevant = {
        iid
        for iid in relevant_items
        if iid in items and items[iid].interaction_count > cold_item_max_interactions
    }

    overall = SplitMetrics(
        split="overall",
        hit_rate=hit_rate_at_k(ranked_items, relevant_items, k),
        ndcg=ndcg_at_k(ranked_items, relevant_items, k),
        num_items=len(ranked_items),
    )
    cold = SplitMetrics(
        split="cold",
        hit_rate=hit_rate_at_k(cold_ranked, cold_relevant, k),
        ndcg=ndcg_at_k(cold_ranked, cold_relevant, k),
        num_items=len(cold_ranked),
    )
    warm = SplitMetrics(
        split="warm",
        hit_rate=hit_rate_at_k(warm_ranked, warm_relevant, k),
        ndcg=ndcg_at_k(warm_ranked, warm_relevant, k),
        num_items=len(warm_ranked),
    )

    return UserMetrics(user_id=user_id, overall=overall, cold=cold, warm=warm)
