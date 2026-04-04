"""Utilities for creating cold-start vs. warm evaluation splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from trend_aware_recs.data.schema import EvaluationCase, Interaction, ItemFeatures
from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from trend_aware_recs.models.baseline import build_base_candidates


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


@dataclass
class TemporalHoldout:
    """Training interactions and user-level held-out evaluation cases."""

    cutoff_time: int
    train_interactions: list[Interaction]
    train_item_counts: dict[str, int]
    cases: list[EvaluationCase]


@dataclass
class AggregateMetrics:
    """Average ranking metrics over multiple held-out cases."""

    split: str
    hit_rate: float
    ndcg: float
    num_cases: int


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


def build_time_based_holdout(
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    seed: int,
    evaluation_fraction: float,
    negatives_per_user: int,
    min_user_train_interactions: int,
    max_eval_users: int | None,
    cold_item_max_interactions: int,
    modality_weights: dict[str, float],
    base_history_weight: float,
    base_popularity_weight: float,
) -> TemporalHoldout:
    """Create a global time-based train/test split with one held-out positive per user."""
    if not interactions:
        return TemporalHoldout(
            cutoff_time=0,
            train_interactions=[],
            train_item_counts={},
            cases=[],
        )

    ordered = sorted(interactions, key=lambda interaction: interaction.timestamp)
    cutoff_index = max(1, min(len(ordered) - 1, int(len(ordered) * (1.0 - evaluation_fraction))))
    cutoff_time = ordered[cutoff_index - 1].timestamp

    train_interactions: list[Interaction] = []
    train_item_counts: Counter[str] = Counter()
    for interaction in ordered:
        if interaction.timestamp <= cutoff_time:
            train_interactions.append(interaction)
            train_item_counts[interaction.item_id] += 1
    case_defs: list[tuple[str, Interaction, list[Interaction]]] = []
    chosen_users: set[str] = set()
    user_histories: dict[str, list[Interaction]] = defaultdict(list)
    user_seen_items: dict[str, set[str]] = defaultdict(set)
    for interaction in ordered:
        history = user_histories[interaction.user_id]
        seen_items = user_seen_items[interaction.user_id]
        if (
            interaction.timestamp > cutoff_time
            and interaction.user_id not in chosen_users
            and interaction.item_id in items
            and len(history) >= min_user_train_interactions
            and interaction.item_id not in seen_items
        ):
            case_defs.append((interaction.user_id, interaction, history.copy()))
            chosen_users.add(interaction.user_id)

        history.append(interaction)
        seen_items.add(interaction.item_id)

    rng = random.Random(seed)
    if max_eval_users is not None and len(case_defs) > max_eval_users:
        rng.shuffle(case_defs)
        case_defs = case_defs[:max_eval_users]

    case_defs.sort(key=lambda row: (row[1].timestamp, row[0], row[1].item_id))
    cases: list[EvaluationCase] = []
    prefix_item_counts: Counter[str] = Counter()
    pointer = 0
    for user_id, positive, history in case_defs:
        while pointer < len(ordered) and ordered[pointer].timestamp < positive.timestamp:
            prefix_item_counts[ordered[pointer].item_id] += 1
            pointer += 1

        seen_items = {interaction.item_id for interaction in history if interaction.item_id in items}
        negative_pool = [
            item_id
            for item_id in sorted(prefix_item_counts)
            if item_id != positive.item_id and item_id not in seen_items
        ]
        if not negative_pool:
            continue

        if negatives_per_user <= 0:
            negatives = negative_pool
        else:
            sample_size = min(negatives_per_user, len(negative_pool))
            negatives = rng.sample(negative_pool, k=sample_size)
        history_items = [items[interaction.item_id] for interaction in history if interaction.item_id in items]
        candidates = build_base_candidates(
            candidate_item_ids=[positive.item_id, *negatives],
            history_items=history_items,
            items=items,
            popularity_counts=dict(prefix_item_counts),
            modality_weights=modality_weights,
            history_weight=base_history_weight,
            popularity_weight=base_popularity_weight,
        )
        cases.append(
            EvaluationCase(
                user_id=user_id,
                positive_item_id=positive.item_id,
                positive_is_cold=prefix_item_counts.get(positive.item_id, 0)
                <= cold_item_max_interactions,
                timestamp=positive.timestamp,
                candidates=candidates,
                history_interactions=history,
            )
        )

    return TemporalHoldout(
        cutoff_time=cutoff_time,
        train_interactions=train_interactions,
        train_item_counts=dict(train_item_counts),
        cases=cases,
    )


def aggregate_case_metrics(
    ranked_cases: list[tuple[EvaluationCase, list[str]]],
    k: int,
) -> dict[str, AggregateMetrics]:
    """Average metrics across held-out cases, with cold-vs-warm breakdown."""
    buckets: dict[str, list[tuple[float, float]]] = {"overall": [], "cold": [], "warm": []}
    for case, ranked_items in ranked_cases:
        relevant_items = {case.positive_item_id}
        values = (
            hit_rate_at_k(ranked_items, relevant_items, k),
            ndcg_at_k(ranked_items, relevant_items, k),
        )
        buckets["overall"].append(values)
        buckets["cold" if case.positive_is_cold else "warm"].append(values)

    summary: dict[str, AggregateMetrics] = {}
    for split, values in buckets.items():
        if not values:
            summary[split] = AggregateMetrics(split=split, hit_rate=0.0, ndcg=0.0, num_cases=0)
            continue
        summary[split] = AggregateMetrics(
            split=split,
            hit_rate=sum(hit for hit, _ in values) / len(values),
            ndcg=sum(ndcg for _, ndcg in values) / len(values),
            num_cases=len(values),
        )
    return summary
