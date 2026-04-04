from __future__ import annotations

from collections import Counter

from trend_aware_recs.data.schema import Interaction, ItemFeatures


def compute_virality_affinity_from_counts(
    trending_events: int,
    total_events: int,
    smoothing: float,
) -> float:
    if total_events <= 0:
        return 0.5
    return float((trending_events + smoothing) / (total_events + 2 * smoothing))


def compute_user_virality_affinity(
    user_id: str,
    interactions: list[Interaction],
    trending_items: list[str],
    smoothing: float,
) -> float:
    history = [interaction for interaction in interactions if interaction.user_id == user_id]
    if not history:
        return 0.5

    interaction_counts = Counter(interaction.item_id for interaction in history)
    trending_events = sum(interaction_counts[item_id] for item_id in trending_items)
    total_events = sum(interaction_counts.values())
    return compute_virality_affinity_from_counts(
        trending_events=trending_events,
        total_events=total_events,
        smoothing=smoothing,
    )


def precompute_user_virality_affinities(
    interactions: list[Interaction],
    trending_items: list[str],
    smoothing: float,
) -> dict[str, float]:
    trending_item_set = set(trending_items)
    total_events: Counter[str] = Counter()
    trending_events: Counter[str] = Counter()

    for interaction in interactions:
        total_events[interaction.user_id] += 1
        if interaction.item_id in trending_item_set:
            trending_events[interaction.user_id] += 1

    return {
        user_id: compute_virality_affinity_from_counts(
            trending_events=trending_events[user_id],
            total_events=total_events[user_id],
            smoothing=smoothing,
        )
        for user_id in total_events
    }


def compute_user_tag_affinities(
    history: list[Interaction],
    items: dict[str, ItemFeatures],
) -> dict[str, float]:
    """Return per-tag affinity from a user's history."""
    tag_counts: Counter[str] = Counter()
    total = 0
    for interaction in history:
        item = items.get(interaction.item_id)
        if item is None or not item.tag:
            continue
        tag_counts[item.tag] += 1
        total += 1

    if total == 0:
        return {}
    return {tag: count / total for tag, count in tag_counts.items()}
