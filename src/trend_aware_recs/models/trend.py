from __future__ import annotations

from collections import Counter

from trend_aware_recs.data.schema import Interaction


def select_trending_item_ids(
    counts: Counter[str],
    top_fraction: float,
    min_items: int = 1,
) -> list[str]:
    if not counts:
        return []

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keep = max(min_items, int(len(ranked) * top_fraction))
    keep = min(len(ranked), keep)
    return [item_id for item_id, _ in ranked[:keep]]


def score_emerging_item_ids(
    recent_counts: Counter[str],
    historical_counts: Counter[str],
    *,
    smoothing: float = 1.0,
    min_recent_interactions: int = 2,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item_id, recent_count in recent_counts.items():
        if recent_count < min_recent_interactions:
            continue
        historical_count = historical_counts.get(item_id, 0)
        scores[item_id] = recent_count / (historical_count + smoothing)
    return scores


def select_top_scored_item_ids(
    scores: dict[str, float],
    top_fraction: float,
    min_items: int = 1,
) -> list[str]:
    if not scores or top_fraction <= 0.0:
        return []

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    keep = max(min_items, int(len(ranked) * top_fraction))
    keep = min(len(ranked), keep)
    return [item_id for item_id, _ in ranked[:keep]]


def identify_trending_items(
    interactions: list[Interaction],
    current_time: int,
    trend_window_hours: int,
    top_fraction: float,
    min_items: int = 1,
) -> list[str]:
    window_seconds = trend_window_hours * 3600
    recent = [
        interaction.item_id
        for interaction in interactions
        if current_time - interaction.timestamp <= window_seconds
    ]
    if not recent:
        return []

    return select_trending_item_ids(Counter(recent), top_fraction=top_fraction, min_items=min_items)
