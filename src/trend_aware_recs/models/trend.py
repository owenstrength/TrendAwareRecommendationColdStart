from __future__ import annotations

from collections import Counter

from trend_aware_recs.data.schema import Interaction


def identify_trending_items(
    interactions: list[Interaction],
    current_time: int,
    trend_window_hours: int,
    top_fraction: float,
) -> list[str]:
    window_seconds = trend_window_hours * 3600
    recent = [
        interaction.item_id
        for interaction in interactions
        if current_time - interaction.timestamp <= window_seconds
    ]
    if not recent:
        return []

    ranked = Counter(recent).most_common()
    keep = max(1, int(len(ranked) * top_fraction))
    return [item_id for item_id, _ in ranked[:keep]]
