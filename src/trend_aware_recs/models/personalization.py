from __future__ import annotations

from collections import Counter

from trend_aware_recs.data.schema import Interaction


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
    return float((trending_events + smoothing) / (total_events + 2 * smoothing))
