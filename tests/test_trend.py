from collections import Counter

from trend_aware_recs.models.trend import (
    score_emerging_item_ids,
    select_top_scored_item_ids,
    select_trending_item_ids,
)


def test_select_trending_item_ids_is_deterministic_on_ties() -> None:
    counts = Counter({"b": 2, "a": 2, "c": 1})
    assert select_trending_item_ids(counts, top_fraction=0.67, min_items=2) == ["a", "b"]


def test_score_emerging_item_ids_downweights_historical_volume() -> None:
    recent = Counter({"hot_new": 4, "steady_hit": 10, "tiny": 1})
    historical = Counter({"hot_new": 4, "steady_hit": 100, "tiny": 0})
    scores = score_emerging_item_ids(
        recent,
        historical,
        smoothing=1.0,
        min_recent_interactions=2,
    )

    assert "tiny" not in scores
    assert scores["hot_new"] > scores["steady_hit"]


def test_select_top_scored_item_ids_prefers_high_scores_then_item_id() -> None:
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    assert select_top_scored_item_ids(scores, top_fraction=0.67, min_items=2) == ["a", "b"]
