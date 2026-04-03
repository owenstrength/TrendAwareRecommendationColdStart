import numpy as np

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures
from trend_aware_recs.models.baseline import content_nn_rerank, no_boost_baseline, popularity_rerank


def _make_items() -> dict[str, ItemFeatures]:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.9, 0.1])
    v3 = np.array([0.0, 1.0])
    return {
        "popular": ItemFeatures("popular", v1.copy(), v1.copy(), v1.copy(), 50),
        "cold_close": ItemFeatures("cold_close", v2.copy(), v2.copy(), v2.copy(), 2),
        "cold_far": ItemFeatures("cold_far", v3.copy(), v3.copy(), v3.copy(), 0),
    }


def _make_candidates() -> list[CandidateScore]:
    return [
        CandidateScore("popular", 0.4),
        CandidateScore("cold_close", 0.5),
        CandidateScore("cold_far", 0.45),
    ]


def test_no_boost_baseline_preserves_base_score_order() -> None:
    candidates = _make_candidates()
    result = no_boost_baseline(candidates)
    ids = [r.item_id for r in result]
    # Should be ordered by base_score descending: cold_close (0.5), cold_far (0.45), popular (0.4)
    assert ids == ["cold_close", "cold_far", "popular"]
    assert all(r.method_score == 0.0 for r in result)


def test_popularity_rerank_promotes_popular_items() -> None:
    items = _make_items()
    candidates = _make_candidates()
    result = popularity_rerank(candidates, items, boost_weight=1.0)
    # "popular" has the highest interaction_count and should rise to the top
    assert result[0].item_id == "popular"


def test_content_nn_rerank_promotes_similar_to_history() -> None:
    items = _make_items()
    candidates = [CandidateScore("cold_close", 0.4), CandidateScore("cold_far", 0.4)]
    # User has interacted with the "popular" item (visual=[1,0]), so cold_close should rank higher
    interactions = [Interaction("u1", "popular", 100)]
    result = content_nn_rerank(
        user_id="u1",
        candidates=candidates,
        interactions=interactions,
        items=items,
        modality_weights={"visual": 1.0, "text": 0.0, "audio": 0.0},
        boost_weight=0.5,
    )
    assert result[0].item_id == "cold_close"


def test_content_nn_rerank_no_history_returns_base_score_order() -> None:
    items = _make_items()
    candidates = [CandidateScore("cold_close", 0.6), CandidateScore("cold_far", 0.3)]
    result = content_nn_rerank(
        user_id="u_unknown",
        candidates=candidates,
        interactions=[],
        items=items,
    )
    # No history → similarity = 0 for all, order by base score
    assert result[0].item_id == "cold_close"
