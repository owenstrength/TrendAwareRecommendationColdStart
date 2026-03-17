import numpy as np

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures
from trend_aware_recs.models.reranker import rerank_with_trend_boost


def test_reranker_boosts_cold_items_close_to_trends() -> None:
    items = {
        "trend": ItemFeatures("trend", np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]), 20),
        "cold_close": ItemFeatures(
            "cold_close",
            np.array([0.95, 0.05]),
            np.array([0.95, 0.05]),
            np.array([0.95, 0.05]),
            0,
        ),
        "cold_far": ItemFeatures(
            "cold_far",
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            0,
        ),
    }
    interactions = [Interaction("u1", "trend", 10), Interaction("u1", "trend", 11)]
    ranked = rerank_with_trend_boost(
        user_id="u1",
        candidates=[CandidateScore("cold_far", 0.5), CandidateScore("cold_close", 0.45)],
        interactions=interactions,
        items=items,
        trending_item_ids=["trend"],
        cold_item_max_interactions=10,
        modality_weights={"visual": 1 / 3, "text": 1 / 3, "audio": 1 / 3},
        boost_weight=0.5,
        smoothing=1.0,
    )
    assert ranked[0].item_id == "cold_close"
