import numpy as np

from trend_aware_recs.data.schema import ItemFeatures
from trend_aware_recs.evaluation.splits import compute_split_metrics, partition_items


def _make_items() -> dict[str, ItemFeatures]:
    return {
        "cold_a": ItemFeatures("cold_a", np.zeros(2), np.zeros(2), np.zeros(2), 0),
        "cold_b": ItemFeatures("cold_b", np.zeros(2), np.zeros(2), np.zeros(2), 5),
        "warm_a": ItemFeatures("warm_a", np.zeros(2), np.zeros(2), np.zeros(2), 20),
        "warm_b": ItemFeatures("warm_b", np.zeros(2), np.zeros(2), np.zeros(2), 50),
    }


def test_partition_items_splits_correctly() -> None:
    items = _make_items()
    cold, warm = partition_items(list(items.keys()), items, cold_item_max_interactions=10)
    assert set(cold) == {"cold_a", "cold_b"}
    assert set(warm) == {"warm_a", "warm_b"}


def test_compute_split_metrics_cold_ndcg() -> None:
    items = _make_items()
    ranked = ["cold_a", "cold_b", "warm_a", "warm_b"]
    relevant = {"cold_a", "warm_a"}
    metrics = compute_split_metrics(
        ranked_items=ranked,
        relevant_items=relevant,
        items=items,
        cold_item_max_interactions=10,
        k=4,
        user_id="u1",
    )
    # overall: both cold_a and warm_a are in top-4 and relevant
    assert metrics.overall.hit_rate == 1.0
    assert metrics.cold.hit_rate == 1.0  # cold_a is relevant and in top cold rank
    assert metrics.warm.hit_rate == 1.0  # warm_a is relevant and in top warm rank
