from trend_aware_recs.data.demo import get_cold_item_ids, get_trending_item_ids, load_demo_dataset
from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures


def test_load_demo_dataset_returns_expected_structure() -> None:
    interactions, items, user_candidates = load_demo_dataset()

    assert len(interactions) > 0
    assert all(isinstance(i, Interaction) for i in interactions)

    assert len(items) > 0
    for feat in items.values():
        assert isinstance(feat, ItemFeatures)
        assert feat.visual.ndim == 1
        assert feat.text.ndim == 1
        assert feat.audio.ndim == 1

    assert set(user_candidates.keys()) == {"user_trend", "user_niche", "user_mixed"}
    for cands in user_candidates.values():
        assert len(cands) > 0
        assert all(isinstance(c, CandidateScore) for c in cands)


def test_load_demo_dataset_is_reproducible() -> None:
    interactions1, items1, _ = load_demo_dataset(seed=0)
    interactions2, items2, _ = load_demo_dataset(seed=0)
    assert len(interactions1) == len(interactions2)
    for i1, i2 in zip(interactions1, interactions2, strict=True):
        assert i1 == i2


def test_cold_and_warm_partitions_are_disjoint() -> None:
    _, items, _ = load_demo_dataset()
    cold = set(get_cold_item_ids(items, max_interactions=10))
    warm = set(get_trending_item_ids(items))
    assert cold.isdisjoint(warm)
    assert cold | warm == set(items.keys())
