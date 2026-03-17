from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(["i1", "i2", "i3"], {"i3"}, 2) == 0.0
    assert hit_rate_at_k(["i1", "i2", "i3"], {"i3"}, 3) == 1.0


def test_ndcg_at_k() -> None:
    score = ndcg_at_k(["i1", "i2", "i3"], {"i1", "i3"}, 3)
    assert 0.9 < score <= 1.0
