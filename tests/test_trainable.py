import numpy as np
from collections import Counter
import random

from trend_aware_recs.data.schema import CandidateScore, ItemFeatures
from trend_aware_recs.models.trainable import (
    FeatureScalers,
    PointwiseHistGradientBoostingRanker,
    _select_training_cases,
    _select_training_negative_ids,
    build_pairwise_ranker_feature_vector,
    build_scalar_ranker_feature_vector,
    score_candidate_with_pointwise_hist_gradient_boosting_ranker,
    score_candidates_with_trained_ranker,
    train_linear_ranker_from_pair_diffs,
    train_hist_gradient_boosting_from_examples,
)


def test_train_linear_ranker_from_pair_diffs_learns_positive_margin() -> None:
    pair_diffs = np.array(
        [
            [2.0, 0.0],
            [1.5, 0.5],
            [1.0, 1.0],
            [0.5, 1.5],
        ],
        dtype=np.float32,
    )

    weights = train_linear_ranker_from_pair_diffs(
        pair_diffs,
        epochs=50,
        learning_rate=0.1,
        l2=1e-4,
        seed=7,
    )

    margins = pair_diffs @ weights
    assert np.all(margins > 0.0)


def test_train_linear_ranker_from_pair_diffs_respects_pair_weights() -> None:
    pair_diffs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    weights = train_linear_ranker_from_pair_diffs(
        pair_diffs,
        pair_weights=np.array([5.0, 1.0], dtype=np.float32),
        epochs=100,
        learning_rate=0.1,
        l2=1e-4,
        seed=7,
    )

    assert weights[0] > weights[1]


def test_select_training_cases_biases_toward_cold_when_weighted() -> None:
    eligible_cases = [(index, index < 3) for index in range(10)]

    selected = _select_training_cases(
        eligible_cases,
        max_cases=5,
        cold_case_weight=4.0,
        seed=7,
    )

    assert len(selected) == 5
    assert sum(1 for is_cold in selected.values() if is_cold) >= 2


def test_train_hist_gradient_boosting_from_examples_learns_simple_boundary() -> None:
    features = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 0, 0], dtype=np.int32)

    model = train_hist_gradient_boosting_from_examples(
        features,
        labels,
        sample_weight=np.ones_like(labels, dtype=np.float32),
        max_iter=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        seed=7,
    )

    probabilities = model.predict_proba(features)[:, 1]
    assert probabilities[0] > probabilities[2]
    assert probabilities[1] > probabilities[3]


def test_pairwise_feature_vector_ignores_snapshot_engagement_metadata() -> None:
    candidate = CandidateScore(item_id="i1", base_score=0.4)
    item_low = ItemFeatures(
        item_id="i1",
        visual=np.array([1.0, 0.0], dtype=np.float32),
        text=np.array([0.0, 1.0], dtype=np.float32),
        audio=np.array([1.0, 0.0], dtype=np.float32),
        interaction_count=100,
        tag="tag_a",
        likes=1,
        views=10,
        comment_count=2,
    )
    item_high = ItemFeatures(
        item_id="i1",
        visual=item_low.visual.copy(),
        text=item_low.text.copy(),
        audio=item_low.audio.copy(),
        interaction_count=100,
        tag="tag_a",
        likes=100000,
        views=500000,
        comment_count=999,
    )

    kwargs = dict(
        history_profile=None,
        prefix_item_counts=Counter({"i1": 3}),
        recent_item_counts=Counter({"i1": 2}),
        trend_item_features=None,
        user_tag_affinities={"tag_a": 0.6},
        tag_uplifts={"tag_a": 0.2},
        max_positive_uplift=0.2,
        emerging_scores={"i1": 1.5},
        max_emerging_score=1.5,
        modality_weights={"visual": 0.4, "text": 0.4, "audio": 0.2},
        cold_item_max_interactions=10,
        scalers=FeatureScalers(),
    )

    low_vector = build_pairwise_ranker_feature_vector(candidate, item_low, **kwargs)
    high_vector = build_pairwise_ranker_feature_vector(candidate, item_high, **kwargs)

    assert np.allclose(low_vector, high_vector)


def test_select_training_negative_ids_prefers_harder_items() -> None:
    negatives = _select_training_negative_ids(
        ["cold", "pop", "recent", "emerging"],
        negatives_per_case=2,
        prefix_item_counts=Counter({"pop": 100, "recent": 3, "emerging": 1}),
        recent_item_counts=Counter({"recent": 20, "emerging": 1}),
        emerging_scores={"emerging": 10.0},
        hard_negative_ratio=1.0,
        hard_negative_pool_size=3,
        rng=random.Random(7),
    )

    assert len(negatives) == 2
    assert "cold" not in negatives


def test_scalar_feature_vector_includes_recent_and_last_item_similarity() -> None:
    candidate = CandidateScore(item_id="target", base_score=0.5)
    target = ItemFeatures(
        item_id="target",
        visual=np.array([1.0, 0.0], dtype=np.float32),
        text=np.array([1.0, 0.0], dtype=np.float32),
        audio=np.array([1.0, 0.0], dtype=np.float32),
        interaction_count=2,
        tag="tag_a",
    )
    close = ItemFeatures(
        item_id="close",
        visual=np.array([1.0, 0.0], dtype=np.float32),
        text=np.array([1.0, 0.0], dtype=np.float32),
        audio=np.array([1.0, 0.0], dtype=np.float32),
        interaction_count=5,
        tag="tag_a",
    )
    far = ItemFeatures(
        item_id="far",
        visual=np.array([0.0, 1.0], dtype=np.float32),
        text=np.array([0.0, 1.0], dtype=np.float32),
        audio=np.array([0.0, 1.0], dtype=np.float32),
        interaction_count=5,
        tag="tag_b",
    )

    vector = build_scalar_ranker_feature_vector(
        candidate,
        target,
        history_profile=far,
        prefix_item_counts=Counter({"target": 1}),
        recent_item_counts=Counter({"target": 1}),
        trend_item_features=None,
        user_tag_affinities={},
        tag_uplifts={},
        max_positive_uplift=0.0,
        emerging_scores={},
        max_emerging_score=0.0,
        modality_weights={"visual": 0.4, "text": 0.4, "audio": 0.2},
        cold_item_max_interactions=10,
        scalers=FeatureScalers(),
        history_items=[far, close],
        recent_history_items=[close],
    )

    assert vector.shape == (14,)
    assert vector[7] < vector[8]
    assert vector[9] == vector[8]


def test_batch_hgb_scoring_matches_per_candidate_path() -> None:
    items = {
        "i1": ItemFeatures(
            item_id="i1",
            visual=np.array([1.0], dtype=np.float32),
            text=np.array([1.0], dtype=np.float32),
            audio=np.array([0.0], dtype=np.float32),
            interaction_count=5,
            tag="tag_a",
        ),
        "i2": ItemFeatures(
            item_id="i2",
            visual=np.array([0.0], dtype=np.float32),
            text=np.array([0.0], dtype=np.float32),
            audio=np.array([1.0], dtype=np.float32),
            interaction_count=20,
            tag="tag_b",
        ),
    }
    candidates = [
        CandidateScore(item_id="i1", base_score=0.8),
        CandidateScore(item_id="i2", base_score=0.2),
    ]
    kwargs = dict(
        history_profile=None,
        prefix_item_counts=Counter({"i1": 2, "i2": 10}),
        recent_item_counts=Counter({"i1": 1, "i2": 4}),
        trend_item_features=None,
        user_tag_affinities={"tag_a": 1.0},
        tag_uplifts={"tag_a": 0.5},
        max_positive_uplift=0.5,
        emerging_scores={"i1": 0.2, "i2": 0.1},
        max_emerging_score=0.2,
        modality_weights={"visual": 0.4, "text": 0.4, "audio": 0.2},
        cold_item_max_interactions=10,
    )
    training_features = np.asarray(
        [
            build_scalar_ranker_feature_vector(
                CandidateScore(item_id="i1", base_score=0.9),
                items["i1"],
                scalers=FeatureScalers(),
                **kwargs,
            ),
            build_scalar_ranker_feature_vector(
                CandidateScore(item_id="i1", base_score=0.7),
                items["i1"],
                scalers=FeatureScalers(),
                **kwargs,
            ),
            build_scalar_ranker_feature_vector(
                CandidateScore(item_id="i2", base_score=0.3),
                items["i2"],
                scalers=FeatureScalers(),
                **kwargs,
            ),
            build_scalar_ranker_feature_vector(
                CandidateScore(item_id="i2", base_score=0.1),
                items["i2"],
                scalers=FeatureScalers(),
                **kwargs,
            ),
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    classifier = train_hist_gradient_boosting_from_examples(
        training_features,
        labels,
        sample_weight=np.ones_like(labels, dtype=np.float32),
        max_iter=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        seed=7,
    )
    ranker = PointwiseHistGradientBoostingRanker(
        model=classifier,
        scalers=FeatureScalers(),
        num_examples=4,
        num_positive_examples=2,
        num_negative_examples=2,
        num_cases=2,
        num_cold_cases=1,
        num_warm_cases=1,
        cold_case_weight=1.0,
    )

    batched = dict(score_candidates_with_trained_ranker(ranker, candidates, items, **kwargs))
    single = {
        candidate.item_id: score_candidate_with_pointwise_hist_gradient_boosting_ranker(
            ranker,
            candidate,
            items[candidate.item_id],
            **kwargs,
        )
        for candidate in candidates
    }

    assert set(batched) == set(single)
    for item_id in single:
        assert batched[item_id] == single[item_id]
