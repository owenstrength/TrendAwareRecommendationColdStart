import random

import numpy as np

from trend_aware_recs.config import DataConfig
from trend_aware_recs.data.microlens import load_microlens_dataset
from trend_aware_recs.data.schema import Interaction, ItemFeatures
from trend_aware_recs.evaluation.splits import build_time_based_holdout


def test_load_microlens_dataset_builds_cached_local_modalities(tmp_path) -> None:
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "user,item,timestamp\n"
        "u1,i1,100\n"
        "u2,i1,200\n"
        "u1,i2,300\n",
        encoding="utf-8",
    )
    titles = tmp_path / "titles.csv"
    titles.write_text("i1,Hero clip\ni2,Niche recipe\ni3,Unused metadata\n", encoding="utf-8")
    tags = tmp_path / "tags.csv"
    tags.write_text("i1,Anime\ni2,Delicacy\ni3,Daily Sharing\n", encoding="utf-8")
    comments = tmp_path / "comments.txt"
    comments.write_text(
        "u9\ti1\tgreat scene\n"
        "u8\ti1\tlove this ending\n"
        "u7\ti2\ttasty\n",
        encoding="utf-8",
    )
    engagement = tmp_path / "likes.txt"
    engagement.write_text("i1\t10\t100\ni2\t5\t50\ni3\t1\t10\n", encoding="utf-8")

    config = DataConfig(
        dataset_name="microlens_test",
        loader="microlens",
        interaction_path=str(pairs),
        derived_feature_dir=str(tmp_path / "derived"),
        title_path=str(titles),
        tags_path=str(tags),
        comments_path=str(comments),
        engagement_path=str(engagement),
        feature_dim=16,
        max_comments_per_item=2,
    )

    interactions, items = load_microlens_dataset(config)

    assert len(interactions) == 3
    assert set(items) == {"i1", "i2", "i3"}
    assert items["i1"].interaction_count == 2
    assert items["i3"].interaction_count == 0
    assert items["i1"].title == "Hero clip"
    assert items["i1"].tag == "Anime"
    assert items["i1"].comment_count == 2
    assert items["i2"].likes == 5
    assert items["i1"].visual.ndim == 1
    assert items["i1"].text.ndim == 1
    assert items["i1"].audio.ndim == 1
    assert (tmp_path / "derived" / "microlens_visual_tfidf_svd.npy").exists()
    assert (tmp_path / "derived" / "microlens_text_tfidf_svd.npy").exists()
    assert (tmp_path / "derived" / "microlens_char_tfidf_svd.npy").exists()


def test_load_microlens_dataset_uses_official_feature_arrays_when_available(tmp_path) -> None:
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "user,item,timestamp\n"
        "u1,1,100\n"
        "u2,2,200\n",
        encoding="utf-8",
    )
    titles = tmp_path / "titles.csv"
    titles.write_text("1,Hero clip\n2,Niche recipe\n", encoding="utf-8")
    tags = tmp_path / "tags.csv"
    tags.write_text("1,Anime\n2,Delicacy\n", encoding="utf-8")
    comments = tmp_path / "comments.txt"
    comments.write_text("u9\t1\tgreat scene\n", encoding="utf-8")
    engagement = tmp_path / "likes.txt"
    engagement.write_text("1\t10\t100\n2\t5\t50\n", encoding="utf-8")

    visual = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
    audio = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
    visual_path = tmp_path / "visual.npy"
    text_path = tmp_path / "text.npy"
    audio_path = tmp_path / "audio.npy"
    np.save(visual_path, visual)
    np.save(text_path, text)
    np.save(audio_path, audio)

    config = DataConfig(
        dataset_name="microlens_test",
        loader="microlens",
        interaction_path=str(pairs),
        title_path=str(titles),
        tags_path=str(tags),
        comments_path=str(comments),
        engagement_path=str(engagement),
        item_feature_paths={
            "visual": str(visual_path),
            "text": str(text_path),
            "audio": str(audio_path),
        },
        feature_dim=2,
    )

    _interactions, items = load_microlens_dataset(config)

    np.testing.assert_allclose(items["1"].visual, visual[0])
    np.testing.assert_allclose(items["2"].text, text[1] / np.linalg.norm(text[1]))


def test_build_time_based_holdout_creates_eval_cases() -> None:
    zeros = np.zeros(4)
    items = {
        "i1": ItemFeatures("i1", zeros.copy(), zeros.copy(), zeros.copy(), 3),
        "i2": ItemFeatures("i2", zeros.copy(), zeros.copy(), zeros.copy(), 2),
        "i3": ItemFeatures("i3", zeros.copy(), zeros.copy(), zeros.copy(), 1),
        "i4": ItemFeatures("i4", zeros.copy(), zeros.copy(), zeros.copy(), 0),
        "i5": ItemFeatures("i5", zeros.copy(), zeros.copy(), zeros.copy(), 0),
        "i6": ItemFeatures("i6", zeros.copy(), zeros.copy(), zeros.copy(), 1),
        "i7": ItemFeatures("i7", zeros.copy(), zeros.copy(), zeros.copy(), 1),
    }
    interactions = [
        # u1 train
        *[
            Interaction("u1", item_id, ts)
            for item_id, ts in [("i1", 10), ("i2", 20), ("i3", 30)]
        ],
        # u2 train
        *[
            Interaction("u2", item_id, ts)
            for item_id, ts in [("i1", 11), ("i2", 21), ("i3", 31)]
        ],
        Interaction("u3", "i6", 12),
        Interaction("u3", "i7", 13),
        # post-cutoff positives
        Interaction("u1", "i4", 100),
        Interaction("u2", "i5", 101),
    ]

    holdout = build_time_based_holdout(
        interactions=interactions,
        items=items,
        seed=7,
        evaluation_fraction=0.25,
        negatives_per_user=2,
        min_user_train_interactions=3,
        max_eval_users=None,
        cold_item_max_interactions=1,
        modality_weights={"visual": 1.0, "text": 0.0, "audio": 0.0},
        base_history_weight=0.7,
        base_popularity_weight=0.3,
    )

    assert len(holdout.cases) == 2
    for case in holdout.cases:
        assert case.positive_item_id in {candidate.item_id for candidate in case.candidates}
        assert case.positive_is_cold is True
        assert len(case.history_interactions) >= 3


def test_build_time_based_holdout_samples_negatives_deterministically() -> None:
    zeros = np.zeros(4)
    items = {
        item_id: ItemFeatures(item_id, zeros.copy(), zeros.copy(), zeros.copy(), 1)
        for item_id in ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"]
    }
    interactions = [
        Interaction("u1", "i1", 10),
        Interaction("u2", "i4", 11),
        Interaction("u3", "i5", 12),
        Interaction("u4", "i6", 13),
        Interaction("u5", "i7", 14),
        Interaction("u1", "i2", 20),
        Interaction("u1", "i3", 30),
        Interaction("u1", "i8", 100),
    ]

    holdout = build_time_based_holdout(
        interactions=interactions,
        items=items,
        seed=7,
        evaluation_fraction=0.25,
        negatives_per_user=2,
        min_user_train_interactions=3,
        max_eval_users=None,
        cold_item_max_interactions=1,
        modality_weights={"visual": 1.0, "text": 0.0, "audio": 0.0},
        base_history_weight=0.7,
        base_popularity_weight=0.3,
    )

    assert len(holdout.cases) == 1
    case = holdout.cases[0]
    expected_negatives = random.Random(7).sample(["i4", "i5", "i6", "i7"], k=2)
    assert [candidate.item_id for candidate in case.candidates] == [case.positive_item_id, *expected_negatives]


def test_build_time_based_holdout_can_use_exhaustive_negatives() -> None:
    zeros = np.zeros(4)
    items = {
        item_id: ItemFeatures(item_id, zeros.copy(), zeros.copy(), zeros.copy(), 1)
        for item_id in ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"]
    }
    interactions = [
        Interaction("u1", "i1", 10),
        Interaction("u2", "i4", 11),
        Interaction("u3", "i5", 12),
        Interaction("u4", "i6", 13),
        Interaction("u5", "i7", 14),
        Interaction("u1", "i2", 20),
        Interaction("u1", "i3", 30),
        Interaction("u1", "i8", 100),
    ]

    holdout = build_time_based_holdout(
        interactions=interactions,
        items=items,
        seed=7,
        evaluation_fraction=0.25,
        negatives_per_user=0,
        min_user_train_interactions=3,
        max_eval_users=None,
        cold_item_max_interactions=1,
        modality_weights={"visual": 1.0, "text": 0.0, "audio": 0.0},
        base_history_weight=0.7,
        base_popularity_weight=0.3,
    )

    assert len(holdout.cases) == 1
    case = holdout.cases[0]
    assert [candidate.item_id for candidate in case.candidates] == ["i8", "i4", "i5", "i6", "i7"]
