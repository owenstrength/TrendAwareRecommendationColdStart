"""Synthetic demo dataset for quick prototyping and CI."""

from __future__ import annotations

import random

import numpy as np

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures

_RNG_SEED = 42
_DIM = 16


def _unit_vec(rng: random.Random, dim: int) -> np.ndarray:
    arr = np.array([rng.gauss(0, 1) for _ in range(dim)])
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        arr[0] = 1.0
        norm = 1.0
    return arr / norm


def load_demo_dataset(
    seed: int = _RNG_SEED,
) -> tuple[list[Interaction], dict[str, ItemFeatures], dict[str, list[CandidateScore]]]:
    """Return a small synthetic dataset suitable for smoke-testing the pipeline.

    Returns
    -------
    interactions:
        Timestamped user–item interaction log.
    items:
        Mapping from item_id to multimodal ``ItemFeatures``.
    user_candidates:
        Per-user candidate lists with base scores.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Items: 5 "warm" trending items + 10 cold items split into
    # "trend-close" and "trend-far" clusters.
    # ------------------------------------------------------------------
    trend_direction = _unit_vec(rng, _DIM)

    items: dict[str, ItemFeatures] = {}

    # Warm / trending items (interaction_count > 10)
    trending_ids: list[str] = []
    for i in range(5):
        noise = np.array([rng.gauss(0, 0.05) for _ in range(_DIM)])
        vec = trend_direction + noise
        vec /= float(np.linalg.norm(vec))
        item_id = f"warm_{i}"
        items[item_id] = ItemFeatures(
            item_id=item_id,
            visual=vec.copy(),
            text=vec.copy(),
            audio=vec.copy(),
            interaction_count=rng.randint(20, 100),
        )
        trending_ids.append(item_id)

    # Cold items close to the trend cluster (interaction_count <= 10)
    cold_close_ids: list[str] = []
    for i in range(5):
        noise = np.array([rng.gauss(0, 0.1) for _ in range(_DIM)])
        vec = trend_direction + noise
        vec /= float(np.linalg.norm(vec))
        item_id = f"cold_close_{i}"
        items[item_id] = ItemFeatures(
            item_id=item_id,
            visual=vec.copy(),
            text=vec.copy(),
            audio=vec.copy(),
            interaction_count=rng.randint(0, 5),
        )
        cold_close_ids.append(item_id)

    # Cold items far from the trend cluster
    cold_far_ids: list[str] = []
    for i in range(5):
        vec = _unit_vec(rng, _DIM)
        # Orthogonalise away from trend direction
        vec = vec - float(np.dot(vec, trend_direction)) * trend_direction
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        item_id = f"cold_far_{i}"
        items[item_id] = ItemFeatures(
            item_id=item_id,
            visual=vec.copy(),
            text=vec.copy(),
            audio=vec.copy(),
            interaction_count=rng.randint(0, 5),
        )
        cold_far_ids.append(item_id)

    all_item_ids = list(items.keys())

    # ------------------------------------------------------------------
    # Interactions: 3 users with different virality affinities.
    # ------------------------------------------------------------------
    base_time = 1_000_000
    interactions: list[Interaction] = []

    # user_trend: heavy consumer of trending content
    for _ in range(20):
        item_id = rng.choice(trending_ids)
        interactions.append(
            Interaction("user_trend", item_id, base_time + rng.randint(0, 3600 * 72))
        )

    # user_niche: prefers non-trending (cold-far) content
    for _ in range(20):
        item_id = rng.choice(cold_far_ids)
        interactions.append(
            Interaction("user_niche", item_id, base_time + rng.randint(0, 3600 * 72))
        )

    # user_mixed: balanced engagement
    for _ in range(10):
        item_id = rng.choice(trending_ids)
        interactions.append(
            Interaction("user_mixed", item_id, base_time + rng.randint(0, 3600 * 72))
        )
    for _ in range(10):
        item_id = rng.choice(cold_far_ids + cold_close_ids)
        interactions.append(
            Interaction("user_mixed", item_id, base_time + rng.randint(0, 3600 * 72))
        )

    # ------------------------------------------------------------------
    # Candidates: every user gets all non-warm items as candidates with
    # random base scores so the reranker has something to reorder.
    # ------------------------------------------------------------------
    candidate_pool = cold_close_ids + cold_far_ids
    user_candidates: dict[str, list[CandidateScore]] = {}
    for user_id in ("user_trend", "user_niche", "user_mixed"):
        shuffled = candidate_pool[:]
        rng.shuffle(shuffled)
        user_candidates[user_id] = [
            CandidateScore(item_id, round(rng.uniform(0.3, 0.7), 4))
            for item_id in shuffled
        ]

    return interactions, items, user_candidates


def get_trending_item_ids(items: dict[str, ItemFeatures]) -> list[str]:
    """Return item IDs that are warm (interaction_count > 10) for use as ground-truth trends."""
    return [item_id for item_id, feat in items.items() if feat.interaction_count > 10]


def get_cold_item_ids(items: dict[str, ItemFeatures], max_interactions: int = 10) -> list[str]:
    """Return item IDs that qualify as cold-start items."""
    return [
        item_id
        for item_id, feat in items.items()
        if feat.interaction_count <= max_interactions
    ]
