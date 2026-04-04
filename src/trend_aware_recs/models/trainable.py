from __future__ import annotations

import heapq
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from trend_aware_recs.data.schema import CandidateScore, Interaction, ItemFeatures
from trend_aware_recs.models.baseline import build_base_candidates
from trend_aware_recs.models.similarity import (
    average_item_features,
    cosine_similarity,
    multimodal_similarity,
    weighted_average_item_features,
)
from trend_aware_recs.models.trend import (
    score_emerging_item_ids,
    select_top_scored_item_ids,
    select_trending_item_ids,
)


@dataclass(frozen=True)
class FeatureScalers:
    """Placeholder for train-time feature scaling.

    Snapshot engagement metadata such as likes/views/comment totals is intentionally
    excluded from trainable ranker features because it is not time-safe at
    recommendation time in this project setup.
    """


_NUM_SCALAR_FEATURES = 14


@dataclass(frozen=True)
class LinearPairwiseRanker:
    weights: np.ndarray
    scalers: FeatureScalers
    num_pairs: int
    num_cases: int
    num_cold_cases: int
    num_warm_cases: int
    cold_case_weight: float


@dataclass(frozen=True)
class PointwiseHistGradientBoostingRanker:
    model: HistGradientBoostingClassifier
    scalers: FeatureScalers
    num_examples: int
    num_positive_examples: int
    num_negative_examples: int
    num_cases: int
    num_cold_cases: int
    num_warm_cases: int
    cold_case_weight: float


@dataclass
class _StreamingUserState:
    seen_items: set[str]
    recent_item_ids: deque[str]
    visual_sum: np.ndarray
    text_sum: np.ndarray
    audio_sum: np.ndarray
    interaction_count: int
    tag_counts: Counter[str]


def _make_user_state(
    visual_dim: int,
    text_dim: int,
    audio_dim: int,
) -> _StreamingUserState:
    return _StreamingUserState(
        seen_items=set(),
        recent_item_ids=deque(maxlen=3),
        visual_sum=np.zeros(visual_dim, dtype=np.float32),
        text_sum=np.zeros(text_dim, dtype=np.float32),
        audio_sum=np.zeros(audio_dim, dtype=np.float32),
        interaction_count=0,
        tag_counts=Counter(),
    )


def _safe_log_norm(value: int, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(math.log1p(max(0, value)) / denominator)


def _mean_unit_vector(vector_sum: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros_like(vector_sum)
    vector = vector_sum / float(count)
    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return np.zeros_like(vector_sum)
    return vector / norm


def _history_profile_from_state(state: _StreamingUserState) -> ItemFeatures | None:
    if state.interaction_count <= 0:
        return None
    return ItemFeatures(
        item_id="user_profile",
        visual=_mean_unit_vector(state.visual_sum, state.interaction_count),
        text=_mean_unit_vector(state.text_sum, state.interaction_count),
        audio=_mean_unit_vector(state.audio_sum, state.interaction_count),
        interaction_count=state.interaction_count,
    )


def _trend_similarity(
    item: ItemFeatures,
    trend_item_features: list[ItemFeatures] | None,
    modality_weights: dict[str, float],
) -> float:
    if not trend_item_features:
        return 0.0
    if len(trend_item_features) == 1:
        return max(0.0, multimodal_similarity(item, trend_item_features, modality_weights))
    return max(
        0.0,
        max(
            multimodal_similarity(item, [trend_item], modality_weights)
            for trend_item in trend_item_features
        ),
    )


def build_pairwise_ranker_feature_vector(
    candidate: CandidateScore,
    item: ItemFeatures,
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    scalers: FeatureScalers,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> np.ndarray:
    visual_dim = item.visual.shape[0]
    text_dim = item.text.shape[0]
    audio_dim = item.audio.shape[0]

    if history_profile is None:
        history_visual = np.zeros(visual_dim, dtype=np.float32)
        history_text = np.zeros(text_dim, dtype=np.float32)
        history_audio = np.zeros(audio_dim, dtype=np.float32)
        history_visual_similarity = 0.0
        history_text_similarity = 0.0
        history_audio_similarity = 0.0
    else:
        history_visual = history_profile.visual * item.visual
        history_text = history_profile.text * item.text
        history_audio = history_profile.audio * item.audio
        history_visual_similarity = max(0.0, cosine_similarity(history_profile.visual, item.visual))
        history_text_similarity = max(0.0, cosine_similarity(history_profile.text, item.text))
        history_audio_similarity = max(0.0, cosine_similarity(history_profile.audio, item.audio))

    history_similarity = max(
        0.0,
        modality_weights.get("visual", 0.0) * history_visual_similarity
        + modality_weights.get("text", 0.0) * history_text_similarity
        + modality_weights.get("audio", 0.0) * history_audio_similarity,
    )
    last_item_similarity = 0.0
    if recent_history_items:
        last_item_similarity = max(0.0, multimodal_similarity(item, [recent_history_items[-1]], modality_weights))
    recent_history_similarity = 0.0
    if recent_history_items:
        recent_profile = average_item_features("recent_user_profile", recent_history_items)
        recent_history_similarity = max(0.0, multimodal_similarity(item, [recent_profile], modality_weights))

    prefix_count = prefix_item_counts.get(item.item_id, 0)
    max_prefix_count = max(prefix_item_counts.values(), default=0)
    popularity_norm = _safe_log_norm(prefix_count, math.log1p(max_prefix_count) if max_prefix_count > 0 else 0.0)

    recent_count = recent_item_counts.get(item.item_id, 0)
    max_recent_count = max(recent_item_counts.values(), default=0)
    recent_norm = float(recent_count / max_recent_count) if max_recent_count > 0 else 0.0

    trend_similarity = _trend_similarity(item, trend_item_features, modality_weights)
    tag_affinity = user_tag_affinities.get(item.tag, 0.0) if item.tag else 0.0
    tag_uplift_norm = 0.0
    if item.tag and max_positive_uplift > 0.0:
        tag_uplift_norm = max(0.0, tag_uplifts.get(item.tag, 0.0)) / max_positive_uplift

    emerging_score_norm = (
        emerging_scores.get(item.item_id, 0.0) / max_emerging_score if max_emerging_score > 0.0 else 0.0
    )
    cold_indicator = 1.0 if prefix_count <= cold_item_max_interactions else 0.0

    scalars = np.asarray(
        [
            candidate.base_score,
            popularity_norm,
            recent_norm,
            cold_indicator,
            history_visual_similarity,
            history_text_similarity,
            history_audio_similarity,
            history_similarity,
            last_item_similarity,
            recent_history_similarity,
            trend_similarity,
            tag_affinity,
            tag_uplift_norm,
            emerging_score_norm,
        ],
        dtype=np.float32,
    )
    return np.concatenate([history_visual, history_text, history_audio, scalars]).astype(np.float32, copy=False)


def build_scalar_ranker_feature_vector(
    candidate: CandidateScore,
    item: ItemFeatures,
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    scalers: FeatureScalers,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> np.ndarray:
    full_vector = build_pairwise_ranker_feature_vector(
        candidate,
        item,
        history_profile=history_profile,
        prefix_item_counts=prefix_item_counts,
        recent_item_counts=recent_item_counts,
        trend_item_features=trend_item_features,
        user_tag_affinities=user_tag_affinities,
        tag_uplifts=tag_uplifts,
        max_positive_uplift=max_positive_uplift,
        emerging_scores=emerging_scores,
        max_emerging_score=max_emerging_score,
        modality_weights=modality_weights,
        cold_item_max_interactions=cold_item_max_interactions,
        scalers=scalers,
        history_items=history_items,
        recent_history_items=recent_history_items,
    )
    return full_vector[-_NUM_SCALAR_FEATURES:]


def _negative_priority_score(
    item_id: str,
    *,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    emerging_scores: dict[str, float],
) -> float:
    return (
        math.log1p(prefix_item_counts.get(item_id, 0))
        + math.log1p(recent_item_counts.get(item_id, 0))
        + (2.0 * emerging_scores.get(item_id, 0.0))
    )


def _select_training_negative_ids(
    negative_pool: list[str],
    *,
    negatives_per_case: int,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    emerging_scores: dict[str, float],
    hard_negative_ratio: float,
    hard_negative_pool_size: int,
    rng: random.Random,
) -> list[str]:
    if negatives_per_case <= 0 or not negative_pool:
        return []

    sample_size = min(negatives_per_case, len(negative_pool))
    hard_ratio = min(max(hard_negative_ratio, 0.0), 1.0)
    hard_target = min(sample_size, int(round(sample_size * hard_ratio)))
    if hard_target <= 0:
        return rng.sample(negative_pool, k=sample_size)

    shortlist_size = min(
        len(negative_pool),
        max(hard_target, hard_negative_pool_size, hard_target * 3),
    )
    shortlist = heapq.nlargest(
        shortlist_size,
        negative_pool,
        key=lambda item_id: (
            _negative_priority_score(
                item_id,
                prefix_item_counts=prefix_item_counts,
                recent_item_counts=recent_item_counts,
                emerging_scores=emerging_scores,
            ),
            item_id,
        ),
    )
    hard_negatives = rng.sample(shortlist, k=hard_target) if len(shortlist) > hard_target else shortlist

    selected = list(dict.fromkeys(hard_negatives))
    if len(selected) >= sample_size:
        return selected[:sample_size]

    selected_set = set(selected)
    remaining_pool = [item_id for item_id in negative_pool if item_id not in selected_set]
    remaining_target = min(sample_size - len(selected), len(remaining_pool))
    if remaining_target > 0:
        selected.extend(rng.sample(remaining_pool, k=remaining_target))
    return selected


def build_scalar_ranker_feature_matrix(
    candidates: list[CandidateScore],
    items: dict[str, ItemFeatures],
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    scalers: FeatureScalers,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> tuple[list[str], np.ndarray]:
    item_ids: list[str] = []
    feature_rows: list[np.ndarray] = []
    for candidate in candidates:
        item = items.get(candidate.item_id)
        if item is None:
            continue
        item_ids.append(candidate.item_id)
        feature_rows.append(
            build_scalar_ranker_feature_vector(
                candidate,
                item,
                history_profile=history_profile,
                prefix_item_counts=prefix_item_counts,
                recent_item_counts=recent_item_counts,
                trend_item_features=trend_item_features,
                user_tag_affinities=user_tag_affinities,
                tag_uplifts=tag_uplifts,
                max_positive_uplift=max_positive_uplift,
                emerging_scores=emerging_scores,
                max_emerging_score=max_emerging_score,
                modality_weights=modality_weights,
                cold_item_max_interactions=cold_item_max_interactions,
                scalers=scalers,
                history_items=history_items,
                recent_history_items=recent_history_items,
            )
        )
    if not feature_rows:
        return item_ids, np.empty((0, _NUM_SCALAR_FEATURES), dtype=np.float32)
    return item_ids, np.asarray(feature_rows, dtype=np.float32)


def train_linear_ranker_from_pair_diffs(
    pair_diffs: np.ndarray,
    *,
    pair_weights: np.ndarray | None = None,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> np.ndarray:
    if pair_diffs.size == 0:
        raise ValueError("pair_diffs must not be empty")

    if pair_weights is None:
        pair_weights = np.ones(pair_diffs.shape[0], dtype=np.float32)
    elif pair_weights.shape[0] != pair_diffs.shape[0]:
        raise ValueError("pair_weights must align with pair_diffs")

    normalized_weights = pair_weights.astype(np.float64, copy=False)
    mean_weight = float(normalized_weights.mean())
    if mean_weight <= 0.0:
        raise ValueError("pair_weights must have positive mean")
    normalized_weights = normalized_weights / mean_weight

    rng = np.random.default_rng(seed)
    weights = np.zeros(pair_diffs.shape[1], dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0

    for _ in range(max(1, epochs)):
        order = rng.permutation(pair_diffs.shape[0])
        shuffled = pair_diffs[order]
        shuffled_weights = normalized_weights[order]
        logits = shuffled @ weights
        # Pairwise logistic loss on x_pos - x_neg.
        margins = 1.0 / (1.0 + np.exp(np.clip(logits, -30.0, 30.0)))
        grad = -((margins * shuffled_weights)[:, None] * shuffled).sum(axis=0) / shuffled_weights.sum()
        grad += l2 * weights
        step += 1
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + eps)

    return weights.astype(np.float32)


def train_hist_gradient_boosting_from_examples(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    max_iter: int,
    learning_rate: float,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    if features.size == 0:
        raise ValueError("features must not be empty")
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain at least two classes")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def _feature_scalers(items: dict[str, ItemFeatures]) -> FeatureScalers:
    return FeatureScalers()


def _select_training_cases(
    eligible_cases: list[tuple[int, bool]],
    *,
    max_cases: int,
    cold_case_weight: float,
    seed: int,
) -> dict[int, bool]:
    if len(eligible_cases) <= max_cases:
        return {index: is_cold for index, is_cold in eligible_cases}

    rng = random.Random(seed)
    cold_indices = [index for index, is_cold in eligible_cases if is_cold]
    warm_indices = [index for index, is_cold in eligible_cases if not is_cold]
    if not cold_indices or not warm_indices:
        selected = rng.sample([index for index, _ in eligible_cases], k=max_cases)
        cold_index_set = set(cold_indices)
        return {index: index in cold_index_set for index in selected}

    cold_fraction = len(cold_indices) / len(eligible_cases)
    effective_cold_weight = max(0.0, cold_case_weight)
    weighted_cold_mass = cold_fraction * effective_cold_weight
    weighted_total_mass = weighted_cold_mass + (1.0 - cold_fraction)
    target_cold_fraction = weighted_cold_mass / weighted_total_mass if weighted_total_mass > 0.0 else 0.0

    target_cold_count = min(len(cold_indices), int(round(max_cases * target_cold_fraction)))
    target_warm_count = min(len(warm_indices), max_cases - target_cold_count)

    while target_cold_count + target_warm_count < max_cases:
        cold_remaining = len(cold_indices) - target_cold_count
        warm_remaining = len(warm_indices) - target_warm_count
        if cold_remaining <= 0 and warm_remaining <= 0:
            break
        if cold_remaining >= warm_remaining and cold_remaining > 0:
            target_cold_count += 1
        elif warm_remaining > 0:
            target_warm_count += 1
        else:
            target_cold_count += 1

    selected_cold = rng.sample(cold_indices, k=target_cold_count) if target_cold_count > 0 else []
    selected_warm = rng.sample(warm_indices, k=target_warm_count) if target_warm_count > 0 else []
    return {
        **{index: True for index in selected_cold},
        **{index: False for index in selected_warm},
    }


def train_pairwise_linear_ranker(
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    *,
    seed: int,
    max_cases: int,
    negatives_per_case: int,
    min_user_train_interactions: int,
    trend_window_hours: int,
    top_trend_fraction: float,
    emerging_fraction: float,
    min_trend_items: int,
    emerging_min_recent_interactions: int,
    emerging_smoothing: float,
    cold_item_max_interactions: int,
    modality_weights: dict[str, float],
    base_history_weight: float,
    base_popularity_weight: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    cold_case_weight: float,
    hard_negative_ratio: float,
    hard_negative_pool_size: int,
) -> LinearPairwiseRanker | None:
    ordered = sorted(interactions, key=lambda interaction: interaction.timestamp)
    if not ordered:
        return None

    eligible_cases: list[tuple[int, bool]] = []
    history_counts: Counter[str] = Counter()
    seen_items: dict[str, set[str]] = defaultdict(set)
    item_counts: Counter[str] = Counter()
    for index, interaction in enumerate(ordered):
        if (
            interaction.item_id in items
            and history_counts[interaction.user_id] >= min_user_train_interactions
            and interaction.item_id not in seen_items[interaction.user_id]
        ):
            eligible_cases.append(
                (
                    index,
                    item_counts.get(interaction.item_id, 0) <= cold_item_max_interactions,
                )
            )
        history_counts[interaction.user_id] += 1
        seen_items[interaction.user_id].add(interaction.item_id)
        item_counts[interaction.item_id] += 1

    if not eligible_cases:
        return None

    selected_cases = _select_training_cases(
        eligible_cases,
        max_cases=max_cases,
        cold_case_weight=cold_case_weight,
        seed=seed,
    )
    rng = random.Random(seed)

    example_item = next(iter(items.values()))
    visual_dim = example_item.visual.shape[0]
    text_dim = example_item.text.shape[0]
    audio_dim = example_item.audio.shape[0]
    user_states: dict[str, _StreamingUserState] = defaultdict(
        lambda: _make_user_state(visual_dim, text_dim, audio_dim)
    )
    prefix_item_counts: Counter[str] = Counter()
    prefix_tag_counts: Counter[str] = Counter()
    recent_item_counts: Counter[str] = Counter()
    recent_tag_counts: Counter[str] = Counter()
    recent_window: deque[Interaction] = deque()
    window_ms = trend_window_hours * 3600 * 1000
    scalers = _feature_scalers(items)
    pair_diffs: list[np.ndarray] = []
    pair_weights: list[float] = []
    sampled_case_count = 0
    sampled_cold_case_count = 0

    for index, interaction in enumerate(ordered):
        while recent_window and interaction.timestamp - recent_window[0].timestamp > window_ms:
            expired = recent_window.popleft()
            recent_item_counts[expired.item_id] -= 1
            if recent_item_counts[expired.item_id] <= 0:
                del recent_item_counts[expired.item_id]
            expired_item = items.get(expired.item_id)
            if expired_item is not None and expired_item.tag:
                recent_tag_counts[expired_item.tag] -= 1
                if recent_tag_counts[expired_item.tag] <= 0:
                    del recent_tag_counts[expired_item.tag]

        state = user_states[interaction.user_id]
        is_selected_cold = selected_cases.get(index)
        if is_selected_cold is not None and interaction.item_id in items and interaction.item_id not in state.seen_items:
            negative_pool = [
                item_id
                for item_id in prefix_item_counts
                if item_id != interaction.item_id and item_id not in state.seen_items
            ]
            if negative_pool:
                emerging_scores = score_emerging_item_ids(
                    recent_item_counts,
                    prefix_item_counts,
                    smoothing=emerging_smoothing,
                    min_recent_interactions=emerging_min_recent_interactions,
                )
                negatives = _select_training_negative_ids(
                    negative_pool,
                    negatives_per_case=negatives_per_case,
                    prefix_item_counts=prefix_item_counts,
                    recent_item_counts=recent_item_counts,
                    emerging_scores=emerging_scores,
                    hard_negative_ratio=hard_negative_ratio,
                    hard_negative_pool_size=hard_negative_pool_size,
                    rng=rng,
                )
                history_profile = _history_profile_from_state(state)
                history_items = []
                if history_profile is not None:
                    history_items = [
                        ItemFeatures(
                            item_id="user_profile",
                            visual=history_profile.visual,
                            text=history_profile.text,
                            audio=history_profile.audio,
                            interaction_count=history_profile.interaction_count,
                        )
                    ]
                candidates = build_base_candidates(
                    candidate_item_ids=[interaction.item_id, *negatives],
                    history_items=history_items,
                    items=items,
                    popularity_counts=dict(prefix_item_counts),
                    modality_weights=modality_weights,
                    history_weight=base_history_weight,
                    popularity_weight=base_popularity_weight,
                )
                candidate_map = {candidate.item_id: candidate for candidate in candidates}

                latest_trending_items = select_trending_item_ids(
                    recent_item_counts,
                    top_fraction=top_trend_fraction,
                    min_items=min_trend_items,
                )
                latest_emerging_items = select_top_scored_item_ids(
                    emerging_scores,
                    top_fraction=emerging_fraction,
                    min_items=min_trend_items,
                )
                trend_item_features = []
                trend_source_items = [items[item_id] for item_id in latest_trending_items if item_id in items]
                trend_weights = [recent_item_counts[item_id] for item_id in latest_trending_items if item_id in items]
                if trend_source_items:
                    trend_item_features.append(
                        weighted_average_item_features("trend_centroid", trend_source_items, trend_weights)
                    )
                emerging_source_items = [items[item_id] for item_id in latest_emerging_items if item_id in items]
                emerging_weights = [emerging_scores[item_id] for item_id in latest_emerging_items if item_id in items]
                if emerging_source_items:
                    trend_item_features.append(
                        weighted_average_item_features(
                            "emerging_centroid",
                            emerging_source_items,
                            emerging_weights,
                        )
                    )
                trend_item_features = trend_item_features or None
                total_tag_events = sum(state.tag_counts.values())
                user_tag_affinities = (
                    {tag: count / total_tag_events for tag, count in state.tag_counts.items()}
                    if total_tag_events > 0
                    else {}
                )
                total_overall = sum(prefix_tag_counts.values())
                total_recent = sum(recent_tag_counts.values())
                tag_uplifts: dict[str, float] = {}
                if total_overall > 0 and total_recent > 0:
                    for tag, recent_count in recent_tag_counts.items():
                        tag_uplifts[tag] = (recent_count / total_recent) - (
                            prefix_tag_counts[tag] / total_overall
                        )
                max_positive_uplift = max(
                    (score for score in tag_uplifts.values() if score > 0.0),
                    default=0.0,
                )
                max_emerging_score = max(emerging_scores.values(), default=0.0)
                recent_history_items = [items[item_id] for item_id in state.recent_item_ids if item_id in items]

                positive_vector = build_pairwise_ranker_feature_vector(
                    candidate_map[interaction.item_id],
                    items[interaction.item_id],
                    history_profile=history_profile,
                    prefix_item_counts=prefix_item_counts,
                    recent_item_counts=recent_item_counts,
                    trend_item_features=trend_item_features,
                    user_tag_affinities=user_tag_affinities,
                    tag_uplifts=tag_uplifts,
                    max_positive_uplift=max_positive_uplift,
                    emerging_scores=emerging_scores,
                    max_emerging_score=max_emerging_score,
                    modality_weights=modality_weights,
                    cold_item_max_interactions=cold_item_max_interactions,
                    scalers=scalers,
                    recent_history_items=recent_history_items,
                )
                for negative_id in negatives:
                    negative_vector = build_pairwise_ranker_feature_vector(
                        candidate_map[negative_id],
                        items[negative_id],
                        history_profile=history_profile,
                        prefix_item_counts=prefix_item_counts,
                        recent_item_counts=recent_item_counts,
                        trend_item_features=trend_item_features,
                        user_tag_affinities=user_tag_affinities,
                        tag_uplifts=tag_uplifts,
                        max_positive_uplift=max_positive_uplift,
                        emerging_scores=emerging_scores,
                        max_emerging_score=max_emerging_score,
                        modality_weights=modality_weights,
                        cold_item_max_interactions=cold_item_max_interactions,
                        scalers=scalers,
                        recent_history_items=recent_history_items,
                    )
                    pair_diffs.append((positive_vector - negative_vector).astype(np.float32))
                    pair_weights.append(cold_case_weight if is_selected_cold else 1.0)
                sampled_case_count += 1
                if is_selected_cold:
                    sampled_cold_case_count += 1

        item = items.get(interaction.item_id)
        if item is not None:
            prefix_item_counts[interaction.item_id] += 1
            recent_item_counts[interaction.item_id] += 1
            recent_window.append(interaction)
            state.visual_sum += item.visual
            state.text_sum += item.text
            state.audio_sum += item.audio
            state.interaction_count += 1
            state.seen_items.add(interaction.item_id)
            state.recent_item_ids.append(interaction.item_id)
            if item.tag:
                prefix_tag_counts[item.tag] += 1
                recent_tag_counts[item.tag] += 1
                state.tag_counts[item.tag] += 1

    if not pair_diffs:
        return None

    weights = train_linear_ranker_from_pair_diffs(
        np.asarray(pair_diffs, dtype=np.float32),
        pair_weights=np.asarray(pair_weights, dtype=np.float32),
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    return LinearPairwiseRanker(
        weights=weights,
        scalers=scalers,
        num_pairs=len(pair_diffs),
        num_cases=sampled_case_count,
        num_cold_cases=sampled_cold_case_count,
        num_warm_cases=sampled_case_count - sampled_cold_case_count,
        cold_case_weight=cold_case_weight,
    )


def train_pointwise_hist_gradient_boosting_ranker(
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    *,
    seed: int,
    max_cases: int,
    negatives_per_case: int,
    min_user_train_interactions: int,
    trend_window_hours: int,
    top_trend_fraction: float,
    emerging_fraction: float,
    min_trend_items: int,
    emerging_min_recent_interactions: int,
    emerging_smoothing: float,
    cold_item_max_interactions: int,
    modality_weights: dict[str, float],
    base_history_weight: float,
    base_popularity_weight: float,
    max_iter: int,
    learning_rate: float,
    max_depth: int | None,
    min_samples_leaf: int,
    cold_case_weight: float,
    hard_negative_ratio: float,
    hard_negative_pool_size: int,
) -> PointwiseHistGradientBoostingRanker | None:
    ordered = sorted(interactions, key=lambda interaction: interaction.timestamp)
    if not ordered:
        return None

    eligible_cases: list[tuple[int, bool]] = []
    history_counts: Counter[str] = Counter()
    seen_items: dict[str, set[str]] = defaultdict(set)
    item_counts: Counter[str] = Counter()
    for index, interaction in enumerate(ordered):
        if (
            interaction.item_id in items
            and history_counts[interaction.user_id] >= min_user_train_interactions
            and interaction.item_id not in seen_items[interaction.user_id]
        ):
            eligible_cases.append(
                (
                    index,
                    item_counts.get(interaction.item_id, 0) <= cold_item_max_interactions,
                )
            )
        history_counts[interaction.user_id] += 1
        seen_items[interaction.user_id].add(interaction.item_id)
        item_counts[interaction.item_id] += 1

    if not eligible_cases:
        return None

    selected_cases = _select_training_cases(
        eligible_cases,
        max_cases=max_cases,
        cold_case_weight=cold_case_weight,
        seed=seed,
    )
    rng = random.Random(seed)

    example_item = next(iter(items.values()))
    visual_dim = example_item.visual.shape[0]
    text_dim = example_item.text.shape[0]
    audio_dim = example_item.audio.shape[0]
    user_states: dict[str, _StreamingUserState] = defaultdict(
        lambda: _make_user_state(visual_dim, text_dim, audio_dim)
    )
    prefix_item_counts: Counter[str] = Counter()
    prefix_tag_counts: Counter[str] = Counter()
    recent_item_counts: Counter[str] = Counter()
    recent_tag_counts: Counter[str] = Counter()
    recent_window: deque[Interaction] = deque()
    window_ms = trend_window_hours * 3600 * 1000
    scalers = _feature_scalers(items)
    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    sampled_case_count = 0
    sampled_cold_case_count = 0

    for index, interaction in enumerate(ordered):
        while recent_window and interaction.timestamp - recent_window[0].timestamp > window_ms:
            expired = recent_window.popleft()
            recent_item_counts[expired.item_id] -= 1
            if recent_item_counts[expired.item_id] <= 0:
                del recent_item_counts[expired.item_id]
            expired_item = items.get(expired.item_id)
            if expired_item is not None and expired_item.tag:
                recent_tag_counts[expired_item.tag] -= 1
                if recent_tag_counts[expired_item.tag] <= 0:
                    del recent_tag_counts[expired_item.tag]

        state = user_states[interaction.user_id]
        is_selected_cold = selected_cases.get(index)
        if is_selected_cold is not None and interaction.item_id in items and interaction.item_id not in state.seen_items:
            negative_pool = [
                item_id
                for item_id in prefix_item_counts
                if item_id != interaction.item_id and item_id not in state.seen_items
            ]
            if negative_pool:
                emerging_scores = score_emerging_item_ids(
                    recent_item_counts,
                    prefix_item_counts,
                    smoothing=emerging_smoothing,
                    min_recent_interactions=emerging_min_recent_interactions,
                )
                negatives = _select_training_negative_ids(
                    negative_pool,
                    negatives_per_case=negatives_per_case,
                    prefix_item_counts=prefix_item_counts,
                    recent_item_counts=recent_item_counts,
                    emerging_scores=emerging_scores,
                    hard_negative_ratio=hard_negative_ratio,
                    hard_negative_pool_size=hard_negative_pool_size,
                    rng=rng,
                )
                history_profile = _history_profile_from_state(state)
                history_items = []
                if history_profile is not None:
                    history_items = [
                        ItemFeatures(
                            item_id="user_profile",
                            visual=history_profile.visual,
                            text=history_profile.text,
                            audio=history_profile.audio,
                            interaction_count=history_profile.interaction_count,
                        )
                    ]
                candidates = build_base_candidates(
                    candidate_item_ids=[interaction.item_id, *negatives],
                    history_items=history_items,
                    items=items,
                    popularity_counts=dict(prefix_item_counts),
                    modality_weights=modality_weights,
                    history_weight=base_history_weight,
                    popularity_weight=base_popularity_weight,
                )
                candidate_map = {candidate.item_id: candidate for candidate in candidates}

                latest_trending_items = select_trending_item_ids(
                    recent_item_counts,
                    top_fraction=top_trend_fraction,
                    min_items=min_trend_items,
                )
                latest_emerging_items = select_top_scored_item_ids(
                    emerging_scores,
                    top_fraction=emerging_fraction,
                    min_items=min_trend_items,
                )
                trend_item_features = []
                trend_source_items = [items[item_id] for item_id in latest_trending_items if item_id in items]
                trend_weights = [recent_item_counts[item_id] for item_id in latest_trending_items if item_id in items]
                if trend_source_items:
                    trend_item_features.append(
                        weighted_average_item_features("trend_centroid", trend_source_items, trend_weights)
                    )
                emerging_source_items = [items[item_id] for item_id in latest_emerging_items if item_id in items]
                emerging_weights = [emerging_scores[item_id] for item_id in latest_emerging_items if item_id in items]
                if emerging_source_items:
                    trend_item_features.append(
                        weighted_average_item_features(
                            "emerging_centroid",
                            emerging_source_items,
                            emerging_weights,
                        )
                    )
                trend_item_features = trend_item_features or None
                total_tag_events = sum(state.tag_counts.values())
                user_tag_affinities = (
                    {tag: count / total_tag_events for tag, count in state.tag_counts.items()}
                    if total_tag_events > 0
                    else {}
                )
                total_overall = sum(prefix_tag_counts.values())
                total_recent = sum(recent_tag_counts.values())
                tag_uplifts: dict[str, float] = {}
                if total_overall > 0 and total_recent > 0:
                    for tag, recent_count in recent_tag_counts.items():
                        tag_uplifts[tag] = (recent_count / total_recent) - (
                            prefix_tag_counts[tag] / total_overall
                        )
                max_positive_uplift = max(
                    (score for score in tag_uplifts.values() if score > 0.0),
                    default=0.0,
                )
                max_emerging_score = max(emerging_scores.values(), default=0.0)
                case_weight = cold_case_weight if is_selected_cold else 1.0
                recent_history_items = [items[item_id] for item_id in state.recent_item_ids if item_id in items]

                positive_vector = build_scalar_ranker_feature_vector(
                    candidate_map[interaction.item_id],
                    items[interaction.item_id],
                    history_profile=history_profile,
                    prefix_item_counts=prefix_item_counts,
                    recent_item_counts=recent_item_counts,
                    trend_item_features=trend_item_features,
                    user_tag_affinities=user_tag_affinities,
                    tag_uplifts=tag_uplifts,
                    max_positive_uplift=max_positive_uplift,
                    emerging_scores=emerging_scores,
                    max_emerging_score=max_emerging_score,
                    modality_weights=modality_weights,
                    cold_item_max_interactions=cold_item_max_interactions,
                    scalers=scalers,
                    recent_history_items=recent_history_items,
                )
                features.append(positive_vector)
                labels.append(1)
                sample_weights.append(case_weight)
                for negative_id in negatives:
                    negative_vector = build_scalar_ranker_feature_vector(
                        candidate_map[negative_id],
                        items[negative_id],
                        history_profile=history_profile,
                        prefix_item_counts=prefix_item_counts,
                        recent_item_counts=recent_item_counts,
                        trend_item_features=trend_item_features,
                        user_tag_affinities=user_tag_affinities,
                        tag_uplifts=tag_uplifts,
                        max_positive_uplift=max_positive_uplift,
                        emerging_scores=emerging_scores,
                        max_emerging_score=max_emerging_score,
                        modality_weights=modality_weights,
                        cold_item_max_interactions=cold_item_max_interactions,
                        scalers=scalers,
                        recent_history_items=recent_history_items,
                    )
                    features.append(negative_vector)
                    labels.append(0)
                    sample_weights.append(case_weight)
                sampled_case_count += 1
                if is_selected_cold:
                    sampled_cold_case_count += 1

        item = items.get(interaction.item_id)
        if item is not None:
            prefix_item_counts[interaction.item_id] += 1
            recent_item_counts[interaction.item_id] += 1
            recent_window.append(interaction)
            state.visual_sum += item.visual
            state.text_sum += item.text
            state.audio_sum += item.audio
            state.interaction_count += 1
            state.seen_items.add(interaction.item_id)
            state.recent_item_ids.append(interaction.item_id)
            if item.tag:
                prefix_tag_counts[item.tag] += 1
                recent_tag_counts[item.tag] += 1
                state.tag_counts[item.tag] += 1

    if not features:
        return None

    matrix = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    weight_array = np.asarray(sample_weights, dtype=np.float32)
    model = train_hist_gradient_boosting_from_examples(
        matrix,
        label_array,
        sample_weight=weight_array,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    positive_count = int(label_array.sum())
    return PointwiseHistGradientBoostingRanker(
        model=model,
        scalers=scalers,
        num_examples=len(features),
        num_positive_examples=positive_count,
        num_negative_examples=len(features) - positive_count,
        num_cases=sampled_case_count,
        num_cold_cases=sampled_cold_case_count,
        num_warm_cases=sampled_case_count - sampled_cold_case_count,
        cold_case_weight=cold_case_weight,
    )


def score_candidate_with_linear_ranker(
    model: LinearPairwiseRanker,
    candidate: CandidateScore,
    item: ItemFeatures,
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> float:
    features = build_pairwise_ranker_feature_vector(
        candidate,
        item,
        history_profile=history_profile,
        prefix_item_counts=prefix_item_counts,
        recent_item_counts=recent_item_counts,
        trend_item_features=trend_item_features,
        user_tag_affinities=user_tag_affinities,
        tag_uplifts=tag_uplifts,
        max_positive_uplift=max_positive_uplift,
        emerging_scores=emerging_scores,
        max_emerging_score=max_emerging_score,
        modality_weights=modality_weights,
        cold_item_max_interactions=cold_item_max_interactions,
        scalers=model.scalers,
        history_items=history_items,
        recent_history_items=recent_history_items,
    )
    return float(np.dot(features, model.weights))


def score_candidate_with_pointwise_hist_gradient_boosting_ranker(
    model: PointwiseHistGradientBoostingRanker,
    candidate: CandidateScore,
    item: ItemFeatures,
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> float:
    features = build_scalar_ranker_feature_vector(
        candidate,
        item,
        history_profile=history_profile,
        prefix_item_counts=prefix_item_counts,
        recent_item_counts=recent_item_counts,
        trend_item_features=trend_item_features,
        user_tag_affinities=user_tag_affinities,
        tag_uplifts=tag_uplifts,
        max_positive_uplift=max_positive_uplift,
        emerging_scores=emerging_scores,
        max_emerging_score=max_emerging_score,
        modality_weights=modality_weights,
        cold_item_max_interactions=cold_item_max_interactions,
        scalers=model.scalers,
        history_items=history_items,
        recent_history_items=recent_history_items,
    )
    return float(model.model.predict_proba(features.reshape(1, -1))[0, 1])


def score_candidates_with_trained_ranker(
    model: LinearPairwiseRanker | PointwiseHistGradientBoostingRanker,
    candidates: list[CandidateScore],
    items: dict[str, ItemFeatures],
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> list[tuple[str, float]]:
    if isinstance(model, PointwiseHistGradientBoostingRanker):
        item_ids, feature_matrix = build_scalar_ranker_feature_matrix(
            candidates,
            items,
            history_profile=history_profile,
            prefix_item_counts=prefix_item_counts,
            recent_item_counts=recent_item_counts,
            trend_item_features=trend_item_features,
            user_tag_affinities=user_tag_affinities,
            tag_uplifts=tag_uplifts,
            max_positive_uplift=max_positive_uplift,
            emerging_scores=emerging_scores,
            max_emerging_score=max_emerging_score,
            modality_weights=modality_weights,
            cold_item_max_interactions=cold_item_max_interactions,
            scalers=model.scalers,
            history_items=history_items,
            recent_history_items=recent_history_items,
        )
        if feature_matrix.size == 0:
            return []
        probabilities = model.model.predict_proba(feature_matrix)[:, 1]
        return list(zip(item_ids, probabilities.tolist(), strict=True))

    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        item = items.get(candidate.item_id)
        if item is None:
            continue
        scored.append(
            (
                candidate.item_id,
                score_candidate_with_linear_ranker(
                    model,
                    candidate,
                    item,
                    history_profile=history_profile,
                    prefix_item_counts=prefix_item_counts,
                    recent_item_counts=recent_item_counts,
                    trend_item_features=trend_item_features,
                    user_tag_affinities=user_tag_affinities,
                    tag_uplifts=tag_uplifts,
                    max_positive_uplift=max_positive_uplift,
                    emerging_scores=emerging_scores,
                    max_emerging_score=max_emerging_score,
                    modality_weights=modality_weights,
                    cold_item_max_interactions=cold_item_max_interactions,
                    history_items=history_items,
                    recent_history_items=recent_history_items,
                ),
            )
        )
    return scored


def train_trainable_ranker(
    interactions: list[Interaction],
    items: dict[str, ItemFeatures],
    *,
    model_type: str,
    seed: int,
    max_cases: int,
    negatives_per_case: int,
    min_user_train_interactions: int,
    trend_window_hours: int,
    top_trend_fraction: float,
    emerging_fraction: float,
    min_trend_items: int,
    emerging_min_recent_interactions: int,
    emerging_smoothing: float,
    cold_item_max_interactions: int,
    modality_weights: dict[str, float],
    base_history_weight: float,
    base_popularity_weight: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    cold_case_weight: float,
    hgb_max_iter: int,
    hgb_learning_rate: float,
    hgb_max_depth: int | None,
    hgb_min_samples_leaf: int,
    hard_negative_ratio: float,
    hard_negative_pool_size: int,
) -> LinearPairwiseRanker | PointwiseHistGradientBoostingRanker | None:
    if model_type == "pointwise_hgb":
        return train_pointwise_hist_gradient_boosting_ranker(
            interactions,
            items,
            seed=seed,
            max_cases=max_cases,
            negatives_per_case=negatives_per_case,
            min_user_train_interactions=min_user_train_interactions,
            trend_window_hours=trend_window_hours,
            top_trend_fraction=top_trend_fraction,
            emerging_fraction=emerging_fraction,
            min_trend_items=min_trend_items,
            emerging_min_recent_interactions=emerging_min_recent_interactions,
            emerging_smoothing=emerging_smoothing,
            cold_item_max_interactions=cold_item_max_interactions,
            modality_weights=modality_weights,
            base_history_weight=base_history_weight,
            base_popularity_weight=base_popularity_weight,
            max_iter=hgb_max_iter,
            learning_rate=hgb_learning_rate,
            max_depth=hgb_max_depth,
            min_samples_leaf=hgb_min_samples_leaf,
            cold_case_weight=cold_case_weight,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_pool_size=hard_negative_pool_size,
        )
    return train_pairwise_linear_ranker(
        interactions,
        items,
        seed=seed,
        max_cases=max_cases,
        negatives_per_case=negatives_per_case,
        min_user_train_interactions=min_user_train_interactions,
        trend_window_hours=trend_window_hours,
        top_trend_fraction=top_trend_fraction,
        emerging_fraction=emerging_fraction,
        min_trend_items=min_trend_items,
        emerging_min_recent_interactions=emerging_min_recent_interactions,
        emerging_smoothing=emerging_smoothing,
        cold_item_max_interactions=cold_item_max_interactions,
        modality_weights=modality_weights,
        base_history_weight=base_history_weight,
        base_popularity_weight=base_popularity_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        cold_case_weight=cold_case_weight,
        hard_negative_ratio=hard_negative_ratio,
        hard_negative_pool_size=hard_negative_pool_size,
    )


def score_candidate_with_trained_ranker(
    model: LinearPairwiseRanker | PointwiseHistGradientBoostingRanker,
    candidate: CandidateScore,
    item: ItemFeatures,
    *,
    history_profile: ItemFeatures | None,
    prefix_item_counts: Counter[str],
    recent_item_counts: Counter[str],
    trend_item_features: list[ItemFeatures] | None,
    user_tag_affinities: dict[str, float],
    tag_uplifts: dict[str, float],
    max_positive_uplift: float,
    emerging_scores: dict[str, float],
    max_emerging_score: float,
    modality_weights: dict[str, float],
    cold_item_max_interactions: int,
    history_items: list[ItemFeatures] | None = None,
    recent_history_items: list[ItemFeatures] | None = None,
) -> float:
    if isinstance(model, LinearPairwiseRanker):
        return score_candidate_with_linear_ranker(
            model,
            candidate,
            item,
            history_profile=history_profile,
            prefix_item_counts=prefix_item_counts,
            recent_item_counts=recent_item_counts,
            trend_item_features=trend_item_features,
            user_tag_affinities=user_tag_affinities,
            tag_uplifts=tag_uplifts,
            max_positive_uplift=max_positive_uplift,
            emerging_scores=emerging_scores,
            max_emerging_score=max_emerging_score,
            modality_weights=modality_weights,
            cold_item_max_interactions=cold_item_max_interactions,
            history_items=history_items,
            recent_history_items=recent_history_items,
        )
    return score_candidate_with_pointwise_hist_gradient_boosting_ranker(
        model,
        candidate,
        item,
        history_profile=history_profile,
        prefix_item_counts=prefix_item_counts,
        recent_item_counts=recent_item_counts,
        trend_item_features=trend_item_features,
        user_tag_affinities=user_tag_affinities,
        tag_uplifts=tag_uplifts,
        max_positive_uplift=max_positive_uplift,
        emerging_scores=emerging_scores,
        max_emerging_score=max_emerging_score,
        modality_weights=modality_weights,
        cold_item_max_interactions=cold_item_max_interactions,
        history_items=history_items,
        recent_history_items=recent_history_items,
    )
