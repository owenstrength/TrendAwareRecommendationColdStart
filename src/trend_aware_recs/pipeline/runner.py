from __future__ import annotations

import argparse
import json
from collections import Counter, deque

from trend_aware_recs.config import load_experiment_bundle
from trend_aware_recs.data.demo import load_demo_dataset
from trend_aware_recs.data.microlens import load_microlens_dataset, resolve_microlens_feature_source
from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from trend_aware_recs.evaluation.splits import (
    AggregateMetrics,
    aggregate_case_metrics,
    build_time_based_holdout,
    compute_split_metrics,
)
from trend_aware_recs.models.baseline import (
    content_nn_rerank,
    no_boost_baseline,
    popularity_rerank,
)
from trend_aware_recs.models.personalization import (
    compute_user_tag_affinities,
    compute_user_virality_affinity,
)
from trend_aware_recs.models.reranker import rerank_with_trend_boost
from trend_aware_recs.models.similarity import average_item_features, weighted_average_item_features
from trend_aware_recs.models.trainable import (
    LinearPairwiseRanker,
    PointwiseHistGradientBoostingRanker,
    score_candidates_with_trained_ranker,
    train_trainable_ranker,
)
from trend_aware_recs.models.trend import (
    identify_trending_items,
    score_emerging_item_ids,
    select_top_scored_item_ids,
    select_trending_item_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trend-aware cold-start experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/baseline.yaml",
        help="Path to the experiment config file.",
    )
    return parser


def _user_result(
    user_id: str,
    ranked_ids: list[str],
    relevant_items: set[str],
    items,
    cold_max: int,
    k: int,
) -> dict:
    metrics = compute_split_metrics(
        ranked_items=ranked_ids,
        relevant_items=relevant_items,
        items=items,
        cold_item_max_interactions=cold_max,
        k=k,
        user_id=user_id,
    )
    return {
        "ranked_items": ranked_ids,
        "hit_rate_at_k": metrics.overall.hit_rate,
        "ndcg_at_k": metrics.overall.ndcg,
        "cold_hit_rate_at_k": metrics.cold.hit_rate,
        "cold_ndcg_at_k": metrics.cold.ndcg,
        "warm_hit_rate_at_k": metrics.warm.hit_rate,
        "warm_ndcg_at_k": metrics.warm.ndcg,
    }


def _aggregate_payload(metrics: dict[str, AggregateMetrics]) -> dict[str, dict[str, float | int]]:
    return {
        split: {
            "hit_rate_at_k": values.hit_rate,
            "ndcg_at_k": values.ndcg,
            "num_cases": values.num_cases,
        }
        for split, values in metrics.items()
    }


def _trend_preview(item_ids: list[str], items, limit: int = 10) -> list[dict[str, str | int]]:
    preview: list[dict[str, str | int]] = []
    for item_id in item_ids[:limit]:
        item = items.get(item_id)
        if item is None:
            continue
        preview.append(
            {
                "item_id": item_id,
                "tag": item.tag,
                "title": item.title,
                "interaction_count": item.interaction_count,
            }
        )
    return preview


def _tag_uplift_scores(
    overall_tag_counts: Counter[str],
    recent_tag_counts: Counter[str],
) -> dict[str, float]:
    total_overall = sum(overall_tag_counts.values())
    total_recent = sum(recent_tag_counts.values())
    if total_overall == 0 or total_recent == 0:
        return {}

    scores: dict[str, float] = {}
    for tag, recent_count in recent_tag_counts.items():
        scores[tag] = (recent_count / total_recent) - (overall_tag_counts[tag] / total_overall)
    return scores


def _min_max_normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {item_id: 0.0 for item_id in score_map}
    scale = maximum - minimum
    return {item_id: (value - minimum) / scale for item_id, value in score_map.items()}


def run_demo_experiment(config_path: str) -> dict:
    experiment, _data_config, model = load_experiment_bundle(config_path)
    interactions, items, user_candidates = load_demo_dataset(seed=experiment.seed)
    current_time = max(interaction.timestamp for interaction in interactions)
    trending_items = identify_trending_items(
        interactions=interactions,
        current_time=current_time,
        trend_window_hours=model.trend_window_hours,
        top_fraction=model.top_trend_fraction,
        min_items=model.min_trend_items,
    )

    cold_max = model.cold_item_max_interactions
    evaluation_pool = {
        c.item_id
        for candidates in user_candidates.values()
        for c in candidates
        if c.item_id in items and items[c.item_id].interaction_count <= cold_max
    }

    results: dict[str, dict] = {}
    for user_id, candidates in user_candidates.items():
        relevant_items = evaluation_pool

        reranked = rerank_with_trend_boost(
            user_id=user_id,
            candidates=candidates,
            interactions=interactions,
            items=items,
            trending_item_ids=trending_items,
            cold_item_max_interactions=model.cold_item_max_interactions,
            modality_weights=model.modality_weights,
            boost_weight=model.boost_weight,
            smoothing=model.virality_prior_smoothing,
            personalization_strength=model.personalization_strength,
        )
        ranked_ids = [s.item_id for s in reranked]

        reranked_ablation = rerank_with_trend_boost(
            user_id=user_id,
            candidates=candidates,
            interactions=interactions,
            items=items,
            trending_item_ids=trending_items,
            cold_item_max_interactions=model.cold_item_max_interactions,
            modality_weights=model.modality_weights,
            boost_weight=model.boost_weight,
            smoothing=model.virality_prior_smoothing,
            disable_personalization=True,
            personalization_strength=model.personalization_strength,
        )
        ablation_ids = [s.item_id for s in reranked_ablation]

        no_boost = no_boost_baseline(candidates)
        no_boost_ids = [s.item_id for s in no_boost]

        pop_reranked = popularity_rerank(candidates, items, boost_weight=model.boost_weight)
        pop_ids = [s.item_id for s in pop_reranked]

        nn_reranked = content_nn_rerank(
            user_id=user_id,
            candidates=candidates,
            interactions=interactions,
            items=items,
            modality_weights=model.modality_weights,
            boost_weight=model.boost_weight,
        )
        nn_ids = [s.item_id for s in nn_reranked]

        k = experiment.top_k

        results[user_id] = {
            "trend_aware": _user_result(user_id, ranked_ids, relevant_items, items, cold_max, k),
            "ablation_no_personalization": {
                "ranked_items": ablation_ids,
                "hit_rate_at_k": hit_rate_at_k(ablation_ids, relevant_items, k),
                "ndcg_at_k": ndcg_at_k(ablation_ids, relevant_items, k),
            },
            "baseline_no_boost": {
                "ranked_items": no_boost_ids,
                "hit_rate_at_k": hit_rate_at_k(no_boost_ids, relevant_items, k),
                "ndcg_at_k": ndcg_at_k(no_boost_ids, relevant_items, k),
            },
            "baseline_popularity": {
                "ranked_items": pop_ids,
                "hit_rate_at_k": hit_rate_at_k(pop_ids, relevant_items, k),
                "ndcg_at_k": ndcg_at_k(pop_ids, relevant_items, k),
            },
            "baseline_content_nn": {
                "ranked_items": nn_ids,
                "hit_rate_at_k": hit_rate_at_k(nn_ids, relevant_items, k),
                "ndcg_at_k": ndcg_at_k(nn_ids, relevant_items, k),
            },
        }

    return {
        "experiment_name": experiment.experiment_name,
        "loader": "demo",
        "trending_items": trending_items,
        "results": results,
    }


def run_microlens_experiment(config_path: str) -> dict:
    experiment, data_config, model = load_experiment_bundle(config_path)
    feature_source = resolve_microlens_feature_source(data_config)
    interactions, items = load_microlens_dataset(data_config)
    holdout = build_time_based_holdout(
        interactions=interactions,
        items=items,
        seed=experiment.seed,
        evaluation_fraction=data_config.evaluation_fraction,
        negatives_per_user=data_config.negatives_per_user,
        min_user_train_interactions=data_config.min_user_train_interactions,
        max_eval_users=data_config.max_eval_users,
        cold_item_max_interactions=model.cold_item_max_interactions,
        modality_weights=model.modality_weights,
        base_history_weight=model.base_history_weight,
        base_popularity_weight=model.base_popularity_weight,
    )

    ranked_cases: dict[str, list[tuple]] = {}
    if model.evaluate_trend_aware:
        ranked_cases["trend_aware"] = []
    if model.evaluate_ablation_no_personalization:
        ranked_cases["ablation_no_personalization"] = []
    if model.evaluate_baseline_no_boost:
        ranked_cases["baseline_no_boost"] = []
    if model.evaluate_baseline_popularity:
        ranked_cases["baseline_popularity"] = []
    if model.evaluate_baseline_content_nn:
        ranked_cases["baseline_content_nn"] = []
    trained_ranker = (
        train_trainable_ranker(
            holdout.train_interactions,
            items,
            model_type=model.trainable_model_type,
            seed=experiment.seed,
            max_cases=model.trainable_max_cases,
            negatives_per_case=model.trainable_negatives_per_case,
            min_user_train_interactions=data_config.min_user_train_interactions,
            trend_window_hours=model.trend_window_hours,
            top_trend_fraction=model.top_trend_fraction,
            emerging_fraction=model.emerging_fraction,
            min_trend_items=model.min_trend_items,
            emerging_min_recent_interactions=model.emerging_min_recent_interactions,
            emerging_smoothing=model.emerging_smoothing,
            cold_item_max_interactions=model.cold_item_max_interactions,
            modality_weights=model.modality_weights,
            base_history_weight=model.base_history_weight,
            base_popularity_weight=model.base_popularity_weight,
            epochs=model.trainable_epochs,
            learning_rate=model.trainable_learning_rate,
            l2=model.trainable_l2,
            cold_case_weight=model.trainable_cold_case_weight,
            hgb_max_iter=model.trainable_hgb_max_iter,
            hgb_learning_rate=model.trainable_hgb_learning_rate,
            hgb_max_depth=model.trainable_hgb_max_depth,
            hgb_min_samples_leaf=model.trainable_hgb_min_samples_leaf,
            hard_negative_ratio=model.trainable_hard_negative_ratio,
            hard_negative_pool_size=model.trainable_hard_negative_pool_size,
        )
        if model.trainable_ranker_enabled
        else None
    )
    trained_method_name = (
        "trained_pairwise_ranker"
        if isinstance(trained_ranker, LinearPairwiseRanker)
        else "trained_pointwise_hgb"
        if isinstance(trained_ranker, PointwiseHistGradientBoostingRanker)
        else None
    )
    if trained_ranker is not None:
        ranked_cases[trained_method_name] = []
        if model.trainable_blend_weight > 0.0 and model.evaluate_trend_aware:
            ranked_cases["trained_hybrid_ranker"] = []
    ordered_interactions = sorted(interactions, key=lambda interaction: interaction.timestamp)
    ordered_cases = sorted(holdout.cases, key=lambda case: case.timestamp)
    pointer = 0
    window_ms = model.trend_window_hours * 3600 * 1000
    prefix_item_counts: Counter[str] = Counter()
    prefix_tag_counts: Counter[str] = Counter()
    recent_item_counts: Counter[str] = Counter()
    recent_tag_counts: Counter[str] = Counter()
    recent_window: deque = deque()
    latest_trending_items: list[str] = []
    latest_emerging_items: list[str] = []

    for case in ordered_cases:
        while pointer < len(ordered_interactions) and ordered_interactions[pointer].timestamp < case.timestamp:
            interaction = ordered_interactions[pointer]
            item = items.get(interaction.item_id)
            prefix_item_counts[interaction.item_id] += 1
            recent_item_counts[interaction.item_id] += 1
            recent_window.append(interaction)
            if item is not None and item.tag:
                prefix_tag_counts[item.tag] += 1
                recent_tag_counts[item.tag] += 1
            pointer += 1

        while recent_window and case.timestamp - recent_window[0].timestamp > window_ms:
            expired = recent_window.popleft()
            recent_item_counts[expired.item_id] -= 1
            if recent_item_counts[expired.item_id] <= 0:
                del recent_item_counts[expired.item_id]
            expired_item = items.get(expired.item_id)
            if expired_item is not None and expired_item.tag:
                recent_tag_counts[expired_item.tag] -= 1
                if recent_tag_counts[expired_item.tag] <= 0:
                    del recent_tag_counts[expired_item.tag]

        latest_trending_items = select_trending_item_ids(
            recent_item_counts,
            top_fraction=model.top_trend_fraction,
            min_items=model.min_trend_items,
        )
        emerging_scores = score_emerging_item_ids(
            recent_item_counts,
            prefix_item_counts,
            smoothing=model.emerging_smoothing,
            min_recent_interactions=model.emerging_min_recent_interactions,
        )
        latest_emerging_items = select_top_scored_item_ids(
            emerging_scores,
            top_fraction=model.emerging_fraction,
            min_items=model.min_trend_items,
        )

        trend_item_features = []
        trend_source_items = [items[item_id] for item_id in latest_trending_items if item_id in items]
        trend_weights = [recent_item_counts[item_id] for item_id in latest_trending_items if item_id in items]
        if trend_source_items:
            trend_item_features.append(
                weighted_average_item_features(
                    "trend_centroid",
                    trend_source_items,
                    trend_weights,
                )
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
        history_items = [
            items[interaction.item_id]
            for interaction in case.history_interactions
            if interaction.item_id in items
        ]
        recent_history_items = history_items[-3:]
        history_profile = average_item_features("user_profile", history_items) if history_items else None
        user_tag_affinities = compute_user_tag_affinities(case.history_interactions, items)
        tag_uplifts = _tag_uplift_scores(prefix_tag_counts, recent_tag_counts)
        max_positive_uplift = max((score for score in tag_uplifts.values() if score > 0), default=0.0)
        max_emerging_score = max(emerging_scores.values(), default=0.0)
        extra_candidate_boosts: dict[str, float] = {}
        for candidate in case.candidates:
            item = items.get(candidate.item_id)
            if item is None:
                extra_candidate_boosts[candidate.item_id] = 0.0
                continue
            extra_boost = 0.0
            if item.tag and max_positive_uplift > 0.0:
                uplift = max(0.0, tag_uplifts.get(item.tag, 0.0))
                tag_affinity = user_tag_affinities.get(item.tag, 0.0)
                extra_boost += model.tag_boost_weight * tag_affinity * (uplift / max_positive_uplift)
            if max_emerging_score > 0.0:
                extra_boost += (
                    model.emerging_boost_weight
                    * (emerging_scores.get(candidate.item_id, 0.0) / max_emerging_score)
                )
            extra_candidate_boosts[candidate.item_id] = extra_boost
        active_trend_item_ids = list(dict.fromkeys(latest_trending_items + latest_emerging_items))
        virality_affinity = compute_user_virality_affinity(
            case.user_id,
            case.history_interactions,
            active_trend_item_ids,
            smoothing=model.virality_prior_smoothing,
        )
        trend_score_map: dict[str, float] = {}
        if model.evaluate_trend_aware:
            reranked = rerank_with_trend_boost(
                user_id=case.user_id,
                candidates=case.candidates,
                interactions=case.history_interactions,
                items=items,
                trending_item_ids=active_trend_item_ids,
                cold_item_max_interactions=model.cold_item_max_interactions,
                modality_weights=model.modality_weights,
                boost_weight=model.boost_weight,
                smoothing=model.virality_prior_smoothing,
                trend_item_features=trend_item_features,
                precomputed_affinity=virality_affinity,
                item_interaction_counts=dict(prefix_item_counts),
                extra_candidate_boosts=extra_candidate_boosts,
                personalization_strength=model.personalization_strength,
            )
            ranked_cases["trend_aware"].append((case, [score.item_id for score in reranked]))
            trend_score_map = {score.item_id: score.final_score for score in reranked}

        if model.evaluate_ablation_no_personalization:
            reranked_ablation = rerank_with_trend_boost(
                user_id=case.user_id,
                candidates=case.candidates,
                interactions=case.history_interactions,
                items=items,
                trending_item_ids=active_trend_item_ids,
                cold_item_max_interactions=model.cold_item_max_interactions,
                modality_weights=model.modality_weights,
                boost_weight=model.boost_weight,
                smoothing=model.virality_prior_smoothing,
                disable_personalization=True,
                trend_item_features=trend_item_features,
                item_interaction_counts=dict(prefix_item_counts),
                extra_candidate_boosts=extra_candidate_boosts,
                personalization_strength=model.personalization_strength,
            )
            ranked_cases["ablation_no_personalization"].append(
                (case, [score.item_id for score in reranked_ablation])
            )

        if trained_ranker is not None:
            trained_scores = score_candidates_with_trained_ranker(
                trained_ranker,
                case.candidates,
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
                modality_weights=model.modality_weights,
                cold_item_max_interactions=model.cold_item_max_interactions,
                history_items=history_items,
                recent_history_items=recent_history_items,
            )
            trained_scores.sort(key=lambda row: (-row[1], row[0]))
            ranked_cases[trained_method_name].append(
                (case, [item_id for item_id, _score in trained_scores])
            )
            if model.trainable_blend_weight > 0.0 and trend_score_map:
                trend_norm = _min_max_normalize(trend_score_map)
                trained_norm = _min_max_normalize({item_id: score for item_id, score in trained_scores})
                hybrid_scores = []
                for candidate in case.candidates:
                    item_id = candidate.item_id
                    hybrid_scores.append(
                        (
                            item_id,
                            trend_norm.get(item_id, 0.0)
                            + model.trainable_blend_weight * trained_norm.get(item_id, 0.0),
                        )
                    )
                hybrid_scores.sort(key=lambda row: (-row[1], row[0]))
                ranked_cases["trained_hybrid_ranker"].append(
                    (case, [item_id for item_id, _score in hybrid_scores])
                )

        if model.evaluate_baseline_no_boost:
            no_boost = no_boost_baseline(case.candidates)
            ranked_cases["baseline_no_boost"].append((case, [score.item_id for score in no_boost]))

        if model.evaluate_baseline_popularity:
            pop_reranked = popularity_rerank(
                case.candidates,
                items,
                boost_weight=model.boost_weight,
                popularity_counts=dict(prefix_item_counts),
            )
            ranked_cases["baseline_popularity"].append((case, [score.item_id for score in pop_reranked]))

        if model.evaluate_baseline_content_nn:
            nn_reranked = content_nn_rerank(
                user_id=case.user_id,
                candidates=case.candidates,
                interactions=case.history_interactions,
                items=items,
                modality_weights=model.modality_weights,
                boost_weight=model.boost_weight,
                history_items=history_items,
            )
            ranked_cases["baseline_content_nn"].append((case, [score.item_id for score in nn_reranked]))

    metrics = {
        method_name: _aggregate_payload(aggregate_case_metrics(method_cases, experiment.top_k))
        for method_name, method_cases in ranked_cases.items()
    }

    return {
        "experiment_name": experiment.experiment_name,
        "loader": data_config.loader,
        "feature_source": feature_source,
        "trained_ranker": (
            {
                "num_training_pairs": (
                    trained_ranker.num_pairs
                    if isinstance(trained_ranker, LinearPairwiseRanker)
                    else None
                ),
                "num_training_cases": trained_ranker.num_cases,
                "num_training_cold_cases": trained_ranker.num_cold_cases,
                "num_training_warm_cases": trained_ranker.num_warm_cases,
                "cold_case_weight": trained_ranker.cold_case_weight,
                "model_type": model.trainable_model_type,
                "num_training_examples": (
                    trained_ranker.num_examples
                    if isinstance(trained_ranker, PointwiseHistGradientBoostingRanker)
                    else None
                ),
                "num_positive_examples": (
                    trained_ranker.num_positive_examples
                    if isinstance(trained_ranker, PointwiseHistGradientBoostingRanker)
                    else None
                ),
                "num_negative_examples": (
                    trained_ranker.num_negative_examples
                    if isinstance(trained_ranker, PointwiseHistGradientBoostingRanker)
                    else None
                ),
            }
            if trained_ranker is not None
            else None
        ),
        "trained_method_name": trained_method_name,
        "num_interactions": len(interactions),
        "num_items": len(items),
        "cutoff_time": holdout.cutoff_time,
        "num_eval_cases": len(holdout.cases),
        "num_trending_items": len(latest_trending_items),
        "num_emerging_items": len(latest_emerging_items),
        "trend_preview": _trend_preview(latest_trending_items, items),
        "emerging_preview": _trend_preview(latest_emerging_items, items),
        "metrics": metrics,
        "notes": [
            (
                "MicroLens run uses official published extracted modality arrays."
                if feature_source == "official"
                else "MicroLens run uses cached dense local modality arrays built from titles, tags, and comments."
            ),
            "Evaluation cases use only items observable by each case timestamp, with time-aware popularity and trend state.",
            "Cold-start boosts combine volume-trend and emerging-item centroids with tag uplift and emerging-score priors from recent EDA signals.",
            "The trained pairwise ranker, when enabled, learns from pre-cutoff positive-vs-negative candidate pairs under the same time-aware protocol.",
            "A trained-hybrid ranker, when enabled, blends the learned score with the cold-aware heuristic score.",
        ],
    }


def run_experiment(config_path: str) -> dict:
    _experiment, data_config, _model = load_experiment_bundle(config_path)
    if data_config.loader == "microlens":
        return run_microlens_experiment(config_path)
    return run_demo_experiment(config_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_experiment(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
