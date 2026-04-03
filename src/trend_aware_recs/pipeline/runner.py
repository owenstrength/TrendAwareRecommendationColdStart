from __future__ import annotations

import argparse
import json

from trend_aware_recs.config import load_experiment_bundle
from trend_aware_recs.data.demo import load_demo_dataset
from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from trend_aware_recs.evaluation.splits import compute_split_metrics
from trend_aware_recs.models.baseline import content_nn_rerank, no_boost_baseline, popularity_rerank
from trend_aware_recs.models.reranker import rerank_with_trend_boost
from trend_aware_recs.models.trend import identify_trending_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trend-aware cold-start demo experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/baseline.yaml",
        help="Path to the experiment config file.",
    )
    return parser


def _user_result(
    user_id: str, ranked_ids: list[str], relevant_items: set, items, cold_max: int, k: int
) -> dict:
    """Build a per-user result dict with overall and split metrics."""
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


def run_demo_experiment(config_path: str) -> dict:
    experiment, _data_config, model = load_experiment_bundle(config_path)
    interactions, items, user_candidates = load_demo_dataset(seed=experiment.seed)
    current_time = max(interaction.timestamp for interaction in interactions)
    trending_items = identify_trending_items(
        interactions=interactions,
        current_time=current_time,
        trend_window_hours=model.trend_window_hours,
        top_fraction=model.top_trend_fraction,
    )

    # Evaluation pool: all cold-start candidates across all users.
    # These are treated as the relevant set for the demo evaluation, since
    # the demo has no ground-truth held-out interactions.
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

        # --- Proposed method: trend-aware boost ---
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
        )
        ranked_ids = [s.item_id for s in reranked]

        # --- Ablation: no personalization (affinity fixed to 1.0) ---
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
        )
        ablation_ids = [s.item_id for s in reranked_ablation]

        # --- Baseline i: no boost ---
        no_boost = no_boost_baseline(candidates)
        no_boost_ids = [s.item_id for s in no_boost]

        # --- Baseline ii: popularity heuristic ---
        pop_reranked = popularity_rerank(candidates, items, boost_weight=model.boost_weight)
        pop_ids = [s.item_id for s in pop_reranked]

        # --- Baseline iii: content-only NN ---
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
        "trending_items": trending_items,
        "results": results,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_demo_experiment(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
