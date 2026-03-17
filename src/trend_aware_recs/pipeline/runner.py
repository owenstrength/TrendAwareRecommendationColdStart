from __future__ import annotations

import argparse
import json

from trend_aware_recs.config import load_experiment_bundle
from trend_aware_recs.data.demo import load_demo_dataset
from trend_aware_recs.evaluation.metrics import hit_rate_at_k, ndcg_at_k
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


def run_demo_experiment(config_path: str) -> dict:
    experiment, _data_config, model = load_experiment_bundle(config_path)
    interactions, items, user_candidates = load_demo_dataset()
    current_time = max(interaction.timestamp for interaction in interactions)
    trending_items = identify_trending_items(
        interactions=interactions,
        current_time=current_time,
        trend_window_hours=model.trend_window_hours,
        top_fraction=model.top_trend_fraction,
    )

    results = {}
    for user_id, candidates in user_candidates.items():
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
        ranked_ids = [score.item_id for score in reranked]
        relevant_items = set(trending_items)
        results[user_id] = {
            "ranked_items": ranked_ids,
            "hit_rate_at_k": hit_rate_at_k(ranked_ids, relevant_items, experiment.top_k),
            "ndcg_at_k": ndcg_at_k(ranked_ids, relevant_items, experiment.top_k),
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
