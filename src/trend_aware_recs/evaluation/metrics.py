from __future__ import annotations

import math


def hit_rate_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    top_k = ranked_items[:k]
    return float(any(item in relevant_items for item in top_k))


def ndcg_at_k(ranked_items: list[str], relevant_items: set[str], k: int) -> float:
    top_k = ranked_items[:k]
    dcg = 0.0
    for index, item in enumerate(top_k, start=1):
        if item in relevant_items:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg
