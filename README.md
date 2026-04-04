# Trend-Aware Recommendation Cold Start

Implementation of the proposal in `proposal.pdf`: a strict cold-start recommendation system for MicroLens that combines time-aware trend signals, multimodal item features, and a trained reranker.

## What This Project Is Trying To Solve

The proposal is not about general recommendation quality. It is about strict item cold start:

- new or low-interaction short videos are hard to recommend because collaborative signals are weak
- the system should improve ranking for those cold items at recommendation time
- evaluation should use time-aware splits and only rank items that were actually observable at that timestamp

That is the benchmark this repository now treats as primary.

## Current Best Result

The current best verified strict result is:

- local generated artifact: `artifacts/microlens_100k_trained_pointwise_hgb_model_only_full_catalog_50_results.json`
- config: `configs/experiments/microlens_100k_trained_pointwise_hgb_model_only_full_catalog_50.yaml`
- data protocol: full-catalog ranking over all observable items, `50` eval users, `10` cold cases
- feature source: official published MicroLens extracted modality arrays

Metrics for `trained_pointwise_hgb`:

| Split | HR@10 | NDCG@10 | Cases |
| --- | ---: | ---: | ---: |
| Overall | 0.1000 | 0.0797 | 50 |
| Cold | 0.2000 | 0.1356 | 10 |
| Warm | 0.0750 | 0.0658 | 40 |

This model is currently the best strict cold-start model in the repo.

## Strict Cold-Start Comparison

The strict baseline comparison comes from:

- local generated artifact: `artifacts/microlens_100k_trained_pointwise_hgb_hardneg_strict_fast_full_catalog_50_results.json`
- config: `configs/experiments/microlens_100k_trained_pointwise_hgb_hardneg_strict_fast_full_catalog_50.yaml`

Those baselines use the same strict `50`-user full-catalog slice.

| Method | Overall HR@10 | Overall NDCG@10 | Cold HR@10 | Cold NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Trained pointwise HGB (current best strict run) | 0.1000 | 0.0797 | 0.2000 | 0.1356 |
| Trend-aware heuristic | 0.0200 | 0.0200 | 0.1000 | 0.1000 |
| Baseline no boost | 0.0400 | 0.0193 | 0.0000 | 0.0000 |
| Baseline popularity | 0.0200 | 0.0100 | 0.0000 | 0.0000 |
| Hard-negative HGB variant | 0.0200 | 0.0200 | 0.1000 | 0.1000 |

The main takeaway is simple:

- the strict trained model beats the strict heuristics and simple baselines
- the hard-negative variant overfit and is worse than the default trained model
- the strict cold-start metric is now the metric that matters for this project

## How It Works

The pipeline has five parts.

### 1. Load MicroLens With Official Multimodal Features

The loader in `src/trend_aware_recs/data/microlens.py` prefers the official extracted modality arrays when they exist:

- visual
- text
- audio

If those arrays are missing, the repo can build cached local dense features from titles, tags, and comments.

### 2. Build A Time-Aware Holdout

`src/trend_aware_recs/evaluation/splits.py` creates one held-out positive per user after a global cutoff time.

Important details:

- the positive item must be unseen by that user before the holdout event
- coldness is defined at recommendation time, not globally
- in strict mode, negatives are not sampled; the candidate set is the full observable item pool at that timestamp

That strict setting is what the project should use for headline cold-start claims.

### 3. Build Time-Aware Trend And Emerging Signals

The trend logic in `src/trend_aware_recs/models/trend.py` uses a rolling window over recent interactions to compute:

- volume-based trending items
- emerging items with strong recent activity relative to prior history
- tag uplift from shifts in recent category share

These signals were motivated by EDA on MicroLens and are still used by both the heuristic reranker and the trained model.

### 4. Score Candidates With A Trained Pointwise Ranker

The strongest current model is the pointwise HGB ranker in `src/trend_aware_recs/models/trainable.py`.

It is trained on pre-cutoff positive/negative candidate examples and uses scalar features such as:

- base recommender score
- time-safe popularity and recent interaction strength
- cold indicator
- user-profile multimodal similarity
- last-item similarity
- recent-history profile similarity
- trend-centroid similarity
- user tag affinity
- tag uplift
- emerging-item score

Snapshot engagement totals like likes/views/comments are intentionally excluded from the trainable feature vector because they are not time-safe in this setup.

### 5. Evaluate With Cold/Warm Breakdown

The evaluator reports:

- overall HR@10 and NDCG@10
- cold HR@10 and NDCG@10
- warm HR@10 and NDCG@10

For this project, the cold split is the primary score.

## Proposal Alignment

The proposal in `proposal.pdf` asked for three broad ideas:

1. strict cold-start evaluation on MicroLens
2. trend-aware multimodal boosting
3. personalization to avoid hurting niche-preferring users

Status:

- `1` is implemented
- `2` is implemented, both as a heuristic and as features inside the trained ranker
- `3` is not the main source of gains right now; the strongest current result comes from the trained cold-aware ranker, not the personalization term

So the project has moved slightly beyond the original proposal architecture, but it still answers the proposal's main cold-start question.

## Comparison To The MicroLens Dataset Paper

The MicroLens dataset paper reports the regular MicroLens-100K benchmark, not this repository's strict cold-start benchmark.

Representative Table 2 numbers from the paper are approximately:

| Paper Model | HR@10 | NDCG@10 |
| --- | ---: | ---: |
| SASRec | 0.0909 | 0.0517 |
| GRU4Rec V | 0.0954 | 0.0517 |
| SASRec V | 0.0948 | 0.0515 |

Our current strict cold-start result is:

| This Repo Model | HR@10 | NDCG@10 |
| --- | ---: | ---: |
| Trained pointwise HGB, strict full-catalog cold-start slice | 0.1000 | 0.0797 |

Interpretation:

- numerically, the current strict run is above the paper's regular benchmark numbers
- scientifically, this is not a perfect apples-to-apples comparison
- the paper's benchmark is a regular recommendation task
- this repository's headline task is strict cold start, which is also the actual proposal target

So the correct claim is:

- for the proposal's strict cold-start objective, the repository now has a defensible positive result
- for a paper-to-paper claim, the next step would be reproducing the dataset paper's exact protocol or its cold-start popularity-bucket analysis

## Headline Artifacts

Recommended files to inspect:

- `artifacts/microlens_100k_trained_pointwise_hgb_model_only_full_catalog_50_results.json`
- `artifacts/microlens_100k_trained_pointwise_hgb_hardneg_strict_fast_full_catalog_50_results.json`
- `artifacts/tmp_exhaustive_eval/microlens_heuristic_exhaustive_50_results.json`
- `artifacts/microlens_100k_official_balanced_results.json`

The last file uses the easier sampled-negative protocol and should not be used as the main strict cold-start result.
These artifact files are generated locally and are ignored by git.

## Reproducing The Main Results

Install:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Run the strict trained model:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/microlens_100k_trained_pointwise_hgb_model_only_full_catalog_50.yaml
```

Run the strict baseline comparison slice:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/microlens_100k_trained_pointwise_hgb_hardneg_strict_fast_full_catalog_50.yaml
```

Run the easier sampled-negative MicroLens experiment:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/experiments/microlens_100k.yaml
```

## Data Download

The repository script `scripts/get_microlens.sh` downloads the public MicroLens files from the official Westlake host.

Examples:

```bash
bash scripts/get_microlens.sh
bash scripts/get_microlens.sh --subset 100k --with-features
```

If official extracted features are unavailable locally, build cached dense fallback features with:

```bash
python scripts/build_microlens_features.py --config configs/data/microlens.example.yaml
```

## Limitations

- the strict headline benchmark currently uses only `50` eval users and `10` cold cases, so it is still noisy
- full-catalog evaluation is much slower than sampled-negative evaluation
- the strongest gains currently come from the trained cold-aware reranker, not from personalization
- the paper comparison is numeric, not protocol-matched

## Next Steps

- increase the strict benchmark size while keeping full-catalog evaluation tractable
- add the dataset paper's popularity-bucket cold-start analysis
- keep only changes that improve strict cold HR/NDCG, not just overall metrics
