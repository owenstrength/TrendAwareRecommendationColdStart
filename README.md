# Trend-Aware Recommendation Cold Start

Project scaffold for the proposal in [`proposal.pdf`](/Users/owenstrength/Documents/docs/school/gen_ai/TrendAwareRecommendationColdStart/proposal.pdf): a modular, trend-aware score boosting layer for cold-start short-video recommendation.

## Goals

- Use MicroLens interaction logs and multimodal item features.
- Identify currently trending videos from a sliding time window.
- Boost cold-start items when they are semantically close to current trends.
- Scale that boost per user based on historical affinity for viral content.
- Evaluate against base, popularity-only, and content-only baselines.

## Project Layout

```text
.
|-- configs/
|   |-- data/
|   |-- experiments/
|   `-- model/
|-- scripts/
|-- src/trend_aware_recs/
|   |-- data/
|   |-- evaluation/
|   |-- models/
|   |-- pipeline/
|   `-- utils/
`-- tests/
```

## Quick Start

1. Create a virtual environment.
2. Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

3. Review and update dataset paths in [`configs/data/microlens.example.yaml`](/Users/owenstrength/Documents/docs/school/gen_ai/TrendAwareRecommendationColdStart/configs/data/microlens.example.yaml).
4. Run the demo pipeline:

```bash
python scripts/run_experiment.py --config configs/experiments/baseline.yaml
```

## Dataset

The proposal references the official MicroLens dataset. The repository script
[`scripts/get_microlens.sh`](/Users/owenstrength/Documents/docs/school/gen_ai/TrendAwareRecommendationColdStart/scripts/get_microlens.sh)
downloads the public files from the official Westlake host.

Default behavior:

- Downloads `MicroLens-100k` core interaction/metadata files.
- Downloads the published extracted modality features for 100k.
- Does not download large raw media archives unless explicitly requested.

Examples:

```bash
bash scripts/get_microlens.sh
bash scripts/get_microlens.sh --subset 50k --core-only
bash scripts/get_microlens.sh --subset 100k --with-covers --with-videos
```

## Initial Roadmap

- Replace the in-memory demo dataset with MicroLens loaders.
- Add a real base recommender score source.
- Tune trend window, similarity metric, and virality-affinity definition.
- Add cold-start splits and report NDCG@K / HitRate@K by item group.
