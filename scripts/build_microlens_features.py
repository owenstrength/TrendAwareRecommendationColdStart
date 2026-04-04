from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trend_aware_recs.config import DataConfig, load_yaml
from trend_aware_recs.data.microlens import (
    _read_comment_snippets,
    _read_text_map,
    _resolve_path,
    _sorted_item_ids,
    build_or_load_local_microlens_feature_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cached local MicroLens dense feature files.")
    parser.add_argument(
        "--config",
        default="configs/data/microlens.example.yaml",
        help="Path to the data config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild feature files even if cached arrays already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DataConfig.model_validate(load_yaml(args.config))
    titles = _read_text_map(_resolve_path(config.title_path))
    tags = _read_text_map(_resolve_path(config.tags_path))
    comments, _comment_counts = _read_comment_snippets(
        _resolve_path(config.comments_path),
        max_comments_per_item=config.max_comments_per_item,
    )
    item_ids = _sorted_item_ids(set(titles) | set(tags) | set(comments))
    feature_paths = build_or_load_local_microlens_feature_files(
        config=config,
        item_ids=item_ids,
        titles=titles,
        tags=tags,
        comments=comments,
        force_rebuild=args.force,
    )
    for modality, path in feature_paths.items():
        print(f"{modality}\t{path}")


if __name__ == "__main__":
    main()
