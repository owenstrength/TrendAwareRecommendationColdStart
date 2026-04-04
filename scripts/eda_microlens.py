from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight EDA on MicroLens-100k.")
    parser.add_argument(
        "--data-dir",
        default="data/raw/microlens/MicroLens-100k",
        help="Directory containing the MicroLens-100k raw files.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=72,
        help="Recent window used for trend analysis.",
    )
    parser.add_argument(
        "--top-items",
        type=int,
        default=10,
        help="Number of recent top items to display.",
    )
    parser.add_argument(
        "--top-tags",
        type=int,
        default=10,
        help="Number of tag trends to display.",
    )
    return parser


def _read_text_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            values[row[0].strip()] = ",".join(row[1:]).strip()
    return values


def _read_engagement(path: Path) -> dict[str, tuple[int, int]]:
    values: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            item_id, likes, views = parts
            values[item_id] = (int(likes), int(views))
    return values


def _iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pairs_path = data_dir / "MicroLens-100k_pairs.csv"
    title_path = data_dir / "MicroLens-100k_title_en.csv"
    tags_path = data_dir / "tags_to_summary.csv"
    engagement_path = data_dir / "MicroLens-100k_likes_and_views.txt"

    titles = _read_text_map(title_path)
    tags = _read_text_map(tags_path)
    engagement = _read_engagement(engagement_path)

    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    min_timestamp: int | None = None
    max_timestamp: int | None = None

    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = int(row["timestamp"])
            user_id = row["user"]
            item_id = row["item"]

            user_counts[user_id] += 1
            item_counts[item_id] += 1
            min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

    if min_timestamp is None or max_timestamp is None:
        raise RuntimeError("No interactions found in the dataset.")

    recent_window_ms = args.window_hours * 3600 * 1000
    recent_item_counts: Counter[str] = Counter()
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = int(row["timestamp"])
            if max_timestamp - timestamp <= recent_window_ms:
                recent_item_counts[row["item"]] += 1

    overall_tag_events: Counter[str] = Counter()
    for item_id, count in item_counts.items():
        overall_tag_events[tags.get(item_id, "UNKNOWN")] += count

    recent_tag_events: Counter[str] = Counter()
    for item_id, count in recent_item_counts.items():
        recent_tag_events[tags.get(item_id, "UNKNOWN")] += count

    total_events = sum(item_counts.values())
    total_recent_events = sum(recent_item_counts.values())
    tag_uplift = []
    for tag, recent_count in recent_tag_events.items():
        overall_share = overall_tag_events[tag] / total_events if total_events else 0.0
        recent_share = recent_count / total_recent_events if total_recent_events else 0.0
        tag_uplift.append(
            {
                "tag": tag,
                "recent_share": recent_share,
                "overall_share": overall_share,
                "share_uplift": recent_share - overall_share,
                "recent_events": recent_count,
            }
        )
    tag_uplift.sort(key=lambda row: row["share_uplift"], reverse=True)

    top_recent_items = []
    for item_id, recent_events in recent_item_counts.most_common(args.top_items):
        likes, views = engagement.get(item_id, (0, 0))
        top_recent_items.append(
            {
                "item_id": item_id,
                "tag": tags.get(item_id, ""),
                "title": titles.get(item_id, ""),
                "recent_events": recent_events,
                "total_events": item_counts[item_id],
                "likes": likes,
                "views": views,
            }
        )

    payload = {
        "dataset_overview": {
            "users": len(user_counts),
            "items_with_interactions": len(item_counts),
            "events": total_events,
            "time_start_utc": _iso_from_ms(min_timestamp),
            "time_end_utc": _iso_from_ms(max_timestamp),
        },
        "long_tail": {
            "median_user_interactions": median(user_counts.values()),
            "median_item_interactions": median(item_counts.values()),
            "cold_items_le_10": sum(1 for count in item_counts.values() if count <= 10),
            "warm_items_gt_10": sum(1 for count in item_counts.values() if count > 10),
        },
        "recent_window": {
            "hours": args.window_hours,
            "events": total_recent_events,
            "unique_items": len(recent_item_counts),
            "top_recent_items": top_recent_items,
            "tag_uplift": tag_uplift[: args.top_tags],
            "top_recent_tags_by_events": recent_tag_events.most_common(args.top_tags),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
