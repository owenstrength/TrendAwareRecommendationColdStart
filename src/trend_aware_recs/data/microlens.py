"""MicroLens dataset loading utilities."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from trend_aware_recs.config import DataConfig
from trend_aware_recs.data.schema import Interaction, ItemFeatures

_MODALITIES = ("visual", "text", "audio")
_LOCAL_FEATURE_FILENAMES = {
    "visual": "microlens_visual_tfidf_svd.npy",
    "text": "microlens_text_tfidf_svd.npy",
    "audio": "microlens_char_tfidf_svd.npy",
}


def _path_from_string(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path).expanduser()


def _resolve_path(path: str | None) -> Path | None:
    resolved = _path_from_string(path)
    if resolved is None or not resolved.exists():
        return None
    return resolved


def _sorted_item_ids(item_ids: set[str]) -> list[str]:
    def _key(value: str) -> tuple[int, int | str]:
        if value.isdigit():
            return (0, int(value))
        return (1, value)

    return sorted(item_ids, key=_key)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _pad_or_trim(matrix: np.ndarray, feature_dim: int) -> np.ndarray:
    if matrix.shape[1] == feature_dim:
        return matrix.astype(np.float32, copy=False)
    if matrix.shape[1] > feature_dim:
        return matrix[:, :feature_dim].astype(np.float32, copy=False)

    padded = np.zeros((matrix.shape[0], feature_dim), dtype=np.float32)
    padded[:, : matrix.shape[1]] = matrix.astype(np.float32, copy=False)
    return padded


def _read_text_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            item_id = row[0].strip()
            text = ",".join(row[1:]).strip()
            values[item_id] = text
    return values


def _read_comment_snippets(
    path: Path | None,
    max_comments_per_item: int,
) -> tuple[dict[str, str], Counter[str]]:
    if path is None:
        return {}, Counter()

    snippets: dict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            _user_id, item_id, comment = parts
            counts[item_id] += 1
            if len(snippets[item_id]) < max_comments_per_item:
                snippets[item_id].append(comment.strip())

    joined = {item_id: " ".join(texts) for item_id, texts in snippets.items()}
    return joined, counts


def _read_engagement(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}

    engagement: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            item_id, likes, views = parts
            engagement[item_id] = (int(likes), int(views))
    return engagement


def _build_dense_text_features(
    docs: list[str],
    feature_dim: int,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    min_df: int = 2,
) -> np.ndarray:
    if not docs:
        return np.zeros((0, feature_dim), dtype=np.float32)

    # Keep sklearn imports local and single-threaded to avoid OpenMP issues in this environment.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return np.zeros((len(docs), feature_dim), dtype=np.float32)

    if matrix.shape[1] == 0:
        return np.zeros((len(docs), feature_dim), dtype=np.float32)

    if min(matrix.shape) <= 1:
        dense = _pad_or_trim(matrix.toarray().astype(np.float32), feature_dim)
        return _normalize_rows(dense)

    n_components = min(feature_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        dense = _pad_or_trim(matrix.toarray().astype(np.float32), feature_dim)
        return _normalize_rows(dense)

    dense = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(matrix)
    dense = _pad_or_trim(dense.astype(np.float32), feature_dim)
    return _normalize_rows(dense)


def _default_derived_feature_dir(config: DataConfig) -> Path:
    configured = _path_from_string(config.derived_feature_dir)
    if configured is not None:
        return configured

    interaction_path = _path_from_string(config.interaction_path)
    if interaction_path is None:
        raise FileNotFoundError("MicroLens interaction_path is required to derive local features.")
    return interaction_path.parent / "derived_modality_features"


def _official_feature_paths(config: DataConfig) -> dict[str, Path] | None:
    if not config.item_feature_paths:
        return None

    resolved: dict[str, Path] = {}
    for modality in _MODALITIES:
        path = _resolve_path(config.item_feature_paths.get(modality))
        if path is None:
            return None
        resolved[modality] = path
    return resolved


def _derived_feature_paths(config: DataConfig) -> dict[str, Path]:
    base_dir = _default_derived_feature_dir(config)
    return {
        modality: base_dir / filename
        for modality, filename in _LOCAL_FEATURE_FILENAMES.items()
    }


def _align_feature_matrix(
    matrix: np.ndarray,
    item_ids: list[str],
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {matrix.shape}.")

    if item_ids and all(item_id.isdigit() for item_id in item_ids):
        max_item_id = max(int(item_id) for item_id in item_ids)
        if matrix.shape[0] >= max_item_id:
            return np.asarray([matrix[int(item_id) - 1] for item_id in item_ids], dtype=np.float32)

    if matrix.shape[0] != len(item_ids):
        raise ValueError(
            f"Feature matrix row count {matrix.shape[0]} does not match item count {len(item_ids)}."
        )
    return matrix.astype(np.float32, copy=False)


def _load_feature_matrices(
    feature_paths: dict[str, Path],
    item_ids: list[str],
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for modality in _MODALITIES:
        matrix = np.load(feature_paths[modality], mmap_mode="r")
        matrices[modality] = _normalize_rows(_align_feature_matrix(matrix, item_ids))
    return matrices


def _build_local_feature_matrices(
    item_ids: list[str],
    titles: dict[str, str],
    tags: dict[str, str],
    comments: dict[str, str],
    feature_dim: int,
) -> dict[str, np.ndarray]:
    title_docs = [titles.get(item_id, "") for item_id in item_ids]
    tag_docs = [tags.get(item_id, "") for item_id in item_ids]
    comment_docs = [comments.get(item_id, "") for item_id in item_ids]

    visual_docs = [
        " ".join(part for part in (tag, tag, title) if part)
        for tag, title in zip(tag_docs, title_docs, strict=True)
    ]
    text_docs = [
        " ".join(part for part in (title, tag, comment) if part)
        for title, tag, comment in zip(title_docs, tag_docs, comment_docs, strict=True)
    ]
    char_docs = [comment if comment else f"{title} {tag}" for title, tag, comment in zip(title_docs, tag_docs, comment_docs, strict=True)]

    return {
        "visual": _build_dense_text_features(
            visual_docs,
            feature_dim,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40000,
        ),
        "text": _build_dense_text_features(
            text_docs,
            feature_dim,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=60000,
        ),
        "audio": _build_dense_text_features(
            char_docs,
            feature_dim,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=60000,
            min_df=1,
        ),
    }


def build_or_load_local_microlens_feature_files(
    config: DataConfig,
    item_ids: list[str],
    titles: dict[str, str],
    tags: dict[str, str],
    comments: dict[str, str],
    *,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    feature_paths = _derived_feature_paths(config)
    if not force_rebuild and all(path.exists() for path in feature_paths.values()):
        return feature_paths

    matrices = _build_local_feature_matrices(
        item_ids=item_ids,
        titles=titles,
        tags=tags,
        comments=comments,
        feature_dim=config.feature_dim,
    )
    for modality, path in feature_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrices[modality].astype(np.float32))
    return feature_paths


def resolve_microlens_feature_source(config: DataConfig) -> str:
    return "official" if _official_feature_paths(config) is not None else "local_dense"


def _load_or_build_feature_matrices(
    config: DataConfig,
    item_ids: list[str],
    titles: dict[str, str],
    tags: dict[str, str],
    comments: dict[str, str],
) -> tuple[dict[str, np.ndarray], str]:
    official_paths = _official_feature_paths(config)
    if official_paths is not None:
        return _load_feature_matrices(official_paths, item_ids), "official"

    local_paths = build_or_load_local_microlens_feature_files(
        config=config,
        item_ids=item_ids,
        titles=titles,
        tags=tags,
        comments=comments,
    )
    return _load_feature_matrices(local_paths, item_ids), "local_dense"


def load_microlens_dataset(config: DataConfig) -> tuple[list[Interaction], dict[str, ItemFeatures]]:
    """Load interactions and multimodal item features for MicroLens.

    Preference order:
    1. Official published extracted feature arrays if present on disk.
    2. Cached dense local feature files built from titles, tags, and comments.
    """

    interaction_path = _resolve_path(config.interaction_path)
    if interaction_path is None:
        raise FileNotFoundError("MicroLens interaction_path is required and must exist.")

    interactions: list[Interaction] = []
    interaction_counts: Counter[str] = Counter()
    with interaction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = row[config.user_column]
            item_id = row[config.item_column]
            timestamp = int(row[config.timestamp_column])
            interactions.append(Interaction(user_id=user_id, item_id=item_id, timestamp=timestamp))
            interaction_counts[item_id] += 1

    titles = _read_text_map(_resolve_path(config.title_path))
    tags = _read_text_map(_resolve_path(config.tags_path))
    comments, comment_counts = _read_comment_snippets(
        _resolve_path(config.comments_path),
        max_comments_per_item=config.max_comments_per_item,
    )
    engagement = _read_engagement(_resolve_path(config.engagement_path))

    item_ids = _sorted_item_ids(
        set(interaction_counts)
        | set(titles)
        | set(tags)
        | set(comments)
        | set(comment_counts)
        | set(engagement)
    )
    matrices, _feature_source = _load_or_build_feature_matrices(
        config=config,
        item_ids=item_ids,
        titles=titles,
        tags=tags,
        comments=comments,
    )

    items: dict[str, ItemFeatures] = {}
    for index, item_id in enumerate(item_ids):
        title = titles.get(item_id, "")
        tag = tags.get(item_id, "")
        likes, views = engagement.get(item_id, (0, 0))
        interaction_count = interaction_counts.get(item_id, 0)
        comment_count = comment_counts.get(item_id, 0)

        items[item_id] = ItemFeatures(
            item_id=item_id,
            visual=matrices["visual"][index],
            text=matrices["text"][index],
            audio=matrices["audio"][index],
            interaction_count=interaction_count,
            title=title,
            tag=tag,
            likes=likes,
            views=views,
            comment_count=comment_count,
        )

    return interactions, items
