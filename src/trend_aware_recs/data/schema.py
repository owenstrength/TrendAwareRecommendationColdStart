from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interaction:
    """A single user–item interaction event."""

    user_id: str
    item_id: str
    timestamp: int


@dataclass
class ItemFeatures:
    """Multimodal feature representation for a single item."""

    item_id: str
    visual: np.ndarray
    text: np.ndarray
    audio: np.ndarray
    interaction_count: int


@dataclass(frozen=True)
class CandidateScore:
    """A candidate item with a base recommendation score."""

    item_id: str
    base_score: float
