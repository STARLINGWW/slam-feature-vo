"""3-D map point data structure."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class MapPoint:
    """A triangulated 3-D point in the world frame.

    Attributes:
        point_id:     unique sequential ID.
        position:     (3,) float64 world-frame coordinates.
        observations: {frame_id → keypoint_index} — frames that observe this point.
        descriptor:   representative (32,) uint8 descriptor, or None.
        is_bad:       True when the point should be ignored (outlier, behind camera…).
    """

    point_id: int
    position: np.ndarray                               # (3,) float64
    observations: Dict[int, int] = field(default_factory=dict)
    descriptor: Optional[np.ndarray] = None
    is_bad: bool = False

    def add_observation(self, frame_id: int, kpt_idx: int) -> None:
        self.observations[frame_id] = kpt_idx
