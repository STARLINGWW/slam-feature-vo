"""Frame data structure."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class Frame:
    """A single processed image frame.

    Attributes:
        frame_id:      unique sequential frame index.
        timestamp:     capture time in seconds.
        keypoints:     (N, 2) float32 [x, y] pixel coordinates.
        descriptors:   (N, 32) uint8 ORB descriptors, or None.
        pose:          4×4 T_cw (camera←world), None if not estimated.
        is_keyframe:   True when this frame is a keyframe.
        map_point_ids: mapping {kpt_idx → map_point_id}.
    """

    frame_id: int
    timestamp: float
    keypoints: np.ndarray                              # (N, 2) float32
    descriptors: Optional[np.ndarray] = None           # (N, 32) uint8
    pose: Optional[np.ndarray] = None                  # 4×4 T_cw
    is_keyframe: bool = False
    map_point_ids: Dict[int, int] = field(default_factory=dict)

    @property
    def T_wc(self) -> Optional[np.ndarray]:
        """Camera-to-world transform (camera position in world)."""
        if self.pose is None:
            return None
        return np.linalg.inv(self.pose)

    @property
    def position(self) -> Optional[np.ndarray]:
        """Camera centre in world coordinates, shape (3,)."""
        T = self.T_wc
        return T[:3, 3] if T is not None else None
