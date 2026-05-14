"""Local map: container for active 3-D map points."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from slam_vo.vo.map_point import MapPoint

logger = logging.getLogger(__name__)


class LocalMap:
    """Manages the set of active 3-D map points.

    Points are stored in a dict keyed by point_id for O(1) lookup.
    The map is kept small by periodically culling bad points.
    """

    def __init__(self, max_points: int = 3000):
        self._points: Dict[int, MapPoint] = {}
        self._next_id: int = 0
        self.max_points = max_points

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_map_point(
        self,
        position: np.ndarray,
        descriptor: Optional[np.ndarray] = None,
    ) -> MapPoint:
        """Create and register a new map point.

        Args:
            position: (3,) world coordinates.
            descriptor: optional (32,) uint8 descriptor.

        Returns:
            Newly created MapPoint.
        """
        mp = MapPoint(
            point_id=self._next_id,
            position=np.asarray(position, dtype=np.float64),
            descriptor=descriptor,
        )
        self._points[self._next_id] = mp
        self._next_id += 1
        return mp

    def get_point(self, point_id: int) -> Optional[MapPoint]:
        return self._points.get(point_id)

    def remove_bad_points(self) -> int:
        """Remove all points marked as bad. Returns count removed."""
        bad = [pid for pid, mp in self._points.items() if mp.is_bad]
        for pid in bad:
            del self._points[pid]
        if bad:
            logger.debug("Removed %d bad map points", len(bad))
        return len(bad)

    def cull_to_max(self) -> int:
        """If map exceeds max_points, remove oldest points."""
        n_remove = len(self._points) - self.max_points
        if n_remove <= 0:
            return 0
        # Remove oldest (lowest IDs)
        ids_sorted = sorted(self._points.keys())
        for pid in ids_sorted[:n_remove]:
            del self._points[pid]
        logger.debug("Culled %d old map points (map size=%d)", n_remove, len(self._points))
        return n_remove

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self, point_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch-fetch 3-D positions for an array of IDs.

        Args:
            point_ids: (N,) int array.

        Returns:
            pts3d: (N, 3) float64 — NaN row for missing/bad points.
            valid:  (N,) bool.
        """
        pts3d = np.full((len(point_ids), 3), np.nan, dtype=np.float64)
        valid = np.zeros(len(point_ids), dtype=bool)
        for i, pid in enumerate(point_ids):
            mp = self._points.get(int(pid))
            if mp is not None and not mp.is_bad:
                pts3d[i] = mp.position
                valid[i] = True
        return pts3d, valid

    def __len__(self) -> int:
        return len(self._points)
