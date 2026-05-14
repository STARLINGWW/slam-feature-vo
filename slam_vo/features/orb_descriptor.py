"""ORB descriptor (rBRIEF) — hand-written, no cv2.ORB_create.

Pipeline for each keypoint:
  1. Smooth the local patch with a 5×5 box filter (ORB paper §4).
  2. Compute orientation via intensity centroid in a circle of radius R=15.
  3. Rotate the pre-computed 256-pair BRIEF pattern by the orientation angle.
  4. Compare sampled pixel pairs → 256-bit binary descriptor packed as 32×uint8.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from slam_vo.features.base import BaseDescriptor

logger = logging.getLogger(__name__)

# Patch parameters (following ORB paper)
_PATCH_RADIUS = 15          # orientation centroid radius
_HALF_PATCH = 15            # descriptor sampling radius (patch_size=31)
_N_PAIRS = 256              # number of BRIEF test pairs

# Pre-generate the 256-pair sampling pattern once at import time.
# Uses a Gaussian distribution (σ = patch_radius/3) as in the ORB paper.
def _generate_pattern(n_pairs: int = _N_PAIRS, radius: int = _HALF_PATCH, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sigma = radius / 3.0
    pts = np.round(rng.randn(n_pairs * 4) * sigma).astype(np.int32)
    pts = np.clip(pts, -radius, radius).reshape(n_pairs, 4)  # [x1,y1,x2,y2]
    return pts

_PATTERN: np.ndarray = _generate_pattern()   # (256, 4)

# Precompute circular mask offsets for intensity centroid
def _circle_mask(radius: int) -> Tuple[np.ndarray, np.ndarray]:
    offsets = [(dy, dx)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dx * dx + dy * dy <= radius * radius]
    dy_arr = np.array([o[0] for o in offsets], dtype=np.float32)
    dx_arr = np.array([o[1] for o in offsets], dtype=np.float32)
    return dy_arr, dx_arr

_CENTROID_DY, _CENTROID_DX = _circle_mask(_PATCH_RADIUS)   # each shape (P,)


class OrbDescriptor(BaseDescriptor):
    """rBRIEF descriptor extractor (rotation-aware BRIEF).

    Args:
        patch_size: descriptor patch size (must be odd, default 31).
        n_pairs: number of binary tests (default 256 → 32-byte descriptor).
        use_smoothing: apply 5×5 box filter before sampling (default True).
    """

    def __init__(
        self,
        patch_size: int = 31,
        n_pairs: int = 256,
        use_smoothing: bool = True,
    ):
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.n_pairs = n_pairs
        self.use_smoothing = use_smoothing
        # Use the global pattern (re-generated only if n_pairs differs)
        if n_pairs == _N_PAIRS:
            self._pattern = _PATTERN
        else:
            self._pattern = _generate_pattern(n_pairs, self.half_patch)

    # ------------------------------------------------------------------

    def compute(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Compute rBRIEF descriptors.

        Args:
            image: grayscale image, shape (H, W), dtype uint8.
            keypoints: keypoint coordinates, shape (N, 2) [x, y].

        Returns:
            descriptors: shape (N, n_pairs // 8) uint8 (packed binary).
            Returns empty array shape (0, n_pairs//8) if no keypoints.

        Raises:
            ValueError: if image is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

        N = len(keypoints)
        n_bytes = self.n_pairs // 8
        if N == 0:
            return np.empty((0, n_bytes), dtype=np.uint8)

        pts = keypoints.astype(np.float32)

        # 1. Smooth image before descriptor sampling
        img_smooth = image.astype(np.float32)
        if self.use_smoothing:
            img_smooth = cv2.blur(image, (5, 5)).astype(np.float32)

        # 2. Compute orientation for all keypoints
        angles = _compute_orientation(image.astype(np.float32), pts)

        # 3. Compute rBRIEF
        descriptors = _compute_rbrief(img_smooth, pts, angles, self._pattern, self.half_patch)

        logger.debug("ORB: computed %d descriptors (%d bytes each)", N, n_bytes)
        return descriptors


# ---------------------------------------------------------------------------
# Internal functions (vectorised for N keypoints)
# ---------------------------------------------------------------------------


def _compute_orientation(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Intensity-centroid orientation for N keypoints.

    Args:
        image: (H, W) float32.
        pts: (N, 2) [x, y].

    Returns:
        angles: (N,) float32 in radians.
    """
    N = len(pts)
    P = len(_CENTROID_DY)  # pixels in circle

    # Sample all circle pixels for all N keypoints at once
    rows = pts[:, 1].reshape(N, 1) + _CENTROID_DY.reshape(1, P)   # (N, P)
    cols = pts[:, 0].reshape(N, 1) + _CENTROID_DX.reshape(1, P)

    vals = map_coordinates(
        image, [rows.ravel(), cols.ravel()],
        order=1, mode='constant', cval=0.0,
    ).reshape(N, P)   # (N, P)

    m10 = (vals * _CENTROID_DX.reshape(1, P)).sum(axis=1)  # (N,)
    m01 = (vals * _CENTROID_DY.reshape(1, P)).sum(axis=1)

    return np.arctan2(m01, m10).astype(np.float32)


def _compute_rbrief(
    image: np.ndarray,
    pts: np.ndarray,
    angles: np.ndarray,
    pattern: np.ndarray,
    half_patch: int,
) -> np.ndarray:
    """Rotation-aware BRIEF descriptors.

    Args:
        image: (H, W) float32 (smoothed).
        pts: (N, 2) [x, y].
        angles: (N,) radians.
        pattern: (n_pairs, 4) integer sampling offsets [x1, y1, x2, y2].
        half_patch: clamp radius for rotated offsets.

    Returns:
        descriptors: (N, n_pairs//8) uint8.
    """
    N = len(pts)
    n_pairs = len(pattern)

    cos_a = np.cos(angles).reshape(N, 1)   # (N, 1)
    sin_a = np.sin(angles).reshape(N, 1)

    x1 = pattern[:, 0].astype(np.float32)  # (n_pairs,)
    y1 = pattern[:, 1].astype(np.float32)
    x2 = pattern[:, 2].astype(np.float32)
    y2 = pattern[:, 3].astype(np.float32)

    # Rotate pattern: (N, n_pairs)
    rx1 = np.clip(np.round(cos_a * x1 - sin_a * y1), -half_patch, half_patch)
    ry1 = np.clip(np.round(sin_a * x1 + cos_a * y1), -half_patch, half_patch)
    rx2 = np.clip(np.round(cos_a * x2 - sin_a * y2), -half_patch, half_patch)
    ry2 = np.clip(np.round(sin_a * x2 + cos_a * y2), -half_patch, half_patch)

    # Absolute image coordinates: (N, n_pairs)
    row1 = pts[:, 1:2] + ry1
    col1 = pts[:, 0:1] + rx1
    row2 = pts[:, 1:2] + ry2
    col2 = pts[:, 0:1] + rx2

    # Sample with nearest-neighbour (order=0) as in ORB
    vals1 = map_coordinates(
        image, [row1.ravel(), col1.ravel()], order=0, mode='constant', cval=0.0,
    ).reshape(N, n_pairs)
    vals2 = map_coordinates(
        image, [row2.ravel(), col2.ravel()], order=0, mode='constant', cval=0.0,
    ).reshape(N, n_pairs)

    bits = (vals1 > vals2).astype(np.uint8)   # (N, n_pairs)
    return np.packbits(bits, axis=1, bitorder='big')  # (N, n_pairs//8)
