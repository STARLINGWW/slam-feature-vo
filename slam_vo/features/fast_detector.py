"""FAST-9 corner detector — hand-written, no cv2.FAST / goodFeaturesToTrack.

Algorithm:
  For each pixel p, sample 16 pixels on the Bresenham circle (radius 3).
  p is a corner if >= n_consecutive contiguous circle pixels are all brighter
  than p+threshold OR all darker than p-threshold.
  Score = sum(|diff|) over all 16 circle pixels; NMS keeps local maxima.
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter

from slam_vo.features.base import BaseDetector

logger = logging.getLogger(__name__)

# Bresenham circle of radius 3: (row_offset, col_offset), clockwise from top
_CIRCLE: np.ndarray = np.array([
    (-3,  0), (-3,  1), (-2,  2), (-1,  3),
    ( 0,  3), ( 1,  3), ( 2,  2), ( 3,  1),
    ( 3,  0), ( 3, -1), ( 2, -2), ( 1, -3),
    ( 0, -3), (-1, -3), (-2, -2), (-3, -1),
], dtype=np.int32)  # shape (16, 2)

# Compass pixels for the O(1) high-speed pre-filter (indices into _CIRCLE)
_COMPASS_IDX = [0, 4, 8, 12]
_BORDER = 4   # safe border to exclude from detection


class FastDetector(BaseDetector):
    """FAST-9 corner detector with score-based NMS.

    Args:
        threshold: pixel intensity difference threshold.
        nms_radius: NMS suppression radius in pixels.
        max_keypoints: maximum corners to return (top by score).
        n_consecutive: number of consecutive bright/dark pixels required (default 9).
    """

    def __init__(
        self,
        threshold: int = 20,
        nms_radius: int = 4,
        max_keypoints: int = 500,
        n_consecutive: int = 9,
    ):
        self.threshold = threshold
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.n_consecutive = n_consecutive

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect FAST-9 corners in a grayscale image.

        Args:
            image: grayscale image, shape (H, W), dtype uint8.

        Returns:
            Keypoint coordinates, shape (N, 2), each row [x, y], dtype float32.

        Raises:
            ValueError: if image is not 2-D.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

        H, W = image.shape
        img = image.astype(np.int16)
        pad = 3

        # ------------------------------------------------------------------
        # 1. Build (16, H, W) array of circle pixel values using padding
        # ------------------------------------------------------------------
        img_padded = np.pad(img, pad, mode='reflect')
        circle_vals = np.stack([
            img_padded[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
            for dr, dc in _CIRCLE
        ], axis=0)  # (16, H, W)

        diff = circle_vals - img[np.newaxis]  # (16, H, W), signed

        # ------------------------------------------------------------------
        # 2. High-speed pre-filter (Rosten 2006)
        #    FAST-9 requires ≥2 of 4 compass pixels bright or dark;
        #    FAST-12 would require ≥3.  Use ≥2 for correctness with n=9.
        # ------------------------------------------------------------------
        hs = diff[_COMPASS_IDX]  # (4, H, W)
        hs_pass = ((hs > self.threshold).sum(0) >= 2) | ((hs < -self.threshold).sum(0) >= 2)

        # ------------------------------------------------------------------
        # 3. Full FAST-9 test: any n_consecutive contiguous circle pixels
        # ------------------------------------------------------------------
        bright = diff > self.threshold   # (16, H, W)
        dark   = diff < -self.threshold

        bright_d = np.concatenate([bright, bright], axis=0)  # (32, H, W) wrap-around
        dark_d   = np.concatenate([dark,   dark],   axis=0)

        n = self.n_consecutive
        is_corner = np.zeros((H, W), dtype=bool)
        for s in range(16):
            is_corner |= bright_d[s: s + n].all(axis=0)
            is_corner |= dark_d[s: s + n].all(axis=0)

        is_corner &= hs_pass

        # ------------------------------------------------------------------
        # 4. Score = sum |diff| over 16 circle pixels; zero out non-corners
        # ------------------------------------------------------------------
        score = np.abs(diff).sum(axis=0).astype(np.float32)
        score[~is_corner] = 0.0

        # ------------------------------------------------------------------
        # 5. Non-maximum suppression (local max within nms_radius window)
        #    Ties are broken by position index to guarantee uniqueness.
        # ------------------------------------------------------------------
        kernel_size = 2 * self.nms_radius + 1
        # Use float64 for tiebreaking: float32 ULP (~4.86e-4 near score=4080)
        # is too coarse to distinguish per-pixel offsets.  float64 has ample
        # precision (ULP ~9e-13), so a 1e-9/pixel offset is unique for any
        # image size and ensures exactly one winner per NMS window.
        row_idx, col_idx = np.mgrid[0:H, 0:W]
        score64 = score.astype(np.float64)
        score64 += (row_idx * W + col_idx).astype(np.float64) * 1e-9
        local_max64 = maximum_filter(score64, size=kernel_size, mode='constant', cval=0.0)
        nms_mask = (score > 0) & (score64 >= local_max64)

        # Exclude image border
        b = _BORDER
        nms_mask[:b, :]  = False
        nms_mask[-b:, :] = False
        nms_mask[:, :b]  = False
        nms_mask[:, -b:] = False

        ys, xs = np.where(nms_mask)
        if len(xs) == 0:
            logger.debug("FAST: no keypoints detected")
            return np.empty((0, 2), dtype=np.float32)

        # ------------------------------------------------------------------
        # 6. Keep top-k by score
        # ------------------------------------------------------------------
        scores_at = score[ys, xs]
        if len(xs) > self.max_keypoints:
            top_k = np.argpartition(-scores_at, self.max_keypoints)[: self.max_keypoints]
            xs, ys = xs[top_k], ys[top_k]

        keypoints = np.column_stack([xs, ys]).astype(np.float32)
        logger.debug("FAST: detected %d keypoints", len(keypoints))
        return keypoints
