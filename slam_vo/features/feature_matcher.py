"""Brute-force binary descriptor matcher with ratio test and cross-check.

Hamming distance is computed via XOR + vectorised popcount using a 256-entry
lookup table — no cv2 matcher functions used.
"""

import logging
from typing import Optional

import numpy as np

from slam_vo.features.base import BaseMatcher

logger = logging.getLogger(__name__)

# Precompute popcount for every byte value 0-255
_POPCOUNT: np.ndarray = np.array(
    [bin(i).count('1') for i in range(256)], dtype=np.uint8
)


def hamming_distance_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances between two sets of binary descriptors.

    Args:
        desc1: (N1, D) uint8 — packed binary descriptors.
        desc2: (N2, D) uint8 — packed binary descriptors.

    Returns:
        dist: (N1, N2) int32 — Hamming distance matrix.
    """
    # (N1, 1, D) ^ (1, N2, D) → (N1, N2, D) XOR; then popcount + sum
    xor = desc1[:, np.newaxis, :].astype(np.uint8) ^ desc2[np.newaxis, :, :].astype(np.uint8)
    return _POPCOUNT[xor].sum(axis=2).astype(np.int32)


class BruteForceMatcher(BaseMatcher):
    """Brute-force matcher for packed binary (Hamming) descriptors.

    Args:
        ratio_threshold: Lowe's ratio test threshold (default 0.75).
        cross_check: also require mutual nearest-neighbour (default True).
    """

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
    ):
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """Match descriptors using ratio test and optional cross-check.

        Args:
            desc1: (N1, D) uint8 — packed binary descriptors.
            desc2: (N2, D) uint8 — packed binary descriptors.

        Returns:
            matches: (M, 2) int32 — each row [idx_in_desc1, idx_in_desc2].
            Returns shape (0, 2) if no matches found.
        """
        N1, N2 = len(desc1), len(desc2)
        if N1 == 0 or N2 == 0:
            return np.empty((0, 2), dtype=np.int32)

        dist = hamming_distance_matrix(desc1, desc2)   # (N1, N2)

        # ------------------------------------------------------------------
        # Ratio test: best / second-best distance < threshold
        # ------------------------------------------------------------------
        if N2 >= 2:
            # argsort along axis=1, take first two columns
            sort_idx = np.argpartition(dist, kth=[0, 1], axis=1)[:, :2]  # (N1, 2)
            best_idx  = sort_idx[:, 0]
            # Ensure we really have the two smallest
            d_first  = dist[np.arange(N1), sort_idx[:, 0]]
            d_second = dist[np.arange(N1), sort_idx[:, 1]]
            # Fix: argpartition doesn't guarantee order of the first 2
            swap = d_first > d_second
            best_idx[swap], sort_idx[swap, 1] = sort_idx[swap, 1], best_idx[swap].copy()
            d_best   = dist[np.arange(N1), best_idx]
            d_second = dist[np.arange(N1), sort_idx[:, 1]]

            ratio = d_best.astype(np.float32) / (d_second.astype(np.float32) + 1e-6)
            good = ratio < self.ratio_threshold
        else:
            # Only one descriptor in desc2 — skip ratio test
            best_idx = np.zeros(N1, dtype=np.int32)
            good = np.ones(N1, dtype=bool)

        matches = np.column_stack([np.where(good)[0], best_idx[good]])  # (M, 2)

        # ------------------------------------------------------------------
        # Cross-check: each desc2 match must also best-match back to desc1[i]
        # ------------------------------------------------------------------
        if self.cross_check and len(matches) > 0:
            best_in_d1 = np.argmin(dist, axis=0)  # (N2,) best match in desc1 per desc2
            i_idx = matches[:, 0]
            j_idx = matches[:, 1]
            mutual = best_in_d1[j_idx] == i_idx
            matches = matches[mutual]

        logger.debug(
            "BruteForceMatcher: %d/%d matches (ratio=%.2f, cross=%s)",
            len(matches), N1, self.ratio_threshold, self.cross_check,
        )
        return matches.astype(np.int32) if len(matches) > 0 else np.empty((0, 2), dtype=np.int32)
