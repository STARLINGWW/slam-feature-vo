"""Pyramidal KLT optical flow tracker — hand-written, no cv2.calcOpticalFlowPyrLK.

Implements Bouguet's pyramidal Lucas-Kanade with:
  - Gaussian image pyramid via cv2.GaussianBlur + cv2.resize (allowed)
  - Vectorised patch extraction via scipy.ndimage.map_coordinates
  - Per-point 2×2 structure-tensor solve (vectorised over N points)
  - Forward-backward consistency check for outlier rejection
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from slam_vo.features.base import BaseTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------


def build_pyramid(image: np.ndarray, n_levels: int) -> List[np.ndarray]:
    """Build a Gaussian image pyramid.

    Args:
        image: grayscale image, shape (H, W), dtype uint8 or float32.
        n_levels: number of pyramid levels (level 0 = original).

    Returns:
        List of n_levels images, level 0 is finest.
    """
    pyr = [image.astype(np.float32)]
    for _ in range(n_levels - 1):
        blurred = cv2.GaussianBlur(pyr[-1], (5, 5), 1.0)
        pyr.append(cv2.resize(blurred, None, fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_LINEAR))
    return pyr


def extract_patches(
    image: np.ndarray,
    pts: np.ndarray,
    half_win: int,
) -> np.ndarray:
    """Extract (2*half_win+1)² patches for N points via bilinear interpolation.

    Args:
        image: (H, W) float32.
        pts: (N, 2) [x, y] float32.
        half_win: half-window size.

    Returns:
        patches: (N, W, W) float32 where W = 2*half_win+1.
    """
    N = len(pts)
    W = 2 * half_win + 1
    d = np.arange(-half_win, half_win + 1, dtype=np.float32)
    gy, gx = np.meshgrid(d, d, indexing='ij')  # (W, W)

    rows = pts[:, 1].reshape(N, 1, 1) + gy.reshape(1, W, W)  # (N, W, W)
    cols = pts[:, 0].reshape(N, 1, 1) + gx.reshape(1, W, W)

    vals = map_coordinates(
        image, [rows.ravel(), cols.ravel()],
        order=1, mode='constant', cval=0.0,
    )
    return vals.reshape(N, W, W).astype(np.float32)


def image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients.

    Args:
        image: (H, W) float32.

    Returns:
        Ix, Iy: (H, W) float32 each.
    """
    img = image.astype(np.float32)
    Ix = np.zeros_like(img)
    Iy = np.zeros_like(img)
    Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    return Ix, Iy


def lk_at_level(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    prev_pts: np.ndarray,
    init_curr_pts: np.ndarray,
    half_win: int,
    max_iter: int,
    epsilon: float,
    min_eig_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lucas-Kanade optical flow at a single pyramid level.

    Args:
        prev_img: previous frame, (H, W) float32.
        curr_img: current frame, (H, W) float32.
        prev_pts: (N, 2) [x, y] points in previous frame.
        init_curr_pts: (N, 2) initial position estimate in current frame.
        half_win: half-window size.
        max_iter: maximum LK iterations.
        epsilon: convergence threshold (pixels).
        min_eig_thresh: minimum eigenvalue of structure tensor (normalised).

    Returns:
        curr_pts: (N, 2) refined positions in current frame.
        valid: (N,) bool — True if trackable.
    """
    N = len(prev_pts)
    W = 2 * half_win + 1
    W2 = float(W * W)

    Ix, Iy = image_gradients(prev_img)

    prev_patches = extract_patches(prev_img, prev_pts, half_win)
    Ix_p = extract_patches(Ix, prev_pts, half_win)
    Iy_p = extract_patches(Iy, prev_pts, half_win)

    # Structure tensor components (fixed per point)
    A_xx = (Ix_p * Ix_p).sum(axis=(1, 2))   # (N,)
    A_xy = (Ix_p * Iy_p).sum(axis=(1, 2))
    A_yy = (Iy_p * Iy_p).sum(axis=(1, 2))

    det = A_xx * A_yy - A_xy * A_xy         # (N,)
    # Minimum eigenvalue of 2×2 symmetric matrix
    trace = A_xx + A_yy
    disc = np.sqrt(np.maximum((A_xx - A_yy) ** 2 + 4.0 * A_xy ** 2, 0.0))
    min_eig = (trace - disc) * 0.5 / W2     # normalise by window area

    valid = (np.abs(det) > 1e-8) & (min_eig > min_eig_thresh)
    safe_det = np.where(valid, det, 1.0)     # avoid division by zero

    curr_pts = init_curr_pts.copy()

    for _ in range(max_iter):
        curr_patches = extract_patches(curr_img, curr_pts, half_win)
        It = curr_patches - prev_patches     # (N, W, W)

        bx = -(Ix_p * It).sum(axis=(1, 2))  # (N,)
        by = -(Iy_p * It).sum(axis=(1, 2))

        dx = np.where(valid, (A_yy * bx - A_xy * by) / safe_det, 0.0)
        dy = np.where(valid, (A_xx * by - A_xy * bx) / safe_det, 0.0)

        curr_pts[:, 0] += dx.astype(np.float32)
        curr_pts[:, 1] += dy.astype(np.float32)

        if float(np.sqrt(dx ** 2 + dy ** 2).max()) < epsilon:
            break

    return curr_pts, valid


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------


class KltTracker(BaseTracker):
    """Pyramidal KLT feature tracker.

    Args:
        num_levels: number of pyramid levels.
        win_size: tracking window diameter (must be odd).
        max_iterations: maximum LK iterations per level.
        epsilon: convergence threshold in pixels.
        min_eig_threshold: minimum eigenvalue threshold for structure tensor.
        fb_check: enable forward-backward consistency check.
        fb_threshold: max allowed FB error in pixels.
    """

    def __init__(
        self,
        num_levels: int = 4,
        win_size: int = 21,
        max_iterations: int = 30,
        epsilon: float = 0.01,
        min_eig_threshold: float = 1e-4,
        fb_check: bool = True,
        fb_threshold: float = 1.0,
    ):
        self.num_levels = num_levels
        self.half_win = win_size // 2
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.min_eig_threshold = min_eig_threshold
        self.fb_check = fb_check
        self.fb_threshold = fb_threshold

    def track(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track keypoints from prev_img to curr_img using pyramidal KLT.

        Args:
            prev_img: previous frame, shape (H, W), dtype uint8.
            curr_img: current frame, shape (H, W), dtype uint8.
            prev_pts: keypoints in previous frame, shape (N, 2) [x, y], float32.

        Returns:
            curr_pts: tracked positions in current frame, shape (N, 2), float32.
            status: per-point success flag, shape (N,), uint8 (1=success, 0=lost).
        """
        if len(prev_pts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.uint8)

        N = len(prev_pts)
        prev_pts = prev_pts.astype(np.float32)

        prev_pyr = build_pyramid(prev_img, self.num_levels)
        curr_pyr = build_pyramid(curr_img, self.num_levels)

        # Accumulated flow (in current level's coordinate system)
        g = np.zeros((N, 2), dtype=np.float32)

        for level in range(self.num_levels - 1, -1, -1):
            scale = 2.0 ** level
            prev_l = prev_pts / scale
            init_curr_l = prev_l + g

            final_curr_l, valid = lk_at_level(
                prev_pyr[level], curr_pyr[level],
                prev_l, init_curr_l,
                self.half_win, self.max_iterations,
                self.epsilon, self.min_eig_threshold,
            )

            g = final_curr_l - prev_l          # flow at this level

            if level > 0:
                g = g * 2.0                    # upscale to next finer level

        curr_pts = prev_pts + g                # in original image coordinates
        status = valid.astype(np.uint8)

        # ------------------------------------------------------------------
        # Forward-backward consistency check
        # ------------------------------------------------------------------
        if self.fb_check:
            tracked = curr_pts[status == 1]
            if len(tracked) > 0:
                # Use only tracked subset for backward pass
                indices = np.where(status == 1)[0]
                g_back = np.zeros((len(tracked), 2), dtype=np.float32)

                for level in range(self.num_levels - 1, -1, -1):
                    scale = 2.0 ** level
                    curr_l = tracked / scale
                    init_prev_l = curr_l + g_back

                    final_prev_l, _ = lk_at_level(
                        curr_pyr[level], prev_pyr[level],
                        curr_l, init_prev_l,
                        self.half_win, self.max_iterations,
                        self.epsilon, self.min_eig_threshold,
                    )
                    g_back = final_prev_l - curr_l
                    if level > 0:
                        g_back = g_back * 2.0

                pts_back = tracked + g_back
                fb_err = np.linalg.norm(prev_pts[indices] - pts_back, axis=1)
                failed = fb_err > self.fb_threshold
                status[indices[failed]] = 0

        # ------------------------------------------------------------------
        # Boundary check
        # ------------------------------------------------------------------
        H, W = prev_img.shape
        out_of_bounds = (
            (curr_pts[:, 0] < 0) | (curr_pts[:, 0] >= W) |
            (curr_pts[:, 1] < 0) | (curr_pts[:, 1] >= H)
        )
        status[out_of_bounds] = 0

        n_tracked = int(status.sum())
        logger.debug("KLT: tracked %d / %d points", n_tracked, N)
        return curr_pts, status
