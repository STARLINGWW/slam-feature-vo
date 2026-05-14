"""Geometry helpers wrapping allowed OpenCV functions.

All functions use pixel-coordinate convention with an explicit camera matrix K.
Poses use T_cw (camera←world): p_cam = T_cw @ p_world_h.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def triangulate_points(
    T_cw0: np.ndarray,
    T_cw1: np.ndarray,
    K: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate 3-D world points from two calibrated views.

    Args:
        T_cw0: (4, 4) camera0←world pose.
        T_cw1: (4, 4) camera1←world pose.
        K: (3, 3) intrinsic matrix.
        pts0: (N, 2) pixel points in frame 0.
        pts1: (N, 2) pixel points in frame 1.

    Returns:
        pts3d: (N, 3) world-frame 3-D points.
        valid:  (N,) bool — positive depth in both cameras and w > eps.
    """
    P0 = (K @ T_cw0[:3]).astype(np.float32)   # (3, 4)
    P1 = (K @ T_cw1[:3]).astype(np.float32)

    pts4d = cv2.triangulatePoints(P0, P1,
                                  pts0.T.astype(np.float32),
                                  pts1.T.astype(np.float32))   # (4, N)
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-8
    pts3d = np.where(valid_w, pts4d[:3] / np.where(valid_w, w, 1.0), 0.0).T  # (N, 3)

    # Depth in each camera: z_cam = R[2] @ p_world + t[2]
    depth0 = pts3d @ T_cw0[2, :3] + T_cw0[2, 3]
    depth1 = pts3d @ T_cw1[2, :3] + T_cw1[2, 3]

    valid = valid_w & (depth0 > 0.0) & (depth1 > 0.0)
    return pts3d, valid


def reprojection_error(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    T_cw: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Compute per-point reprojection error in pixels.

    Args:
        pts3d: (N, 3) world points.
        pts2d: (N, 2) observed pixel points.
        T_cw: (4, 4) camera←world pose.
        K: (3, 3) intrinsics.

    Returns:
        errors: (N,) float32 reprojection distances in pixels.
    """
    N = len(pts3d)
    pts3d_h = np.hstack([pts3d, np.ones((N, 1))])       # (N, 4)
    p_cam = (T_cw[:3] @ pts3d_h.T)                      # (3, N)
    z = p_cam[2]
    valid = z > 1e-6
    p_img = np.where(valid, p_cam[:2] / np.where(valid, z, 1.0), 0.0)  # (2, N)
    p_px = K[:2, :2] @ p_img + K[:2, 2:3]               # (2, N)
    err = np.linalg.norm(p_px.T - pts2d, axis=1)
    err[~valid] = 1e6
    return err.astype(np.float32)


def solve_pnp(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    reproj_thresh: float = 2.0,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate camera pose from 3-D/2-D correspondences (PnP RANSAC).

    Args:
        pts3d: (N, 3) world points.
        pts2d: (N, 2) pixel observations (undistorted).
        K: (3, 3) intrinsics.
        reproj_thresh: RANSAC reprojection threshold in pixels.

    Returns:
        success: True if pose found with >= 4 inliers.
        T_cw: (4, 4) camera←world pose, or None.
        inliers: (M,) integer indices of inlier rows, or None.
    """
    if len(pts3d) < 4:
        return False, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64).reshape(-1, 1, 2),
        K, None,
        iterationsCount=100,
        reprojectionError=reproj_thresh,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok or inliers is None or len(inliers) < 4:
        return False, None, None

    R, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = tvec.ravel()
    return True, T_cw, inliers.ravel()


def essential_and_recover_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Compute essential matrix then recover R, t.

    Args:
        pts0: (N, 2) pixel points in frame 0.
        pts1: (N, 2) pixel points in frame 1.
        K: (3, 3) intrinsics.
        threshold: RANSAC pixel threshold.

    Returns:
        success: True if enough inliers.
        R: (3, 3) rotation.
        t: (3, 1) unit translation.
        mask: (N,) bool inlier mask (after essential + cheirality).
    """
    if len(pts0) < 8:
        return False, None, None, np.zeros(len(pts0), dtype=bool)

    E, e_mask = cv2.findEssentialMat(
        pts0, pts1, K,
        method=cv2.RANSAC, prob=0.999, threshold=threshold,
    )
    if E is None or E.shape != (3, 3):
        return False, None, None, np.zeros(len(pts0), dtype=bool)

    # recoverPose internally cheirality-checks and returns an updated mask
    n_inliers, R, t, p_mask = cv2.recoverPose(E, pts0, pts1, K, mask=e_mask.copy())

    combined = (p_mask.ravel() == 255)

    if n_inliers < 8 or combined.sum() < 8:
        return False, None, None, combined

    logger.debug("E-mat: %d/%d inliers", combined.sum(), len(pts0))
    return True, R, t, combined
