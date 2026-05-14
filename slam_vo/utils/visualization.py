"""Trajectory and feature visualization helpers."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def plot_trajectory(
    est_poses: List[np.ndarray],
    gt_poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[Path] = None,
    title: str = "Trajectory",
) -> None:
    """Plot top-down (XZ) trajectory.

    Args:
        est_poses: list of 4×4 T_wc matrices (estimated).
        gt_poses:  list of 4×4 T_wc matrices (ground-truth), optional.
        save_path: save PNG to this path if provided.
        title: plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping trajectory plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    est_pos = np.array([T[:3, 3] for T in est_poses])
    ax.plot(est_pos[:, 0], est_pos[:, 2], "b-", linewidth=1.0, label="Estimated")
    ax.plot(est_pos[0, 0], est_pos[0, 2], "bo", markersize=6)

    if gt_poses:
        gt_pos = np.array([T[:3, 3] for T in gt_poses])
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], "r--", linewidth=1.0, label="Ground Truth")
        ax.plot(gt_pos[0, 0], gt_pos[0, 2], "ro", markersize=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory plot → %s", save_path)
    plt.close(fig)


def draw_tracks(
    img: np.ndarray,
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    """Draw optical flow tracks on an image (for debugging).

    Args:
        img: (H, W) or (H, W, 3) image.
        prev_pts: (N, 2) previous points.
        curr_pts: (N, 2) current points.
        status: (N,) 1=tracked, 0=lost.

    Returns:
        vis: (H, W, 3) uint8 BGR image with tracks drawn.
    """
    try:
        import cv2
    except ImportError:
        return img

    import cv2
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    for i, (p0, p1, s) in enumerate(zip(prev_pts, curr_pts, status)):
        color = (0, 255, 0) if s else (0, 0, 255)
        cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), color, 1)
        cv2.circle(vis, tuple(p1.astype(int)), 2, color, -1)

    return vis
