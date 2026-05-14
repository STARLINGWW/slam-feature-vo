"""Trajectory evaluation: ATE, RPE, and metrics.json output."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryMetrics:
    sequence: str = ""
    timestamp: str = ""

    # ATE
    ate_rmse_m: float = 0.0
    ate_mean_m: float = 0.0
    ate_median_m: float = 0.0
    ate_std_m: float = 0.0
    ate_max_m: float = 0.0

    # RPE
    rpe_trans_rmse_m: float = 0.0
    rpe_rot_rmse_deg: float = 0.0

    # Tracking
    num_frames_total: int = 0
    num_frames_tracked: int = 0
    tracking_success_rate: float = 0.0

    # Feature stats
    avg_keypoints_per_frame: float = 0.0
    avg_tracked_points: float = 0.0
    avg_inlier_ratio: float = 0.0
    feature_distribution_score: float = 0.0

    # Timing
    avg_total_ms: float = 0.0
    avg_detection_ms: float = 0.0
    avg_tracking_ms: float = 0.0
    avg_matching_ms: float = 0.0
    avg_pose_estimation_ms: float = 0.0
    fps: float = 0.0

    # System
    peak_memory_mb: float = 0.0
    gpu_used: bool = False


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------


def _umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity transform: dst ≈ scale * R @ src + t.

    Args:
        src: (N, 3) source positions.
        dst: (N, 3) target positions.
        with_scale: if False, fixes scale=1 (SE3 alignment).

    Returns:
        scale, R (3x3), t (3,)
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U @ Vt)
    S = np.eye(3)
    S[2, 2] = det_sign

    R = U @ S @ Vt
    scale = (np.trace(np.diag(D) @ S) / var_src) if with_scale else 1.0
    t = mu_dst - scale * R @ mu_src
    return float(scale), R, t


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class TrajectoryEvaluator:
    """Compute ATE and RPE between estimated and ground-truth trajectories."""

    def __init__(self, align: str = "sim3"):
        """Args:
            align: alignment method — 'sim3' | 'se3' | 'none'.
        """
        assert align in ("sim3", "se3", "none"), f"Unknown align mode: {align}"
        self.align = align

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        est_poses: List[np.ndarray],
        gt_poses: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
        sequence: str = "",
    ) -> TrajectoryMetrics:
        """Compute ATE and RPE.

        Args:
            est_poses: list of 4x4 SE3 matrices (estimated).
            gt_poses:  list of 4x4 SE3 matrices (ground-truth), same length.
            timestamps: optional per-frame timestamps (seconds).
            sequence: dataset sequence name for logging.

        Returns:
            TrajectoryMetrics dataclass.
        """
        assert len(est_poses) == len(gt_poses), "Length mismatch"
        n = len(est_poses)

        est_t = np.stack([T[:3, 3] for T in est_poses])   # (N, 3)
        gt_t = np.stack([T[:3, 3] for T in gt_poses])

        # Alignment
        est_t_aligned = self._align(est_t, gt_t)

        # ATE
        errors = np.linalg.norm(est_t_aligned - gt_t, axis=1)
        ate_rmse = float(np.sqrt((errors ** 2).mean()))
        ate_mean = float(errors.mean())
        ate_median = float(np.median(errors))
        ate_std = float(errors.std())
        ate_max = float(errors.max())

        # RPE (delta=1 frame)
        rpe_trans, rpe_rot = self._compute_rpe(est_poses, gt_poses)

        metrics = TrajectoryMetrics(
            sequence=sequence,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            ate_rmse_m=ate_rmse,
            ate_mean_m=ate_mean,
            ate_median_m=ate_median,
            ate_std_m=ate_std,
            ate_max_m=ate_max,
            rpe_trans_rmse_m=float(np.sqrt((rpe_trans ** 2).mean())),
            rpe_rot_rmse_deg=float(
                np.degrees(np.sqrt((rpe_rot ** 2).mean()))
            ),
            num_frames_total=n,
            num_frames_tracked=n,
            tracking_success_rate=1.0,
        )

        logger.info(
            "[%s] ATE RMSE=%.4f m  RPE_t=%.4f m  RPE_r=%.3f deg",
            sequence or "eval",
            ate_rmse,
            metrics.rpe_trans_rmse_m,
            metrics.rpe_rot_rmse_deg,
        )
        return metrics

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_tum(
        poses: List[np.ndarray],
        timestamps: List[float],
        path: Path,
    ) -> None:
        """Save trajectory in TUM format: timestamp tx ty tz qx qy qz qw."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for ts, T in zip(timestamps, poses):
                t = T[:3, 3]
                q = _rot_to_quat(T[:3, :3])  # xyzw
                f.write(
                    f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )
        logger.info("Saved TUM trajectory to %s", path)

    @staticmethod
    def save_metrics(metrics: TrajectoryMetrics, path: Path) -> None:
        """Save metrics to a structured metrics.json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(metrics)
        # Restructure into the canonical nested format from CLAUDE.md
        out = {
            "sequence": d["sequence"],
            "timestamp": d["timestamp"],
            "trajectory": {
                "ate_rmse_m": d["ate_rmse_m"],
                "ate_mean_m": d["ate_mean_m"],
                "ate_median_m": d["ate_median_m"],
                "ate_std_m": d["ate_std_m"],
                "ate_max_m": d["ate_max_m"],
                "rpe_trans_rmse_m": d["rpe_trans_rmse_m"],
                "rpe_rot_rmse_deg": d["rpe_rot_rmse_deg"],
                "num_frames_total": d["num_frames_total"],
                "num_frames_tracked": d["num_frames_tracked"],
                "tracking_success_rate": d["tracking_success_rate"],
            },
            "features": {
                "avg_keypoints_per_frame": d["avg_keypoints_per_frame"],
                "avg_tracked_points": d["avg_tracked_points"],
                "avg_inlier_ratio": d["avg_inlier_ratio"],
                "feature_distribution_score": d["feature_distribution_score"],
            },
            "timing": {
                "avg_total_ms": d["avg_total_ms"],
                "avg_detection_ms": d["avg_detection_ms"],
                "avg_tracking_ms": d["avg_tracking_ms"],
                "avg_matching_ms": d["avg_matching_ms"],
                "avg_pose_estimation_ms": d["avg_pose_estimation_ms"],
                "fps": d["fps"],
            },
            "system": {
                "peak_memory_mb": d["peak_memory_mb"],
                "gpu_used": d["gpu_used"],
            },
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        logger.info("Saved metrics to %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align(self, est: np.ndarray, gt: np.ndarray) -> np.ndarray:
        if self.align == "none":
            return est
        with_scale = self.align == "sim3"
        scale, R, t = _umeyama_alignment(est, gt, with_scale=with_scale)
        return (scale * (R @ est.T)).T + t

    @staticmethod
    def _compute_rpe(
        est: List[np.ndarray], gt: List[np.ndarray], delta: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-frame relative pose errors.

        Returns:
            rpe_trans: translation error magnitudes, shape (N-delta,).
            rpe_rot:   rotation error in radians, shape (N-delta,).
        """
        n = len(est)
        trans_errors = []
        rot_errors = []
        for i in range(n - delta):
            Q_rel = np.linalg.inv(gt[i]) @ gt[i + delta]
            P_rel = np.linalg.inv(est[i]) @ est[i + delta]
            E = np.linalg.inv(Q_rel) @ P_rel
            trans_errors.append(np.linalg.norm(E[:3, 3]))
            # rotation angle from trace(R)
            cos_angle = (np.trace(E[:3, :3]) - 1.0) / 2.0
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            rot_errors.append(float(np.arccos(cos_angle)))
        return np.array(trans_errors), np.array(rot_errors)


# ---------------------------------------------------------------------------
# Quaternion helper (R → xyzw)
# ---------------------------------------------------------------------------


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)
