#!/usr/bin/env python3
"""VO pipeline integration tests.

Tests cover:
  1. Synthetic two-frame initialisation (known R, t)
  2. Full tracker state machine on synthetic random-walk data
  3. Comparison between hand-written KLT and cv2.calcOpticalFlowPyrLK
     on the same EuRoC-style scenario
  4. (Optional) ATE on MH_01_easy — skipped if dataset absent
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam_vo.datasets.euroc import CameraIntrinsics
from slam_vo.utils.geometry import (
    essential_and_recover_pose,
    reprojection_error,
    solve_pnp,
    triangulate_points,
)
from slam_vo.vo.local_map import LocalMap
from slam_vo.vo.tracker import Tracker

# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

EUROC_CAM = CameraIntrinsics(
    fx=458.654, fy=457.296, cx=367.215, cy=248.375,
    width=752, height=480,
    k1=-0.28340811, k2=0.07395907, p1=0.00019359, p2=1.76187114e-05,
)

MINIMAL_CFG = {
    "fast": {"threshold": 20, "nms_radius": 4, "max_keypoints": 300},
    "klt":  {"num_levels": 3, "win_size": 15, "max_iterations": 20, "epsilon": 0.01},
    "vo":   {
        "min_tracked_points": 20,
        "keyframe_min_tracked_ratio": 0.6,
        "keyframe_min_distance": 0.05,
        "init_min_parallax": 1.5,
    },
}


def _random_scene(n_pts=80, seed=0):
    """Return (K, T_cw0, T_cw1, pts3d_w, pts2d_0, pts2d_1)."""
    rng = np.random.default_rng(seed)
    K = EUROC_CAM.K.astype(np.float64)

    # Random 3-D points in front of camera 0
    pts3d = rng.uniform([-2, -1.5, 3], [2, 1.5, 8], size=(n_pts, 3))

    T_cw0 = np.eye(4)
    R1, _ = cv2.Rodrigues(np.array([0.05, 0.1, 0.02]))
    T_cw1 = np.eye(4)
    T_cw1[:3, :3] = R1
    T_cw1[:3, 3] = [0.3, 0.05, 0.0]

    def project(T, pts):
        p = (T[:3, :3] @ pts.T + T[:3, 3:4])
        p_img = K @ p
        return (p_img[:2] / p_img[2]).T

    pts2d_0 = project(T_cw0, pts3d)
    pts2d_1 = project(T_cw1, pts3d)
    return K, T_cw0, T_cw1, pts3d, pts2d_0.astype(np.float32), pts2d_1.astype(np.float32)


def _synthetic_frame(size=(480, 752), n_blobs=120, seed=0):
    """Synthetic textured image with random gaussian blobs."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 128, dtype=np.uint8)
    for _ in range(n_blobs):
        cx = int(rng.integers(10, size[1] - 10))
        cy = int(rng.integers(10, size[0] - 10))
        r = int(rng.integers(3, 15))
        val = int(rng.integers(0, 255))
        cv2.circle(img, (cx, cy), r, val, -1)
    return img


def _warp_image(img, dx=20.0, dy=5.0, angle_deg=2.0):
    """Translate + rotate an image to simulate camera motion (affine, for KLT tests)."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _perspective_warp_scene(size=(480, 752), n_blobs=180, d_plane=5.0, seed=0,
                             rvec=None, tvec=None):
    """Two images of a planar 3D scene under a proper camera homography.

    Returns (img0, img1, H) where H maps img0→img1 exactly, so KLT
    correspondences satisfy epipolar geometry.
    """
    rng = np.random.default_rng(seed)
    K = EUROC_CAM.K.astype(np.float64)

    img0 = np.full(size, 50, dtype=np.uint8)
    for _ in range(n_blobs):
        cx_ = int(rng.integers(15, size[1] - 15))
        cy_ = int(rng.integers(15, size[0] - 15))
        r_ = int(rng.integers(4, 16))
        val_ = int(rng.integers(20, 235))
        cv2.circle(img0, (cx_, cy_), r_, val_, -1)
    img0 = cv2.GaussianBlur(img0, (3, 3), 1.0)

    if rvec is None:
        rvec = np.array([0.02, 0.05, 0.01])
    if tvec is None:
        tvec = np.array([0.12, 0.02, 0.0])

    R1, _ = cv2.Rodrigues(rvec)
    n_plane = np.array([0.0, 0.0, 1.0])

    # Planar homography: H = K (R - t n^T / d) K^-1
    H = K @ (R1 - np.outer(tvec, n_plane) / d_plane) @ np.linalg.inv(K)
    H /= H[2, 2]

    img1 = cv2.warpPerspective(img0, H, (size[1], size[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    return img0, img1.astype(np.uint8)


# ─────────────────────────────────────────────────────
# 1. Geometry utilities
# ─────────────────────────────────────────────────────

class TestGeometry:
    def test_triangulate_recovers_3d(self):
        K, T_cw0, T_cw1, pts3d_gt, pts2d_0, pts2d_1 = _random_scene(60)
        pts3d_rec, valid = triangulate_points(T_cw0, T_cw1, K, pts2d_0, pts2d_1)
        assert valid.sum() >= 55, f"Only {valid.sum()} valid triangulations"
        err = np.linalg.norm(pts3d_rec[valid] - pts3d_gt[valid], axis=1)
        assert err.mean() < 0.05, f"Mean 3D error {err.mean():.4f} too large"

    def test_reprojection_error_perfect(self):
        K, T_cw0, _, pts3d_gt, pts2d_0, _ = _random_scene(40)
        errs = reprojection_error(pts3d_gt, pts2d_0, T_cw0, K)
        assert errs.max() < 0.5, f"Max reprojection {errs.max():.3f}px on perfect data"

    def test_solve_pnp_recovers_pose(self):
        K, T_cw0, T_cw1, pts3d, _, pts2d_1 = _random_scene(50)
        ok, T_est, inliers = solve_pnp(pts3d, pts2d_1, K, reproj_thresh=2.0)
        assert ok, "PnP failed"
        assert len(inliers) >= 40
        # Translation error in absolute units
        t_err = np.linalg.norm(T_est[:3, 3] - T_cw1[:3, 3])
        assert t_err < 0.05, f"PnP translation error {t_err:.4f}"

    def test_essential_and_recover_pose(self):
        # Use a scene with clear depth variation and larger baseline for stable E-mat.
        # Subpixel noise is added so RANSAC thresholding is well-conditioned.
        rng = np.random.default_rng(42)
        K = EUROC_CAM.K.astype(np.float64)
        pts3d = rng.uniform([-3, -2, 4], [3, 2, 10], size=(100, 3))

        T_cw0 = np.eye(4)
        R1, _ = cv2.Rodrigues(np.array([0.08, 0.15, 0.03]))
        T_cw1 = np.eye(4)
        T_cw1[:3, :3] = R1
        T_cw1[:3, 3] = [0.5, 0.05, 0.0]

        def project(T, pts):
            p = (T[:3, :3] @ pts.T + T[:3, 3:4])
            p_img = K @ p
            return (p_img[:2] / p_img[2]).T.astype(np.float32)

        pts2d_0 = project(T_cw0, pts3d) + rng.normal(0, 0.3, (100, 2)).astype(np.float32)
        pts2d_1 = project(T_cw1, pts3d) + rng.normal(0, 0.3, (100, 2)).astype(np.float32)

        ok, R, t, mask = essential_and_recover_pose(pts2d_0, pts2d_1, K)
        assert ok, "essential_and_recover_pose failed on a well-conditioned scene"
        assert mask.sum() >= 50


# ─────────────────────────────────────────────────────
# 2. Local map
# ─────────────────────────────────────────────────────

class TestLocalMap:
    def test_add_and_retrieve(self):
        lm = LocalMap()
        p = np.array([1.0, 2.0, 3.0])
        mp = lm.add_map_point(p)
        assert mp.point_id == 0
        pts, valid = lm.get_positions(np.array([0], dtype=np.int64))
        assert valid[0]
        np.testing.assert_allclose(pts[0], p)

    def test_cull_to_max(self):
        lm = LocalMap(max_points=10)
        for i in range(25):
            lm.add_map_point(np.array([float(i), 0.0, 5.0]))
        lm.cull_to_max()
        assert len(lm) == 10

    def test_bad_point_excluded(self):
        lm = LocalMap()
        mp = lm.add_map_point(np.array([0.0, 0.0, 1.0]))
        mp.is_bad = True
        _, valid = lm.get_positions(np.array([mp.point_id], dtype=np.int64))
        assert not valid[0]


# ─────────────────────────────────────────────────────
# 3. Tracker state machine
# ─────────────────────────────────────────────────────

class TestTrackerStateMachine:
    """Tracker tests use a perspective-warped planar scene so that KLT
    correspondences satisfy epipolar geometry (essential matrix can be computed)."""

    def test_initialises_from_synthetic(self):
        tracker = Tracker(EUROC_CAM, MINIMAL_CFG)
        assert tracker.state == "UNINITIALIZED"

        # Build a sequence of perspective-warped frames with increasing baseline
        base_img, _ = _perspective_warp_scene(seed=1)
        K_inv = np.linalg.inv(EUROC_CAM.K.astype(np.float64))
        n_plane = np.array([0.0, 0.0, 1.0])
        d_plane = 5.0

        # Feed first frame
        tracker.process_frame(base_img, 0.0)
        assert tracker.state == "INITIALIZING", "Should enter INITIALIZING after first frame"

        initialized = False
        for i in range(1, 60):
            # Grow baseline gradually; need ~12px median flow for parallax threshold
            rvec_i = np.array([0.002 * i, 0.003 * i, 0.0])
            tvec_i = np.array([0.015 * i, 0.002 * i, 0.0])
            R_i, _ = cv2.Rodrigues(rvec_i)
            H_i = EUROC_CAM.K.astype(np.float64) @ (
                R_i - np.outer(tvec_i, n_plane) / d_plane
            ) @ K_inv
            H_i /= H_i[2, 2]
            img_i = cv2.warpPerspective(base_img, H_i, (752, 480),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            tracker.process_frame(img_i, float(i) * 0.05)
            if tracker.state == "TRACKING":
                initialized = True
                break

        assert initialized, "Tracker never left INITIALIZING state"
        assert tracker.current_pose is not None
        assert tracker.current_pose.shape == (4, 4)

    def test_tracking_persists_over_frames(self):
        tracker = Tracker(EUROC_CAM, MINIMAL_CFG)

        base_img, _ = _perspective_warp_scene(seed=7)
        K_inv = np.linalg.inv(EUROC_CAM.K.astype(np.float64))
        n_plane = np.array([0.0, 0.0, 1.0])
        d_plane = 5.0

        def make_frame(scale):
            rvec_i = np.array([0.003 * scale, 0.004 * scale, 0.001 * scale])
            tvec_i = np.array([0.018 * scale, 0.003 * scale, 0.0])
            R_i, _ = cv2.Rodrigues(rvec_i)
            H_i = EUROC_CAM.K.astype(np.float64) @ (
                R_i - np.outer(tvec_i, n_plane) / d_plane
            ) @ K_inv
            H_i /= H_i[2, 2]
            return cv2.warpPerspective(base_img, H_i, (752, 480),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

        # Initialise
        for i in range(50):
            tracker.process_frame(make_frame(i), float(i) * 0.05)
            if tracker.state == "TRACKING":
                break

        assert tracker.state == "TRACKING", "Did not initialise after 50 frames"

        # Continue tracking with slow incremental motion
        lost_count = 0
        base_i = 50
        for j in range(30):
            tracker.process_frame(make_frame(base_i + j), float(base_i + j) * 0.05)
            if tracker.state == "LOST":
                lost_count += 1

        assert lost_count <= 5, f"Tracker lost too many times ({lost_count})"

    def test_keyframe_insertion(self):
        tracker = Tracker(EUROC_CAM, MINIMAL_CFG)

        base_img, _ = _perspective_warp_scene(seed=3)
        K_inv = np.linalg.inv(EUROC_CAM.K.astype(np.float64))
        n_plane = np.array([0.0, 0.0, 1.0])
        d_plane = 5.0

        for i in range(120):
            rvec_i = np.array([0.003 * i, 0.004 * i, 0.001 * i])
            tvec_i = np.array([0.018 * i, 0.003 * i, 0.0])
            R_i, _ = cv2.Rodrigues(rvec_i)
            H_i = EUROC_CAM.K.astype(np.float64) @ (
                R_i - np.outer(tvec_i, n_plane) / d_plane
            ) @ K_inv
            H_i /= H_i[2, 2]
            img_i = cv2.warpPerspective(base_img, H_i, (752, 480),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            tracker.process_frame(img_i, float(i) * 0.05)

        assert tracker.n_keyframes >= 1, "No keyframe inserted after 120 frames"


# ─────────────────────────────────────────────────────
# 4. KLT: custom vs cv2 comparison
# ─────────────────────────────────────────────────────

class TestKltVsOpenCV:
    """Compare hand-written pyramidal KLT to cv2.calcOpticalFlowPyrLK."""

    def _run_cv2_klt(self, img0, img1, pts0, num_levels=3, win_size=15):
        lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=num_levels - 1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        pts_cv, status, _ = cv2.calcOpticalFlowPyrLK(
            img0, img1, pts0.astype(np.float32).reshape(-1, 1, 2), None, **lk_params
        )
        return pts_cv.reshape(-1, 2), status.ravel()

    def test_endpoint_agreement(self):
        from slam_vo.features.klt_tracker import KltTracker

        img0 = _synthetic_frame(size=(480, 752), n_blobs=200, seed=10)
        dx, dy = 12.0, 6.0
        img1 = _warp_image(img0, dx=dx, dy=dy, angle_deg=0.0)

        # Seed points in a grid
        pts0 = np.array([[x, y] for y in range(40, 450, 40)
                         for x in range(40, 720, 40)], dtype=np.float32)

        klt = KltTracker(num_levels=3, win_size=15, max_iterations=20,
                         epsilon=0.01, fb_check=False)
        pts_custom, status_custom = klt.track(img0, img1, pts0)
        pts_cv2, status_cv2 = self._run_cv2_klt(img0, img1, pts0,
                                                  num_levels=3, win_size=15)

        # Only compare points both methods tracked
        both_ok = (status_custom == 1) & (status_cv2 == 1)
        assert both_ok.sum() >= 40, f"Too few mutually-tracked points: {both_ok.sum()}"

        diff = np.linalg.norm(pts_custom[both_ok] - pts_cv2[both_ok], axis=1)
        mean_err = diff.mean()
        max_err = diff.max()

        print(f"\nKLT vs cv2 (n={both_ok.sum()}): "
              f"mean_err={mean_err:.3f}px  max_err={max_err:.3f}px")

        assert mean_err < 2.0, f"Mean endpoint deviation {mean_err:.3f}px (threshold 2px)"
        assert max_err < 6.0, f"Max endpoint deviation {max_err:.3f}px (threshold 6px)"

    def test_tracking_rate_comparable(self):
        from slam_vo.features.klt_tracker import KltTracker

        img0 = _synthetic_frame(size=(480, 752), n_blobs=200, seed=11)
        img1 = _warp_image(img0, dx=8.0, dy=4.0, angle_deg=0.5)
        pts0 = np.array([[x, y] for y in range(40, 450, 40)
                         for x in range(40, 720, 40)], dtype=np.float32)

        klt = KltTracker(num_levels=3, win_size=15, max_iterations=20,
                         epsilon=0.01, fb_check=False)
        _, status_custom = klt.track(img0, img1, pts0)
        _, status_cv2 = self._run_cv2_klt(img0, img1, pts0)

        rate_custom = status_custom.mean()
        rate_cv2 = status_cv2.mean()
        print(f"\nTracking rate — custom: {rate_custom:.2%}  cv2: {rate_cv2:.2%}")

        # Custom tracking rate should be within 15% of cv2
        assert abs(rate_custom - rate_cv2) < 0.15, \
            f"Tracking rate gap too large: custom={rate_custom:.2%} cv2={rate_cv2:.2%}"


# ─────────────────────────────────────────────────────
# 5. Optional: ATE on MH_01_easy
# ─────────────────────────────────────────────────────

EUROC_BASE = Path("F:/_AAA_FILE_ALL/_AAA_CODE_ALL/VINS_ALL/myVO_ws/datasets")
DATASET_AVAILABLE = (EUROC_BASE / "MH_01_easy_ov").exists() or \
                    (EUROC_BASE / "MH_01_easy").exists()


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="EuRoC MH_01_easy not found")
class TestVoOnEuRoC:
    def test_ate_on_mh01(self):
        """Run 500-frame VO on MH_01_easy and check ATE < 0.5 m (sanity bound)."""
        import time
        import yaml
        from slam_vo.datasets.euroc import EuRoCDataset
        from slam_vo.utils.evaluation import TrajectoryEvaluator

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "euroc.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        ds = EuRoCDataset(str(EUROC_BASE), "MH_01_easy",
                          cfg["dataset"].get("camera", "cam0"))
        tracker = Tracker(EUROC_CAM, cfg)

        est_timestamps, est_poses = [], []
        frame_times = []
        max_frames = 500

        for i, (frame_meta, img, _) in enumerate(ds.iter_frames_with_gt(undistort=True)):
            if i >= max_frames:
                break
            t0 = time.perf_counter()
            pose = tracker.process_frame(img, frame_meta.timestamp_s)
            frame_times.append((time.perf_counter() - t0) * 1e3)
            if pose is not None:
                est_timestamps.append(frame_meta.timestamp_s)
                est_poses.append(pose)

        n_est = len(est_poses)
        coverage = n_est / max_frames
        avg_ms = float(np.mean(frame_times)) if frame_times else 0.0
        print(f"\nEuRoC MH_01: {n_est}/{max_frames} poses ({coverage:.1%}), "
              f"avg={avg_ms:.1f}ms/frame")

        assert coverage >= 0.5, f"Only {coverage:.1%} frames tracked (need ≥50%)"

        # Match to GT
        gt_ts = np.array([p.timestamp_ns for p in ds.gt_poses], dtype=np.int64)
        paired_est, paired_gt = [], []
        for ts, T_est in zip(est_timestamps, est_poses):
            ts_ns = int(ts * 1e9)
            idx = int(np.argmin(np.abs(gt_ts - ts_ns)))
            if abs(int(gt_ts[idx]) - ts_ns) < 50_000_000:
                paired_est.append(T_est)
                paired_gt.append(ds.gt_poses[idx].pose_matrix)

        assert len(paired_est) >= 50, f"Too few GT matches ({len(paired_est)})"

        evaluator = TrajectoryEvaluator(align="sim3")
        metrics = evaluator.evaluate(paired_est, paired_gt, sequence="MH_01_easy")

        print(f"ATE RMSE = {metrics.ate_rmse:.4f} m  "
              f"ATE mean = {metrics.ate_mean:.4f} m")

        # Sanity bound: any working VO should be well under 0.5m on 500 frames
        assert metrics.ate_rmse < 0.5, f"ATE RMSE {metrics.ate_rmse:.4f} m > 0.5 m"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
