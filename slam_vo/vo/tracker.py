"""Monocular VO tracker.

State machine:
  UNINITIALIZED → INITIALIZING → TRACKING → (LOST → UNINITIALIZED)

Initialization (two-frame):
  FAST detect → KLT track → Essential matrix → recoverPose → triangulate → map

Continuous tracking:
  KLT track active map points → PnP RANSAC → update pose

Keyframe management:
  When tracked-ratio drops below threshold or distance threshold exceeded:
    triangulate pending seeds from last KF → new map points
    detect new seeds in current frame
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from slam_vo.datasets.euroc import CameraIntrinsics
from slam_vo.features.fast_detector import FastDetector
from slam_vo.features.klt_tracker import KltTracker
from slam_vo.utils.geometry import (
    essential_and_recover_pose,
    reprojection_error,
    solve_pnp,
    triangulate_points,
)
from slam_vo.vo.local_map import LocalMap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_near_existing(
    new_kpts: np.ndarray,
    existing: np.ndarray,
    min_dist: float = 20.0,
) -> np.ndarray:
    """Remove keypoints that are too close to any existing tracked point."""
    if len(new_kpts) == 0:
        return new_kpts
    if len(existing) == 0:
        return new_kpts
    diffs = new_kpts[:, np.newaxis, :] - existing[np.newaxis, :, :]  # (M, N, 2)
    min_dists = np.linalg.norm(diffs, axis=2).min(axis=1)            # (M,)
    return new_kpts[min_dists >= min_dist]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    """Monocular visual odometry front-end.

    Args:
        camera: CameraIntrinsics.
        cfg:    full YAML config dict (keys: 'fast', 'klt', 'vo').
    """

    _STATES = ("UNINITIALIZED", "INITIALIZING", "TRACKING", "LOST")

    def __init__(self, camera: CameraIntrinsics, cfg: dict):
        self.camera = camera
        self.K = camera.K.astype(np.float64)
        self._cfg_vo = cfg.get("vo", {})

        # Feature modules
        fc = cfg.get("fast", {})
        self.detector = FastDetector(
            threshold=fc.get("threshold", 20),
            nms_radius=fc.get("nms_radius", 4),
            max_keypoints=fc.get("max_keypoints", 500),
        )
        kc = cfg.get("klt", {})
        self.klt = KltTracker(
            num_levels=kc.get("num_levels", 4),
            win_size=kc.get("win_size", 21),
            max_iterations=kc.get("max_iterations", 30),
            epsilon=kc.get("epsilon", 0.01),
            fb_check=True,
        )

        # VO params
        self._min_tracked: int = int(self._cfg_vo.get("min_tracked_points", 40))
        self._kf_ratio: float = float(self._cfg_vo.get("keyframe_min_tracked_ratio", 0.65))
        self._kf_min_frames: int = 3
        self._kf_min_dist: float = float(self._cfg_vo.get("keyframe_min_distance", 0.05))
        self._init_parallax_px: float = float(
            self._cfg_vo.get("init_min_parallax", 2.0)
        ) / 180.0 * np.pi * camera.fx
        self._reproj_thresh: float = 2.5
        self._min_3d_pts: int = 20

        # State
        self.state: str = "UNINITIALIZED"
        self.frame_count: int = 0
        self.n_tracked: int = 0
        self.n_keyframes: int = 0

        # Previous frame image (for KLT)
        self._prev_img: Optional[np.ndarray] = None

        # Init state
        self._init_img: Optional[np.ndarray] = None
        self._init_pts: Optional[np.ndarray] = None
        self._init_frames: int = 0

        # Active tracking state
        self._curr_pts2d: Optional[np.ndarray] = None  # (N, 2) in current frame
        self._curr_mp_ids: Optional[np.ndarray] = None  # (N,) int

        # Keyframe state (for triangulating new seeds)
        self._kf_img: Optional[np.ndarray] = None
        self._kf_T_cw: Optional[np.ndarray] = None
        self._kf_seeds: Optional[np.ndarray] = None  # (M, 2) pending features
        self._frames_since_kf: int = 0
        self._kf_n_tracked: int = 0

        # Map
        self.local_map = LocalMap()

        # Trajectory: list of (timestamp, T_wc 4×4)
        self.trajectory: List[Tuple[float, np.ndarray]] = []
        self.current_pose: Optional[np.ndarray] = None  # T_wc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, img: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Process one greyscale frame and return estimated T_wc (or None).

        Args:
            img: (H, W) uint8 greyscale, already undistorted.
            timestamp: frame capture time in seconds.

        Returns:
            T_wc (4×4) if pose is known, else None.
        """
        self.frame_count += 1

        if self.state == "UNINITIALIZED":
            self._on_uninitialized(img)

        elif self.state == "INITIALIZING":
            self._on_initializing(img, timestamp)

        elif self.state == "TRACKING":
            self._on_tracking(img, timestamp)

        elif self.state == "LOST":
            logger.info("Frame %d: LOST → resetting", self.frame_count)
            self._reset()
            self._on_uninitialized(img)

        self._prev_img = img
        return self.current_pose

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _on_uninitialized(self, img: np.ndarray) -> None:
        """Detect features in the first frame."""
        kpts = self.detector.detect(img)
        if len(kpts) < 50:
            logger.debug("Init: too few keypoints (%d)", len(kpts))
            return
        self._init_img = img.copy()
        self._init_pts = kpts.copy()
        self._init_frames = 0
        self.state = "INITIALIZING"
        logger.debug("Frame %d: init frame stored (%d kpts)", self.frame_count, len(kpts))

    def _on_initializing(self, img: np.ndarray, timestamp: float) -> None:
        """Track from init frame; try to bootstrap when parallax is sufficient."""
        self._init_frames += 1
        if self._init_frames > 200:     # give up after 200 frames
            logger.info("Init: timeout, resetting")
            self._reset()
            return

        # KLT from stored init frame
        pts1, status = self.klt.track(self._init_img, img, self._init_pts)
        good = status == 1
        if good.sum() < 30:
            logger.debug("Init: KLT lost (%d remain), resetting", good.sum())
            self._reset()
            return

        pts0_g = self._init_pts[good]
        pts1_g = pts1[good]

        # Parallax check
        flow = np.linalg.norm(pts1_g - pts0_g, axis=1)
        median_flow = float(np.median(flow))
        if median_flow < self._init_parallax_px:
            return   # keep waiting for more motion

        # Essential matrix + relative pose
        ok, R, t, mask = essential_and_recover_pose(pts0_g, pts1_g, self.K)
        if not ok:
            logger.debug("Init: E-mat failed, continuing to wait")
            return

        pts0_tri = pts0_g[mask]
        pts1_tri = pts1_g[mask]

        T_cw0 = np.eye(4)
        T_cw1 = np.eye(4)
        T_cw1[:3, :3] = R
        T_cw1[:3, 3] = t.ravel()

        # Triangulate
        pts3d, valid = triangulate_points(T_cw0, T_cw1, self.K, pts0_tri, pts1_tri)

        # Reprojection filter
        err1 = reprojection_error(pts3d, pts1_tri, T_cw1, self.K)
        valid &= err1 < self._reproj_thresh

        # Depth spread check: avoid near-degenerate scenes
        depths = pts3d[valid, 2] if valid.sum() > 0 else np.array([])
        if valid.sum() < self._min_3d_pts or (len(depths) > 0 and depths.max() > 200 * depths.min()):
            logger.debug("Init: %d valid 3D pts (need %d), waiting", valid.sum(), self._min_3d_pts)
            return

        pts3d_v = pts3d[valid]
        pts2d_v = pts1_tri[valid]   # 2D coords in current frame

        # Normalise scale: median depth in frame 0 → 10 units
        med_depth = float(np.median(pts3d_v[:, 2]))
        if med_depth <= 0:
            return
        scale = 10.0 / med_depth
        pts3d_v = pts3d_v * scale
        T_cw1[:3, 3] = T_cw1[:3, 3] * scale

        # Register map points
        mp_ids = []
        for p3d in pts3d_v:
            mp = self.local_map.add_map_point(p3d)
            mp_ids.append(mp.point_id)

        self._curr_pts2d = pts2d_v.copy()
        self._curr_mp_ids = np.array(mp_ids, dtype=np.int64)
        self.current_pose = np.linalg.inv(T_cw1)
        self.trajectory.append((timestamp, self.current_pose.copy()))

        # First keyframe = current frame
        self._kf_img = img.copy()
        self._kf_T_cw = T_cw1.copy()
        self._kf_seeds = _filter_near_existing(
            self.detector.detect(img), self._curr_pts2d
        )
        self._frames_since_kf = 0
        self._kf_n_tracked = len(self._curr_pts2d)
        self.n_keyframes = 1

        self.state = "TRACKING"
        logger.info(
            "Frame %d: Initialized  3D_pts=%d  baseline=%.3f  scale=%.2f  flow=%.1fpx",
            self.frame_count, len(pts3d_v),
            float(np.linalg.norm(T_cw1[:3, 3])), scale, median_flow,
        )

    def _on_tracking(self, img: np.ndarray, timestamp: float) -> None:
        """KLT + PnP per-frame pose update."""
        assert self._curr_pts2d is not None

        # 1. KLT track map-associated points
        pts_new, status = self.klt.track(self._prev_img, img, self._curr_pts2d)
        good = status == 1
        n_tracked = int(good.sum())

        if n_tracked < self._min_tracked:
            logger.info("Frame %d: LOST (%d tracked)", self.frame_count, n_tracked)
            self.state = "LOST"
            return

        pts2d_t = pts_new[good]
        mp_ids_t = self._curr_mp_ids[good]

        # 2. Fetch 3D positions
        pts3d_t, valid3d = self.local_map.get_positions(mp_ids_t)
        if valid3d.sum() < self._min_tracked:
            self.state = "LOST"
            return

        pts2d_pnp = pts2d_t[valid3d]
        mp_ids_pnp = mp_ids_t[valid3d]
        pts3d_pnp = pts3d_t[valid3d]

        # 3. PnP
        ok, T_cw, inliers = solve_pnp(pts3d_pnp, pts2d_pnp, self.K, self._reproj_thresh)
        if not ok or len(inliers) < 6:
            logger.info("Frame %d: PnP failed (%s inliers)", self.frame_count,
                        len(inliers) if inliers is not None else 0)
            self.state = "LOST"
            return

        # 4. Keep inliers only
        self._curr_pts2d = pts2d_pnp[inliers]
        self._curr_mp_ids = mp_ids_pnp[inliers]
        self.n_tracked = len(inliers)

        # 5. Update pose
        T_wc = np.linalg.inv(T_cw)
        self.current_pose = T_wc
        self.trajectory.append((timestamp, T_wc.copy()))

        self._frames_since_kf += 1

        # 6. Keyframe decision
        if self._should_insert_keyframe(T_cw):
            self._insert_keyframe(img, T_cw)

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def _should_insert_keyframe(self, T_cw: np.ndarray) -> bool:
        if self._frames_since_kf < self._kf_min_frames:
            return False

        # Tracked-ratio criterion
        tracked_ratio = self.n_tracked / max(self._kf_n_tracked, 1)
        if tracked_ratio < self._kf_ratio:
            return True

        # Translation criterion (only if we have a KF pose)
        if self._kf_T_cw is not None:
            T_rel = T_cw @ np.linalg.inv(self._kf_T_cw)
            dist = float(np.linalg.norm(T_rel[:3, 3]))
            if dist > self._kf_min_dist:
                return True

        return False

    def _insert_keyframe(self, img: np.ndarray, T_cw: np.ndarray) -> None:
        """Triangulate pending seeds from last KF; detect new seeds."""
        n_new = 0

        # 1. Triangulate seeds from last keyframe → current frame
        if (self._kf_seeds is not None and len(self._kf_seeds) >= 5
                and self._kf_T_cw is not None and self._kf_img is not None):
            seeds_curr, s_status = self.klt.track(self._kf_img, img, self._kf_seeds)
            s_good = s_status == 1

            if s_good.sum() >= 5:
                pts3d, valid = triangulate_points(
                    self._kf_T_cw, T_cw, self.K,
                    self._kf_seeds[s_good], seeds_curr[s_good],
                )
                err = reprojection_error(pts3d, seeds_curr[s_good], T_cw, self.K)
                valid &= err < self._reproj_thresh

                new_pts2d = []
                new_ids = []
                for i, (is_v, p3d, p2d) in enumerate(
                    zip(valid, pts3d, seeds_curr[s_good])
                ):
                    if is_v:
                        mp = self.local_map.add_map_point(p3d)
                        new_pts2d.append(p2d)
                        new_ids.append(mp.point_id)
                        n_new += 1

                if new_pts2d:
                    self._curr_pts2d = np.vstack(
                        [self._curr_pts2d, np.array(new_pts2d, dtype=np.float32)]
                    )
                    self._curr_mp_ids = np.concatenate(
                        [self._curr_mp_ids, np.array(new_ids, dtype=np.int64)]
                    )

        # 2. Detect new seeds in current frame
        all_pts = self._curr_pts2d if self._curr_pts2d is not None else np.empty((0, 2))
        new_kpts = self.detector.detect(img)
        new_seeds = _filter_near_existing(new_kpts, all_pts)

        # 3. Cull old map points
        self.local_map.cull_to_max()

        # 4. Update KF state
        self._kf_img = img.copy()
        self._kf_T_cw = T_cw.copy()
        self._kf_seeds = new_seeds
        self._frames_since_kf = 0
        self._kf_n_tracked = len(self._curr_pts2d)
        self.n_keyframes += 1

        logger.info(
            "KF #%d  new_3D=%d  total_tracked=%d  map_size=%d  seeds=%d",
            self.n_keyframes, n_new, len(self._curr_pts2d),
            len(self.local_map), len(new_seeds),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self.state = "UNINITIALIZED"
        self._init_img = None
        self._init_pts = None
        self._curr_pts2d = None
        self._curr_mp_ids = None
        self._kf_img = None
        self._kf_T_cw = None
        self._kf_seeds = None
        self._frames_since_kf = 0
        self._kf_n_tracked = 0
        # Keep the map and trajectory — don't discard previous work
        logger.debug("Tracker reset to UNINITIALIZED")
