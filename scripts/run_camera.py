#!/usr/bin/env python3
"""Run monocular VO on a live camera (webcam or external USB/IP camera).

Usage examples:
    # Built-in webcam (index 0)
    python scripts/run_camera.py

    # External USB camera (index 1)
    python scripts/run_camera.py --camera 1

    # IP camera / RTSP stream
    python scripts/run_camera.py --camera "rtsp://user:pass@192.168.1.100:554/stream"

    # Video file (for offline testing without EuRoC)
    python scripts/run_camera.py --camera path/to/video.mp4

    # Override intrinsics (required for non-EuRoC cameras)
    python scripts/run_camera.py --camera 0 --fx 600 --fy 600 --cx 320 --cy 240

    # Use a custom YAML config
    python scripts/run_camera.py --config configs/webcam.yaml --camera 0

Press  q / ESC   to quit
Press  r         to reset the tracker
Press  s         to save current trajectory snapshot
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live monocular VO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",  default="configs/euroc.yaml",
                   help="YAML config (for VO / feature params)")
    p.add_argument("--camera",  default="0",
                   help="Camera index (int), RTSP/HTTP URL, or video file path")
    # Intrinsics override (takes priority over config file)
    p.add_argument("--fx",      type=float, default=None)
    p.add_argument("--fy",      type=float, default=None)
    p.add_argument("--cx",      type=float, default=None)
    p.add_argument("--cy",      type=float, default=None)
    p.add_argument("--width",   type=int,   default=None,
                   help="Desired capture width (resize if needed)")
    p.add_argument("--height",  type=int,   default=None,
                   help="Desired capture height (resize if needed)")
    p.add_argument("--k1",      type=float, default=0.0)
    p.add_argument("--k2",      type=float, default=0.0)
    p.add_argument("--p1",      type=float, default=0.0)
    p.add_argument("--p2",      type=float, default=0.0)
    p.add_argument("--no_display", action="store_true",
                   help="Disable OpenCV window (headless mode)")
    p.add_argument("--save_video", default=None,
                   help="Save annotated frames to this .mp4 file")
    p.add_argument("--out_dir", default="results/camera",
                   help="Directory for trajectory output")
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Camera source
# ─────────────────────────────────────────────────────────────────────────────

def open_capture(source: str) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture from an integer index, URL, or file path."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source!r}")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_intrinsics(args, cap: cv2.VideoCapture, cfg: dict):
    """Return CameraIntrinsics, preferring CLI args → config → auto-estimate."""
    from slam_vo.datasets.euroc import CameraIntrinsics

    # Actual capture resolution
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = args.width or cap_w
    target_h = args.height or cap_h

    cam_cfg = cfg.get("camera", {})

    def _get(arg_val, cfg_key, default):
        if arg_val is not None:
            return arg_val
        if cfg_key in cam_cfg:
            cfg_val = cam_cfg[cfg_key]
            # Scale if resolution differs
            if cfg_key == "fx" or cfg_key == "cx":
                scale = target_w / cam_cfg.get("width", target_w)
                return float(cfg_val) * scale
            if cfg_key == "fy" or cfg_key == "cy":
                scale = target_h / cam_cfg.get("height", target_h)
                return float(cfg_val) * scale
            return float(cfg_val)
        return default

    # Auto-estimate: assume ~60° horizontal FoV
    default_fx = target_w / (2.0 * np.tan(np.radians(30)))

    fx = _get(args.fx, "fx", default_fx)
    fy = _get(args.fy, "fy", default_fx)
    cx = _get(args.cx, "cx", target_w / 2.0)
    cy = _get(args.cy, "cy", target_h / 2.0)

    dist = cam_cfg.get("distortion", {})
    k1 = args.k1 or dist.get("k1", 0.0)
    k2 = args.k2 or dist.get("k2", 0.0)
    p1 = args.p1 or dist.get("p1", 0.0)
    p2 = args.p2 or dist.get("p2", 0.0)

    logger.info(
        "Camera intrinsics: %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        target_w, target_h, fx, fy, cx, cy,
    )

    return CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=target_w, height=target_h,
        k1=k1, k2=k2, p1=p1, p2=p2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(
    frame_bgr: np.ndarray,
    tracker,
    fps: float,
    n_poses: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # Tracked points
    if tracker._curr_pts2d is not None and len(tracker._curr_pts2d) > 0:
        for pt in tracker._curr_pts2d.astype(int):
            cv2.circle(vis, tuple(pt), 3, (0, 255, 0), -1)

    # KF seed points
    if tracker._kf_seeds is not None and len(tracker._kf_seeds) > 0:
        for pt in tracker._kf_seeds.astype(int):
            cv2.circle(vis, tuple(pt), 2, (255, 100, 0), -1)

    # HUD
    state_color = {
        "UNINITIALIZED": (0, 0, 200),
        "INITIALIZING":  (0, 180, 255),
        "TRACKING":      (0, 220, 0),
        "LOST":          (0, 0, 255),
    }.get(tracker.state, (200, 200, 200))

    lines = [
        f"State: {tracker.state}",
        f"Tracked: {tracker.n_tracked}  KF: {tracker.n_keyframes}",
        f"Poses: {n_poses}  FPS: {fps:.1f}",
        f"Map pts: {len(tracker.local_map)}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(vis, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)

    cv2.putText(vis, "q/ESC:quit  r:reset  s:save",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)
    return vis


def _draw_mini_map(
    poses: list,
    size: int = 250,
) -> np.ndarray:
    """Top-down XZ trajectory thumbnail."""
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    if len(poses) < 2:
        return canvas

    positions = np.array([T[:3, 3] for T in poses])
    xz = positions[:, [0, 2]]

    # Auto-scale to fit canvas
    mn, mx = xz.min(0), xz.max(0)
    span = (mx - mn).max()
    if span < 1e-3:
        return canvas
    scale = (size - 20) / span
    offset = (size / 2) - scale * (mn + mx) / 2

    pts_px = (xz * scale + offset).astype(int)

    for i in range(1, len(pts_px)):
        a = tuple(np.clip(pts_px[i - 1], 0, size - 1))
        b = tuple(np.clip(pts_px[i],     0, size - 1))
        cv2.line(canvas, a, b, (80, 200, 80), 1)

    start = tuple(np.clip(pts_px[0],  0, size - 1))
    end   = tuple(np.clip(pts_px[-1], 0, size - 1))
    cv2.circle(canvas, start, 4, (0, 120, 255), -1)
    cv2.circle(canvas, end,   4, (0, 255, 0),   -1)

    cv2.putText(canvas, "Top-down XZ", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Config
    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning("Config %s not found, using defaults", args.config)

    # Camera
    cap = open_capture(args.camera)
    camera = resolve_intrinsics(args, cap, cfg)

    # Set capture resolution if requested
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Tracker
    from slam_vo.vo.tracker import Tracker
    tracker = Tracker(camera, cfg)

    # Output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps_cap,
                                       (cap_w + 250, cap_h))
        logger.info("Saving video → %s", args.save_video)

    est_timestamps: list = []
    est_poses: list = []
    frame_times: list = []
    frame_idx: int = 0

    fps_display = 0.0
    fps_t0 = time.perf_counter()
    fps_count = 0

    logger.info("Starting live VO — press q/ESC to quit")

    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream / capture error")
                break

            ts = time.perf_counter()

            # Preprocess
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if camera.k1 != 0.0 or camera.k2 != 0.0:
                gray = cv2.undistort(gray, camera.K, camera.dist_coeffs)

            # Resize if needed
            if gray.shape[1] != camera.width or gray.shape[0] != camera.height:
                gray = cv2.resize(gray, (camera.width, camera.height))

            # Track
            t0 = time.perf_counter()
            pose = tracker.process_frame(gray, ts)
            dt_ms = (time.perf_counter() - t0) * 1e3
            frame_times.append(dt_ms)

            if pose is not None:
                est_timestamps.append(ts)
                est_poses.append(pose)

            frame_idx += 1

            # FPS counter
            fps_count += 1
            if time.perf_counter() - fps_t0 >= 1.0:
                fps_display = fps_count / (time.perf_counter() - fps_t0)
                fps_count = 0
                fps_t0 = time.perf_counter()

            # Display
            if not args.no_display or video_writer is not None:
                vis = _draw_overlay(frame, tracker, fps_display, len(est_poses))
                mini = _draw_mini_map(est_poses)

                # Composite: main frame + mini-map on the right
                h_vis = vis.shape[0]
                pad = np.full((h_vis - mini.shape[0], mini.shape[1], 3), 30, dtype=np.uint8)
                side = np.vstack([mini, pad])
                composite = np.hstack([vis, side])

                if not args.no_display:
                    cv2.imshow("Monocular VO", composite)

                if video_writer is not None:
                    video_writer.write(composite)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):   # q or ESC
                    logger.info("Quit requested")
                    break
                elif key == ord("r"):
                    logger.info("Resetting tracker")
                    tracker._reset()
                elif key == ord("s"):
                    _save_snapshot(tracker, est_poses, est_timestamps, out_dir, frame_idx)

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────────
    n_est = len(est_poses)
    avg_ms = float(np.mean(frame_times)) if frame_times else 0.0
    logger.info(
        "Finished: %d frames, %d poses (%.1f%%), avg=%.1f ms/frame (%.1f fps)",
        frame_idx, n_est,
        n_est / max(frame_idx, 1) * 100,
        avg_ms, 1000.0 / (avg_ms + 1e-9),
    )

    if n_est > 0:
        from slam_vo.utils.evaluation import TrajectoryEvaluator
        traj_path = out_dir / "trajectory_est.txt"
        TrajectoryEvaluator.save_tum(est_poses, est_timestamps, traj_path)
        logger.info("Trajectory saved → %s", traj_path)

        try:
            from slam_vo.utils.visualization import plot_trajectory
            plot_trajectory(
                est_poses,
                save_path=out_dir / "trajectory_plot.png",
                title="Live Camera VO",
            )
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    return 0


def _save_snapshot(tracker, est_poses, est_timestamps, out_dir, frame_idx):
    from slam_vo.utils.evaluation import TrajectoryEvaluator
    snap_path = out_dir / f"snapshot_f{frame_idx:05d}.txt"
    if est_poses:
        TrajectoryEvaluator.save_tum(est_poses, est_timestamps, snap_path)
        logger.info("Snapshot saved → %s  (%d poses)", snap_path, len(est_poses))
    else:
        logger.warning("No poses to save yet")


if __name__ == "__main__":
    sys.exit(main())
