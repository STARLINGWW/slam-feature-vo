#!/usr/bin/env python3
"""Run monocular VO on EuRoC MAV dataset.

Usage:
    conda activate slam_vo
    python scripts/run_vo.py --seq MH_01_easy [--max_frames 1000] [--config configs/euroc.yaml]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monocular VO on EuRoC")
    p.add_argument("--config", default="configs/euroc.yaml")
    p.add_argument("--seq", default=None, help="Override sequence name")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_eval", action="store_true", help="Skip ATE evaluation")
    p.add_argument("--no_plot", action="store_true", help="Skip trajectory plot")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_camera(cfg: dict):
    from slam_vo.datasets.euroc import CameraIntrinsics
    cam_cfg = cfg["camera"]
    dist = cam_cfg.get("distortion", {})
    return CameraIntrinsics(
        fx=cam_cfg["fx"], fy=cam_cfg["fy"],
        cx=cam_cfg["cx"], cy=cam_cfg["cy"],
        width=cam_cfg["width"], height=cam_cfg["height"],
        k1=dist.get("k1", 0.0), k2=dist.get("k2", 0.0),
        p1=dist.get("p1", 0.0), p2=dist.get("p2", 0.0),
    )


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)
    seq = args.seq or cfg["dataset"]["sequences"][0]
    cfg["dataset"]["sequences"] = [seq]

    out_dir = Path("results/euroc") / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    from slam_vo.datasets.euroc import EuRoCDataset
    ds = EuRoCDataset(
        cfg["dataset"]["base_path"],
        seq,
        cfg["dataset"].get("camera", "cam0"),
    )
    logger.info("Dataset: %s  frames=%d  GT_poses=%d", seq, len(ds), len(ds.gt_poses))

    # Camera & tracker
    camera = build_camera(cfg)
    from slam_vo.vo.tracker import Tracker
    tracker = Tracker(camera, cfg)

    # Run
    max_frames = args.max_frames or len(ds)
    est_timestamps: list = []
    est_poses: list = []
    frame_times: list = []

    for i, (frame_meta, img, _) in enumerate(ds.iter_frames_with_gt(undistort=True)):
        if i >= max_frames:
            break

        t0 = time.perf_counter()
        pose = tracker.process_frame(img, frame_meta.timestamp_s)
        frame_times.append((time.perf_counter() - t0) * 1000)

        if pose is not None:
            est_timestamps.append(frame_meta.timestamp_s)
            est_poses.append(pose)

        if i % 200 == 0:
            logger.info(
                "Frame %4d/%d  state=%-15s  tracked=%3d  kf=%d  fps=%.1f",
                i, min(max_frames, len(ds)),
                tracker.state, tracker.n_tracked, tracker.n_keyframes,
                1000.0 / (np.mean(frame_times[-50:]) + 1e-9),
            )

    n_est = len(est_poses)
    coverage = n_est / min(max_frames, len(ds)) * 100
    logger.info(
        "Done: %d frames processed, %d poses estimated (%.1f%%)  "
        "avg=%.1f ms/frame",
        min(max_frames, len(ds)), n_est, coverage, np.mean(frame_times),
    )

    if n_est == 0:
        logger.error("No poses estimated — check init_min_parallax and data path")
        return 1

    # Save TUM trajectory
    from slam_vo.utils.evaluation import TrajectoryEvaluator
    traj_path = out_dir / "trajectory_est.txt"
    TrajectoryEvaluator.save_tum(est_poses, est_timestamps, traj_path)

    # --- Evaluation ---
    if not args.no_eval and len(ds.gt_poses) > 0:
        # Match each estimated timestamp to nearest GT pose
        gt_ts = np.array([p.timestamp_ns for p in ds.gt_poses], dtype=np.int64)
        paired_est, paired_gt = [], []
        for ts, T_est in zip(est_timestamps, est_poses):
            ts_ns = int(ts * 1e9)
            idx = int(np.argmin(np.abs(gt_ts - ts_ns)))
            if abs(int(gt_ts[idx]) - ts_ns) < 50_000_000:   # 50 ms tolerance
                paired_est.append(T_est)
                paired_gt.append(ds.gt_poses[idx].pose_matrix)

        if len(paired_est) >= 10:
            evaluator = TrajectoryEvaluator(align=cfg.get("eval", {}).get("align", "sim3"))
            metrics = evaluator.evaluate(paired_est, paired_gt, sequence=seq)

            # Add timing + coverage stats
            metrics.num_frames_total = min(max_frames, len(ds))
            metrics.num_frames_tracked = n_est
            metrics.tracking_success_rate = coverage / 100.0
            metrics.avg_total_ms = float(np.mean(frame_times))
            metrics.fps = 1000.0 / (metrics.avg_total_ms + 1e-9)

            evaluator.save_metrics(metrics, out_dir / "metrics.json")

            # Save GT TUM (at matched timestamps)
            gt_ts_s = [p.timestamp_ns * 1e-9 for p in ds.gt_poses
                       if any(abs(p.timestamp_ns - int(ts * 1e9)) < 50_000_000
                              for ts in est_timestamps)]
            if paired_gt:
                gt_timestamps = [p.timestamp_ns * 1e-9 for p in ds.gt_poses
                                  if int(p.timestamp_ns * 1e-9 * 1e9) in
                                  {int(ts * 1e9) for ts in est_timestamps}]
                # Simpler: just save paired GT
                TrajectoryEvaluator.save_tum(
                    paired_gt,
                    [est_timestamps[i] for i in range(len(paired_est))],
                    out_dir / "trajectory_gt_matched.txt",
                )
        else:
            logger.warning("Not enough GT matches for evaluation (%d)", len(paired_est))

    # --- Plot ---
    if not args.no_plot:
        try:
            from slam_vo.utils.visualization import plot_trajectory
            gt_T_wc = [p.pose_matrix for p in ds.gt_poses]
            plot_trajectory(
                est_poses, gt_T_wc,
                save_path=out_dir / "trajectory_plot.png",
                title=f"VO Trajectory — {seq}",
            )
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    logger.info("Results saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
