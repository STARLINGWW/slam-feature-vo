#!/usr/bin/env python3
"""Evaluate a saved TUM trajectory against EuRoC ground truth.

Usage:
    python scripts/evaluate.py --est results/euroc/MH_01_easy/trajectory_est.txt
                               --gt  results/euroc/MH_01_easy/trajectory_gt_matched.txt
                               [--align sim3]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_tum(path: str):
    """Load TUM-format trajectory.  Returns (timestamps, poses)."""
    timestamps, poses = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            ts = float(vals[0])
            tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
            qx, qy, qz, qw = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])

            # Quaternion → rotation matrix
            q = np.array([qw, qx, qy, qz])
            w, x, y, z = q
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            timestamps.append(ts)
            poses.append(T)
    return np.array(timestamps), poses


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--est", required=True, help="Estimated TUM trajectory")
    p.add_argument("--gt",  required=True, help="Ground-truth TUM trajectory")
    p.add_argument("--align", default="sim3", choices=["sim3", "se3", "none"])
    p.add_argument("--out", default=None, help="Save metrics.json here")
    args = p.parse_args()

    ts_est, poses_est = load_tum(args.est)
    ts_gt,  poses_gt  = load_tum(args.gt)

    if len(poses_est) == 0:
        logger.error("Empty estimated trajectory")
        return 1

    # Nearest-timestamp matching if lengths differ
    if len(poses_est) != len(poses_gt):
        logger.info("Aligning by nearest timestamp (%d est, %d gt)", len(poses_est), len(poses_gt))
        matched_gt = []
        for ts in ts_est:
            idx = int(np.argmin(np.abs(ts_gt - ts)))
            matched_gt.append(poses_gt[idx])
        poses_gt_matched = matched_gt
    else:
        poses_gt_matched = poses_gt

    from slam_vo.utils.evaluation import TrajectoryEvaluator
    evaluator = TrajectoryEvaluator(align=args.align)
    metrics = evaluator.evaluate(
        poses_est, poses_gt_matched,
        sequence=Path(args.est).parent.name,
    )

    out_path = args.out or str(Path(args.est).parent / "metrics_eval.json")
    evaluator.save_metrics(metrics, Path(out_path))
    logger.info("Metrics saved → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
