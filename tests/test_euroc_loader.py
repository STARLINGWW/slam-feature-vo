"""Smoke-test for EuRoC data loader on MH_01_easy."""

import logging
import sys
from pathlib import Path

# Make sure the package is importable when run directly from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH = "F:/_AAA_FILE_ALL/_AAA_CODE_ALL/VINS_ALL/myVO_ws/datasets"
SEQUENCE = "MH_01_easy"


def test_loader():
    from slam_vo.datasets.euroc import EuRoCDataset

    ds = EuRoCDataset(BASE_PATH, SEQUENCE)

    # Basic counts
    assert len(ds) > 0, "No frames loaded"
    assert len(ds.gt_poses) > 0, "No GT poses loaded"
    logger.info("Frames: %d   GT poses: %d", len(ds), len(ds.gt_poses))

    # Camera intrinsics
    K = ds.camera.K
    assert K.shape == (3, 3)
    logger.info("Camera K:\n%s", K)

    # Load first image (raw)
    frame0 = ds[0]
    img_raw = frame0.load(undistort=False)
    assert img_raw.ndim == 2, "Expected grayscale"
    assert img_raw.dtype == np.uint8
    logger.info("Frame 0: ts=%d  img shape=%s  path=%s",
                frame0.timestamp_ns, img_raw.shape, frame0.image_path.name)

    # Load with undistort
    img_undist = frame0.load(undistort=True, camera=ds.camera)
    assert img_undist.shape == img_raw.shape
    logger.info("Undistorted image OK, shape=%s", img_undist.shape)

    # GT lookup
    gt0 = ds.gt_poses[0]
    T = gt0.pose_matrix
    assert T.shape == (4, 4)
    logger.info("GT[0] position: %s  ts=%d ns", gt0.position, gt0.timestamp_ns)

    # iter_frames_with_gt (first 5)
    matches = 0
    for i, (frame, img, gt) in enumerate(ds.iter_frames_with_gt()):
        if i >= 5:
            break
        if gt is not None:
            matches += 1
    logger.info("GT matches in first 5 frames: %d/5", matches)

    # Verify timestamp ordering
    ts_imgs = [f.timestamp_ns for f in ds.frames]
    assert ts_imgs == sorted(ts_imgs), "Image timestamps not sorted"
    ts_gts = [p.timestamp_ns for p in ds.gt_poses]
    assert ts_gts == sorted(ts_gts), "GT timestamps not sorted"
    logger.info("Timestamp ordering OK")

    logger.info("ALL TESTS PASSED")


if __name__ == "__main__":
    test_loader()
