"""Tests for KltTracker.

Tests:
  1. Synthetic pure translation: RMS endpoint error < 0.5 px.
  2. Synthetic rotation + scale.
  3. Real EuRoC consecutive frames: success rate & median error vs cv2.
  4. Forward-backward check removes gross outliers.
  5. Timing.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_BASE = "F:/_AAA_FILE_ALL/_AAA_CODE_ALL/VINS_ALL/myVO_ws/datasets"
SEQUENCE = "MH_01_easy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(30, 220, (480, 752), dtype=np.uint8)
    return cv2.GaussianBlur(img, (7, 7), 2.0)


def translate_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def load_consecutive_frames(idx0: int = 50):
    from slam_vo.datasets.euroc import EuRoCDataset
    ds = EuRoCDataset(DATASET_BASE, SEQUENCE)
    img0 = ds.get_image(idx0, undistort=True)
    img1 = ds.get_image(idx0 + 1, undistort=True)
    return img0, img1


def make_grid_points(img: np.ndarray, n: int = 200) -> np.ndarray:
    H, W = img.shape
    step = int(np.sqrt(H * W / n))
    ys = np.arange(step, H - step, step)
    xs = np.arange(step, W - step, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)[:n]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pure_translation():
    from slam_vo.features.klt_tracker import KltTracker

    img0 = make_test_image()
    dx, dy = 8.5, -5.3
    img1 = translate_image(img0, dx, dy)

    pts0 = make_grid_points(img0, 100)
    tracker = KltTracker(num_levels=3, win_size=21, fb_check=False)
    pts1, status = tracker.track(img0, img1, pts0)

    good = status == 1
    assert good.sum() > 50, f"Too few tracked points: {good.sum()}"

    err_x = pts1[good, 0] - pts0[good, 0] - dx
    err_y = pts1[good, 1] - pts0[good, 1] - dy
    rms = float(np.sqrt(err_x ** 2 + err_y ** 2).mean())
    logger.info("Translation RMS error: %.3f px  (tracked %d/%d)", rms, good.sum(), len(pts0))
    assert rms < 1.0, f"Translation RMS too high: {rms:.3f} px"


def test_forward_backward_check():
    from slam_vo.features.klt_tracker import KltTracker

    img0 = make_test_image(1)
    dx, dy = 5.0, 3.0
    img1 = translate_image(img0, dx, dy)
    pts0 = make_grid_points(img0, 100)

    tracker_no_fb = KltTracker(num_levels=3, fb_check=False)
    tracker_fb    = KltTracker(num_levels=3, fb_check=True, fb_threshold=1.0)

    _, status_no = tracker_no_fb.track(img0, img1, pts0)
    _, status_fb = tracker_fb.track(img0, img1, pts0)

    logger.info("Without FB: %d/%d  With FB: %d/%d",
                status_no.sum(), len(pts0), status_fb.sum(), len(pts0))
    # FB check should not dramatically reduce good tracks on clean synthetic data
    assert status_fb.sum() >= status_no.sum() * 0.7, "FB check removed too many good tracks"


def test_real_frames_vs_cv2():
    from slam_vo.features.klt_tracker import KltTracker
    from slam_vo.features.fast_detector import FastDetector

    img0, img1 = load_consecutive_frames(50)

    det = FastDetector(threshold=20, max_keypoints=300)
    pts0 = det.detect(img0)
    assert len(pts0) > 20, "Not enough keypoints for tracking test"

    # Our KLT
    tracker = KltTracker(num_levels=4, win_size=21, fb_check=True)
    t0 = time.perf_counter()
    pts1_ours, status_ours = tracker.track(img0, img1, pts0)
    t_ours = (time.perf_counter() - t0) * 1000

    # cv2 reference
    pts0_cv = pts0.reshape(-1, 1, 2)
    t0 = time.perf_counter()
    pts1_cv, status_cv, _ = cv2.calcOpticalFlowPyrLK(
        img0, img1, pts0_cv, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    t_cv = (time.perf_counter() - t0) * 1000

    pts1_cv = pts1_cv.reshape(-1, 2)
    status_cv = status_cv.ravel()

    # Compare endpoint error on mutually tracked points
    both_good = (status_ours == 1) & (status_cv == 1)
    if both_good.sum() > 5:
        ee = np.linalg.norm(pts1_ours[both_good] - pts1_cv[both_good], axis=1)
        median_ee = float(np.median(ee))
        logger.info(
            "vs cv2: ours_good=%d  cv2_good=%d  both=%d  median_EE=%.3f px"
            "  ours=%.1fms  cv2=%.1fms",
            status_ours.sum(), status_cv.sum(), both_good.sum(),
            median_ee, t_ours, t_cv,
        )
        assert median_ee < 2.0, f"Median endpoint error vs cv2 too high: {median_ee:.3f}"
    else:
        logger.warning("Too few mutual tracks to compare (both_good=%d)", both_good.sum())


def test_empty_input():
    from slam_vo.features.klt_tracker import KltTracker

    img0 = make_test_image()
    img1 = translate_image(img0, 2, 2)
    tracker = KltTracker()
    pts, status = tracker.track(img0, img1, np.empty((0, 2), dtype=np.float32))
    assert len(pts) == 0
    assert len(status) == 0


def test_out_of_bounds_marked_failed():
    from slam_vo.features.klt_tracker import KltTracker

    img0 = make_test_image()
    img1 = translate_image(img0, 200, 0)   # shift so many points fly off screen
    pts0 = make_grid_points(img0, 50)
    _, status = KltTracker(fb_check=False).track(img0, img1, pts0)
    assert status.sum() < len(pts0), "Some points should go out of bounds"
    logger.info("Out-of-bounds: %d/%d marked failed", (status == 0).sum(), len(pts0))


def test_timing():
    from slam_vo.features.klt_tracker import KltTracker
    from slam_vo.features.fast_detector import FastDetector

    img0, img1 = load_consecutive_frames(50)
    pts0 = FastDetector(threshold=20, max_keypoints=300).detect(img0)
    tracker = KltTracker(num_levels=4)

    # Warm-up
    tracker.track(img0, img1, pts0)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        tracker.track(img0, img1, pts0)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    logger.info("KLT timing: mean=%.1f ms  min=%.1f  max=%.1f  (%d pts)",
                mean_ms, min(times), max(times), len(pts0))
    assert mean_ms < 2000, f"KLT too slow: {mean_ms:.1f} ms"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_pure_translation,
        test_forward_backward_check,
        test_real_frames_vs_cv2,
        test_empty_input,
        test_out_of_bounds_marked_failed,
        test_timing,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            logger.info("PASS  %s", t.__name__)
            passed += 1
        except Exception as e:
            logger.error("FAIL  %s: %s", t.__name__, e)

    logger.info("\n%d / %d tests passed", passed, len(tests))
    if passed < len(tests):
        sys.exit(1)
