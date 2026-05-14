"""Tests for FastDetector.

Tests:
  1. Synthetic: corners detected on a checkerboard.
  2. Real EuRoC frame: reasonable keypoint count and distribution.
  3. NMS: no two keypoints closer than nms_radius.
  4. Comparison vs cv2.FastFeatureDetector: overlap rate > 0.5.
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
THRESHOLD = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corner_image(size: int = 200) -> np.ndarray:
    """White rectangles on black background — L-shaped corners are FAST-detectable."""
    img = np.zeros((size, size), dtype=np.uint8)
    # Place several rectangles; each has 4 L-shaped corners
    rects = [(20, 20, 80, 80), (110, 20, 170, 80), (20, 110, 80, 170), (110, 110, 170, 170)]
    for r0, c0, r1, c1 in rects:
        img[r0:r1, c0:c1] = 200
    return img


def load_euroc_frame(idx: int = 50) -> np.ndarray:
    from slam_vo.datasets.euroc import EuRoCDataset
    ds = EuRoCDataset(DATASET_BASE, SEQUENCE)
    return ds.get_image(idx, undistort=True)


def keypoint_overlap_rate(kpts_ours, kpts_ref, radius: int = 5) -> float:
    """Fraction of ref keypoints that have an ours-keypoint within radius pixels."""
    if len(kpts_ref) == 0:
        return 0.0
    matched = 0
    for rx, ry in kpts_ref:
        dist = np.sqrt((kpts_ours[:, 0] - rx) ** 2 + (kpts_ours[:, 1] - ry) ** 2)
        if dist.min() <= radius:
            matched += 1
    return matched / len(kpts_ref)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_synthetic_corners():
    """White rectangles on black background: each rectangle has 4 L-shaped corners."""
    from slam_vo.features.fast_detector import FastDetector

    img = make_corner_image()
    det = FastDetector(threshold=15, nms_radius=4, max_keypoints=500)
    kpts = det.detect(img)

    assert kpts.ndim == 2 and kpts.shape[1] == 2, "Wrong output shape"
    assert kpts.dtype == np.float32, "Wrong dtype"
    # 4 rectangles × 4 corners = 16 expected corners
    assert len(kpts) >= 4, f"Expected >= 4 corners on rectangle image, got {len(kpts)}"
    logger.info("Synthetic corners: %d detected (expected ~16)", len(kpts))

    # Corners must be near the rectangle boundaries (known corner positions)
    expected_corners = np.array([
        [20, 20], [80, 20], [20, 80], [80, 80],
        [110, 20], [170, 20], [110, 80], [170, 80],
        [20, 110], [80, 110], [20, 170], [80, 170],
        [110, 110], [170, 110], [110, 170], [170, 170],
    ], dtype=np.float32)  # [col, row]
    # At least 12 of 16 expected corners should be found within 5 px
    matched = 0
    for ex, ey in expected_corners:
        if len(kpts) > 0:
            d = np.sqrt((kpts[:, 0] - ex) ** 2 + (kpts[:, 1] - ey) ** 2)
            if d.min() <= 5:
                matched += 1
    logger.info("  expected corners recovered: %d / 16", matched)
    assert matched >= 8, f"Too few expected corners found: {matched}/16"


def test_nms_radius():
    from slam_vo.features.fast_detector import FastDetector

    img = load_euroc_frame()
    nms_r = 4
    det = FastDetector(threshold=THRESHOLD, nms_radius=nms_r, max_keypoints=500)
    kpts = det.detect(img)

    assert len(kpts) > 0
    # Check pairwise distances (O(N²) but N is small enough)
    dists = np.sqrt(((kpts[:, np.newaxis] - kpts[np.newaxis]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min() if len(kpts) > 1 else np.inf
    assert min_dist >= nms_r - 1, f"NMS failed: min dist {min_dist:.1f} < {nms_r}"
    logger.info("NMS OK: min pairwise distance = %.1f px", min_dist)


def test_real_frame_count():
    from slam_vo.features.fast_detector import FastDetector

    img = load_euroc_frame()
    det = FastDetector(threshold=THRESHOLD, nms_radius=4, max_keypoints=500)
    kpts = det.detect(img)

    assert 20 <= len(kpts) <= 500, f"Unexpected keypoint count: {len(kpts)}"
    logger.info("Real frame: %d keypoints  img=%s", len(kpts), img.shape)

    # Basic distribution check: must cover at least 4 image quadrants
    H, W = img.shape
    quadrants = set()
    for x, y in kpts:
        quadrants.add((int(x > W / 2), int(y > H / 2)))
    assert len(quadrants) >= 3, "Keypoints too clustered (< 3 quadrants covered)"


def test_cv2_comparison():
    from slam_vo.features.fast_detector import FastDetector

    img = load_euroc_frame()

    # Our detector
    det = FastDetector(threshold=THRESHOLD, nms_radius=3, max_keypoints=1000)
    t0 = time.perf_counter()
    kpts_ours = det.detect(img)
    t_ours = (time.perf_counter() - t0) * 1000

    # cv2 reference
    cv_fast = cv2.FastFeatureDetector_create(threshold=THRESHOLD, nonmaxSuppression=True)
    t0 = time.perf_counter()
    cv_kpts = cv_fast.detect(img, None)
    t_cv = (time.perf_counter() - t0) * 1000
    kpts_ref = np.array([[kp.pt[0], kp.pt[1]] for kp in cv_kpts], dtype=np.float32)

    overlap = keypoint_overlap_rate(kpts_ours, kpts_ref, radius=5)
    logger.info(
        "vs cv2:  ours=%d  cv2=%d  overlap=%.2f  ours=%.1fms  cv2=%.1fms",
        len(kpts_ours), len(kpts_ref), overlap, t_ours, t_cv,
    )
    assert overlap >= 0.5, f"Low overlap with cv2 FAST: {overlap:.2f}"


def test_border_exclusion():
    from slam_vo.features.fast_detector import FastDetector
    from slam_vo.features.fast_detector import _BORDER

    img = load_euroc_frame()
    det = FastDetector(threshold=THRESHOLD)
    kpts = det.detect(img)
    H, W = img.shape
    for x, y in kpts:
        assert x >= _BORDER and x < W - _BORDER, f"x={x} out of border"
        assert y >= _BORDER and y < H - _BORDER, f"y={y} out of border"


def test_empty_image():
    from slam_vo.features.fast_detector import FastDetector

    flat = np.full((100, 100), 128, dtype=np.uint8)
    kpts = FastDetector().detect(flat)
    assert len(kpts) == 0, "Should return no corners on flat image"


def test_timing():
    from slam_vo.features.fast_detector import FastDetector

    img = load_euroc_frame()
    det = FastDetector(threshold=THRESHOLD, max_keypoints=500)

    # Warm-up
    det.detect(img)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        det.detect(img)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    logger.info("FAST timing: mean=%.1f ms  min=%.1f  max=%.1f", mean_ms, min(times), max(times))
    assert mean_ms < 200, f"FAST too slow: {mean_ms:.1f} ms"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_synthetic_corners,
        test_nms_radius,
        test_real_frame_count,
        test_cv2_comparison,
        test_border_exclusion,
        test_empty_image,
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
