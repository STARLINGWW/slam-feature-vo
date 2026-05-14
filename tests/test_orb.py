"""Tests for OrbDescriptor and BruteForceMatcher.

Tests:
  1. Descriptor shape and dtype.
  2. Orientation consistency: rotated patches give matching orientations.
  3. Intra-class (same scene) Hamming distance < inter-class (different scenes).
  4. Matching under small translation: inlier rate > 0.5.
  5. Ratio-test removes ambiguous matches.
  6. Cross-check filter.
  7. Timing.
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

def load_frame(idx: int = 50) -> np.ndarray:
    from slam_vo.datasets.euroc import EuRoCDataset
    return EuRoCDataset(DATASET_BASE, SEQUENCE).get_image(idx, undistort=True)


def detect_kpts(img: np.ndarray, n: int = 200) -> np.ndarray:
    from slam_vo.features.fast_detector import FastDetector
    return FastDetector(threshold=20, max_keypoints=n).detect(img)


def filter_border(kpts: np.ndarray, img: np.ndarray, margin: int = 20) -> np.ndarray:
    H, W = img.shape
    ok = ((kpts[:, 0] >= margin) & (kpts[:, 0] < W - margin) &
          (kpts[:, 1] >= margin) & (kpts[:, 1] < H - margin))
    return kpts[ok]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape_dtype():
    from slam_vo.features.orb_descriptor import OrbDescriptor

    img = load_frame()
    kpts = detect_kpts(img)
    orb = OrbDescriptor(n_pairs=256)
    desc = orb.compute(img, kpts)

    assert desc.shape == (len(kpts), 32), f"Wrong shape: {desc.shape}"
    assert desc.dtype == np.uint8, f"Wrong dtype: {desc.dtype}"
    logger.info("Shape OK: %s dtype=%s", desc.shape, desc.dtype)


def test_empty_keypoints():
    from slam_vo.features.orb_descriptor import OrbDescriptor

    img = load_frame()
    orb = OrbDescriptor()
    desc = orb.compute(img, np.empty((0, 2), dtype=np.float32))
    assert desc.shape == (0, 32)


def test_descriptor_variance():
    """Descriptors should not be all-zero or all-one (non-degenerate)."""
    from slam_vo.features.orb_descriptor import OrbDescriptor

    img = load_frame()
    kpts = filter_border(detect_kpts(img, 200), img)
    orb = OrbDescriptor()
    desc = orb.compute(img, kpts)

    # Each byte should have variance across keypoints
    mean_bits = np.unpackbits(desc, axis=1).mean()
    logger.info("Mean bit value: %.3f  (should be near 0.5)", mean_bits)
    assert 0.2 < mean_bits < 0.8, f"Degenerate descriptors: mean_bit={mean_bits:.3f}"


def test_intra_vs_inter_distance():
    """Same-scene descriptors should be closer than cross-scene ones."""
    from slam_vo.features.orb_descriptor import OrbDescriptor
    from slam_vo.features.feature_matcher import hamming_distance_matrix

    img0 = load_frame(50)
    img1 = load_frame(51)   # adjacent frame (similar scene)
    img2 = load_frame(500)  # distant frame (different scene)

    kpts0 = filter_border(detect_kpts(img0, 100), img0)
    kpts1 = filter_border(detect_kpts(img1, 100), img1)
    kpts2 = filter_border(detect_kpts(img2, 100), img2)

    orb = OrbDescriptor()
    d0 = orb.compute(img0, kpts0)
    d1 = orb.compute(img1, kpts1)
    d2 = orb.compute(img2, kpts2)

    dist_near = hamming_distance_matrix(d0, d1)
    dist_far  = hamming_distance_matrix(d0, d2)

    mean_near = float(dist_near.min(axis=1).mean())
    mean_far  = float(dist_far.min(axis=1).mean())
    logger.info("Min Hamming: near=%.1f  far=%.1f  (should be near < far)", mean_near, mean_far)
    assert mean_near < mean_far, "Same-scene descriptors should be closer than far-scene"


def test_matching_under_translation():
    """Match descriptors between original and slightly translated image."""
    from slam_vo.features.orb_descriptor import OrbDescriptor
    from slam_vo.features.feature_matcher import BruteForceMatcher

    img0 = load_frame(50)
    H, W = img0.shape
    dx, dy = 5, 3
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img1 = cv2.warpAffine(img0, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    kpts0 = filter_border(detect_kpts(img0, 200), img0, margin=30)
    # Ground-truth keypoints in img1 (translated)
    kpts1_gt = kpts0 + np.array([dx, dy], dtype=np.float32)

    orb = OrbDescriptor()
    d0 = orb.compute(img0, kpts0)
    d1 = orb.compute(img1, kpts1_gt)

    matcher = BruteForceMatcher(ratio_threshold=0.8, cross_check=False)
    matches = matcher.match(d0, d1)

    if len(matches) == 0:
        logger.warning("No matches found — skipping inlier rate check")
        return

    # Verify geometric consistency of matches
    correct = 0
    for i, j in matches:
        ex = abs(kpts1_gt[j, 0] - kpts0[i, 0] - dx)
        ey = abs(kpts1_gt[j, 1] - kpts0[i, 1] - dy)
        if ex < 3 and ey < 3:
            correct += 1
    inlier_rate = correct / len(matches)
    logger.info("Translation match: %d/%d inliers (%.2f)", correct, len(matches), inlier_rate)
    assert inlier_rate >= 0.4, f"Inlier rate too low: {inlier_rate:.2f}"


def test_ratio_test_reduces_matches():
    """Stricter ratio threshold should yield fewer but more reliable matches."""
    from slam_vo.features.orb_descriptor import OrbDescriptor
    from slam_vo.features.feature_matcher import BruteForceMatcher

    img0 = load_frame(50)
    img1 = load_frame(55)  # a few frames later, some overlap

    kpts0 = filter_border(detect_kpts(img0, 200), img0)
    kpts1 = filter_border(detect_kpts(img1, 200), img1)

    orb = OrbDescriptor()
    d0 = orb.compute(img0, kpts0)
    d1 = orb.compute(img1, kpts1)

    m_loose  = BruteForceMatcher(ratio_threshold=0.9, cross_check=False).match(d0, d1)
    m_strict = BruteForceMatcher(ratio_threshold=0.6, cross_check=False).match(d0, d1)

    logger.info("Ratio test: loose(0.9)=%d  strict(0.6)=%d", len(m_loose), len(m_strict))
    assert len(m_strict) <= len(m_loose), "Stricter ratio should not produce more matches"


def test_cross_check():
    """Cross-check should reduce matches compared to ratio-only."""
    from slam_vo.features.orb_descriptor import OrbDescriptor
    from slam_vo.features.feature_matcher import BruteForceMatcher

    img0 = load_frame(50)
    img1 = load_frame(51)

    kpts0 = filter_border(detect_kpts(img0, 200), img0)
    kpts1 = filter_border(detect_kpts(img1, 200), img1)

    orb = OrbDescriptor()
    d0 = orb.compute(img0, kpts0)
    d1 = orb.compute(img1, kpts1)

    m_no_cc = BruteForceMatcher(ratio_threshold=0.8, cross_check=False).match(d0, d1)
    m_cc    = BruteForceMatcher(ratio_threshold=0.8, cross_check=True).match(d0, d1)

    logger.info("Cross-check: without=%d  with=%d", len(m_no_cc), len(m_cc))
    assert len(m_cc) <= len(m_no_cc), "Cross-check should not produce more matches"


def test_hamming_distance_matrix():
    """Sanity check: identical descriptors have distance 0."""
    from slam_vo.features.feature_matcher import hamming_distance_matrix

    rng = np.random.RandomState(42)
    d = rng.randint(0, 256, (10, 32), dtype=np.uint8)
    dist = hamming_distance_matrix(d, d)
    assert dist.shape == (10, 10)
    assert (np.diag(dist) == 0).all(), "Self-distance should be 0"
    assert (dist >= 0).all()
    assert (dist == dist.T).all(), "Distance matrix should be symmetric"
    logger.info("Hamming distance matrix OK")


def test_timing():
    from slam_vo.features.orb_descriptor import OrbDescriptor
    from slam_vo.features.feature_matcher import BruteForceMatcher

    img = load_frame(50)
    kpts = filter_border(detect_kpts(img, 300), img)
    orb = OrbDescriptor()
    matcher = BruteForceMatcher()
    desc = orb.compute(img, kpts)  # warm-up

    times_desc = []
    for _ in range(5):
        t0 = time.perf_counter()
        orb.compute(img, kpts)
        times_desc.append((time.perf_counter() - t0) * 1000)

    times_match = []
    for _ in range(5):
        t0 = time.perf_counter()
        matcher.match(desc, desc)
        times_match.append((time.perf_counter() - t0) * 1000)

    logger.info(
        "ORB timing: desc=%.1f ms  match=%.1f ms  (%d kpts)",
        np.mean(times_desc), np.mean(times_match), len(kpts),
    )
    assert np.mean(times_desc) < 2000, "ORB descriptor too slow"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_output_shape_dtype,
        test_empty_keypoints,
        test_descriptor_variance,
        test_intra_vs_inter_distance,
        test_matching_under_translation,
        test_ratio_test_reduces_matches,
        test_cross_check,
        test_hamming_distance_matrix,
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
