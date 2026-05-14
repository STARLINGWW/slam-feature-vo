# SLAM Feature VO

<div align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</div>

A monocular Visual Odometry (VO) system with **six hand-written feature algorithms**, designed as a graduate-course research project for deep-diving into feature processing pipelines in SLAM/VIO contexts.

---

## Architecture

```
Real-time Front-end (per-frame)                 Back-end Loop Closure (async, KF-level)
┌────────────────────────────────┐              ┌───────────────────────────────────────┐
│ ① FAST corner detection        │              │ ④ SuperPoint feature extraction       │
│ ② Pyramidal KLT optical flow   │  keyframes   │ ⑤ LightGlue loop matching             │
│ ③ rBRIEF ORB descriptor        │────────────→ │ ⑥ LoFTR verification (kornia)         │
│    PnP pose estimation (cv2)   │              │    Pose-graph optimisation            │
└────────────────────────────────┘              └───────────────────────────────────────┘
```

**Rule:** All feature algorithms (①–⑤) are written from scratch using NumPy/SciPy.  
`cv2.ORB_create`, `cv2.goodFeaturesToTrack`, `cv2.calcOpticalFlowPyrLK` are **not used** in production code. Geometry helpers (`findEssentialMat`, `solvePnPRansac`, `triangulatePoints`) use OpenCV.

---

## Quick Start

### 1 · Install

```bash
conda create -n slam_vo python=3.10
conda activate slam_vo
pip install -r requirements.txt
```

### 2 · EuRoC MAV dataset

Download from the [ASL dataset page](https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf) (ZIP, ASL format).  
Edit `configs/euroc.yaml` and set `dataset.base_path`:

```yaml
dataset:
  base_path: "/path/to/your/euroc"   # ← change this
  sequences:
    - MH_01_easy
```

### 3 · Run on EuRoC

```bash
# Full sequence
python scripts/run_vo.py --seq MH_01_easy

# First 1000 frames only, skip plot
python scripts/run_vo.py --seq MH_01_easy --max_frames 1000 --no_plot

# With custom config
python scripts/run_vo.py --seq MH_03_medium --config configs/euroc.yaml
```

Results are saved to `results/euroc/MH_01_easy/`:

| File | Content |
|------|---------|
| `trajectory_est.txt` | Estimated trajectory (TUM format) |
| `trajectory_gt_matched.txt` | Matched ground-truth trajectory |
| `metrics.json` | ATE / RPE / timing metrics |
| `trajectory_plot.png` | Top-down trajectory comparison |

### 4 · Evaluate a saved trajectory

```bash
python scripts/evaluate.py \
  --est results/euroc/MH_01_easy/trajectory_est.txt \
  --gt  results/euroc/MH_01_easy/trajectory_gt_matched.txt \
  --align sim3
```

---

## Live Camera / Webcam

### Requirements

No special hardware needed beyond a camera. For best accuracy, calibrate your camera (see below) and fill in `configs/webcam.yaml`.

### Usage

```bash
# Built-in webcam (index 0)
python scripts/run_camera.py

# External USB camera (index 1)
python scripts/run_camera.py --camera 1

# RTSP / IP camera stream
python scripts/run_camera.py --camera "rtsp://user:pass@192.168.1.100:554/stream"

# Video file (offline testing without EuRoC)
python scripts/run_camera.py --camera path/to/video.mp4

# Override intrinsics on the command line
python scripts/run_camera.py \
    --camera 0 \
    --fx 600 --fy 600 --cx 320 --cy 240 \
    --width 640 --height 480

# Use webcam config + save annotated video
python scripts/run_camera.py \
    --config configs/webcam.yaml \
    --camera 0 \
    --save_video results/camera/live_run.mp4

# Headless mode (no display window, e.g. on a server)
python scripts/run_camera.py --camera 0 --no_display
```

### Keyboard shortcuts (live window)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset tracker |
| `s` | Save trajectory snapshot |

### Camera calibration

Capture ~20 images of a chessboard (9×6 inner corners, 25 mm squares), then:

```bash
python -c "
import cv2, glob, numpy as np
imgs = [cv2.imread(p) for p in sorted(glob.glob('calib/*.jpg'))]
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 0.025
obj_pts, img_pts = [], []
for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        obj_pts.append(objp)
        img_pts.append(corners)
_, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
print('fx=%.3f  fy=%.3f  cx=%.3f  cy=%.3f' % (K[0,0],K[1,1],K[0,2],K[1,2]))
print('k1=%.6f  k2=%.6f  p1=%.6f  p2=%.6f' % tuple(dist.ravel()[:4]))
"
```

Paste the values into `configs/webcam.yaml` → `camera.fx / fy / cx / cy / distortion`.

---

## Running Tests

```bash
conda activate slam_vo

# All unit tests
python -m pytest tests/ -v --tb=short

# Individual modules
python -m pytest tests/test_fast.py -v
python -m pytest tests/test_klt.py  -v
python -m pytest tests/test_orb.py  -v
python -m pytest tests/test_vo_pipeline.py -v

# VO integration test on EuRoC (requires dataset)
python -m pytest tests/test_vo_pipeline.py::TestVoOnEuRoC -v -s
```

---

## Algorithm Notes

### ① FAST-9 Corner Detector

Bresenham circle of radius 3 (16 pixels). A pixel `p` is a corner if ≥ 9 **consecutive** pixels on the circle are all brighter than `p + threshold` or all darker than `p − threshold`.

- **High-speed pre-filter**: checks 4 compass pixels; at least **2** must satisfy the intensity condition (necessary for FAST-9).
- **NMS**: float64 tiebreaking ensures unique per-pixel ordering.
- [`slam_vo/features/fast_detector.py`](slam_vo/features/fast_detector.py)

### ② Pyramidal KLT Tracker

Bouguet's algorithm. Per-level: solve the 2×2 Lucas-Kanade system; accumulate flow `g`, scale ×2 between pyramid levels.

- Patch extraction via `scipy.ndimage.map_coordinates` (vectorised over all N points at once).
- Forward-backward consistency check filters unreliable tracks.
- [`slam_vo/features/klt_tracker.py`](slam_vo/features/klt_tracker.py)

### ③ rBRIEF ORB Descriptor

- **Orientation**: intensity centroid method on disc of radius 15 px.
- **Descriptor**: 256 binary tests on Gaussian-sampled pairs, rotated by computed angle; packed to 32 bytes via `np.packbits`.
- **Matching**: XOR + popcount LUT; Lowe ratio test (0.75) + optional cross-check.
- [`slam_vo/features/orb_descriptor.py`](slam_vo/features/orb_descriptor.py), [`slam_vo/features/feature_matcher.py`](slam_vo/features/feature_matcher.py)

### VO Pipeline

| Component | File |
|-----------|------|
| State machine (UNINIT → INIT → TRACK → LOST) | [`slam_vo/vo/tracker.py`](slam_vo/vo/tracker.py) |
| Local map | [`slam_vo/vo/local_map.py`](slam_vo/vo/local_map.py) |
| Geometry wrappers | [`slam_vo/utils/geometry.py`](slam_vo/utils/geometry.py) |
| EuRoC loader | [`slam_vo/datasets/euroc.py`](slam_vo/datasets/euroc.py) |

**Initialization**: Two-frame Essential matrix → `recoverPose` → triangulate → normalise so median depth = 10 m.  
**Tracking**: KLT track active map points → PnP RANSAC → keep inliers → update pose.  
**Keyframe**: insert when tracked-ratio < 0.65 or translation > threshold; triangulate pending seeds.

---

## Benchmark: Custom vs OpenCV

Synthetic 640×480 frames, 8–12 px/frame translation.

| Metric | Custom KLT | `cv2.calcOpticalFlowPyrLK` |
|--------|:----------:|:---------------------------:|
| Mean endpoint error | < 2 px | — (reference) |
| Tracking rate | within ±15% of cv2 | reference |
| Speed (Python+NumPy) | ~1200 ms/frame | ~2.5 ms/frame |

The ~500× speed gap is inherent to Python vs. C++ and is expected for an algorithm-correctness demonstration. The test suite verifies correctness: see `tests/test_vo_pipeline.py::TestKltVsOpenCV`.

---

## Project Structure

```
slam_feature_vo/
├── configs/              # YAML configs (EuRoC, webcam)
├── slam_vo/
│   ├── features/         # ★ Hand-written feature algorithms
│   ├── vo/               # VO pipeline
│   ├── datasets/         # EuRoC MAV loader
│   └── utils/            # Geometry, evaluation, visualization, timer
├── scripts/
│   ├── run_vo.py         # Run VO on EuRoC dataset
│   ├── run_camera.py     # Run VO on live camera / video
│   ├── evaluate.py       # Standalone TUM trajectory evaluator
│   └── visualize_results.py
└── tests/                # Unit + integration tests
```

---

## License

MIT
