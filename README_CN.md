# SLAM Feature VO

<div align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</div>

一个包含**六种手写特征算法**的单目视觉里程计（VO）系统，用于研究生课程技术调研作业，深入理解 SLAM/VIO 场景下的特征处理全流程。

---

## 系统架构

```
实时前端（逐帧处理）                          后端回环（异步，关键帧级）
┌────────────────────────────────┐           ┌────────────────────────────────────┐
│ ① FAST 角点检测（手写）          │           │ ④ SuperPoint 特征提取（手写）        │
│ ② KLT 金字塔光流跟踪（手写）     │  关键帧   │ ⑤ LightGlue 回环匹配（手写）         │
│ ③ rBRIEF ORB 描述子（手写）     │─────────→ │ ⑥ LoFTR 验证（调 kornia 库）         │
│    PnP 位姿估计（调 OpenCV）     │           │    位姿图优化                        │
└────────────────────────────────┘           └────────────────────────────────────┘
```

**约束：** 特征算法 ①–⑤ 全部基于 NumPy/SciPy 手写实现。  
`cv2.ORB_create`、`cv2.goodFeaturesToTrack`、`cv2.calcOpticalFlowPyrLK` **不用于生产代码**。  
几何计算（`findEssentialMat`、`solvePnPRansac`、`triangulatePoints`）可调用 OpenCV。

---

## 快速开始

### 1 · 安装

```bash
conda create -n slam_vo python=3.10
conda activate slam_vo
pip install -r requirements.txt
```

### 2 · EuRoC MAV 数据集

从 [ASL 数据集页面](https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf) 下载（ZIP，ASL 格式）。  
修改 `configs/euroc.yaml`，填入本地路径：

```yaml
dataset:
  base_path: "/your/path/to/euroc"   # ← 修改此处
  sequences:
    - MH_01_easy
```

### 3 · 在 EuRoC 上运行

```bash
# 完整序列
python scripts/run_vo.py --seq MH_01_easy

# 仅处理前 1000 帧，跳过绘图
python scripts/run_vo.py --seq MH_01_easy --max_frames 1000 --no_plot

# 指定配置文件
python scripts/run_vo.py --seq MH_03_medium --config configs/euroc.yaml
```

结果保存到 `results/euroc/MH_01_easy/`：

| 文件 | 内容 |
|------|------|
| `trajectory_est.txt` | 估计轨迹（TUM 格式） |
| `trajectory_gt_matched.txt` | 对齐的真值轨迹 |
| `metrics.json` | ATE / RPE / 耗时指标 |
| `trajectory_plot.png` | 俯视轨迹对比图 |

### 4 · 评估已保存的轨迹

```bash
python scripts/evaluate.py \
  --est results/euroc/MH_01_easy/trajectory_est.txt \
  --gt  results/euroc/MH_01_easy/trajectory_gt_matched.txt \
  --align sim3
```

---

## 实时摄像头 / 外接摄像头

### 环境要求

只需普通摄像头。为获得最佳精度，建议先标定摄像头（见下方说明），并将参数填入 `configs/webcam.yaml`。

### 使用方式

```bash
# 内置摄像头（索引 0）
python scripts/run_camera.py

# 外接 USB 摄像头（索引 1）
python scripts/run_camera.py --camera 1

# RTSP / IP 摄像头流
python scripts/run_camera.py --camera "rtsp://user:pass@192.168.1.100:554/stream"

# 视频文件（离线测试，无需 EuRoC 数据集）
python scripts/run_camera.py --camera path/to/video.mp4

# 命令行直接指定内参
python scripts/run_camera.py \
    --camera 0 \
    --fx 600 --fy 600 --cx 320 --cy 240 \
    --width 640 --height 480

# 使用摄像头配置文件 + 保存标注视频
python scripts/run_camera.py \
    --config configs/webcam.yaml \
    --camera 0 \
    --save_video results/camera/live_run.mp4

# 无界面模式（如服务器端运行）
python scripts/run_camera.py --camera 0 --no_display
```

### 实时窗口快捷键

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出 |
| `r` | 重置跟踪器 |
| `s` | 保存当前轨迹快照 |

### 摄像头标定

拍摄约 20 张棋盘格图片（9×6 内角点，格子边长 25 mm），然后运行：

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

将输出值填入 `configs/webcam.yaml` 的 `camera.fx / fy / cx / cy / distortion` 字段。

---

## 运行测试

```bash
conda activate slam_vo

# 全部单元测试
python -m pytest tests/ -v --tb=short

# 按模块单独测试
python -m pytest tests/test_fast.py -v
python -m pytest tests/test_klt.py  -v
python -m pytest tests/test_orb.py  -v
python -m pytest tests/test_vo_pipeline.py -v

# EuRoC 集成测试（需要数据集）
python -m pytest tests/test_vo_pipeline.py::TestVoOnEuRoC -v -s
```

---

## 算法说明

### ① FAST-9 角点检测

Bresenham 圆（半径 3，16 像素）。像素 `p` 被判定为角点的条件：圆上存在 ≥ 9 个**连续**像素，亮度均高于 `p + threshold` 或均低于 `p − threshold`。

- **高速预筛选**：检查 4 个罗盘方向像素，至少 **2 个**满足强度条件（FAST-9 的必要条件）。
- **NMS**：float64 位偏移保证每个像素的唯一排序，消除平分问题。
- 实现：[`slam_vo/features/fast_detector.py`](slam_vo/features/fast_detector.py)

### ② 金字塔 KLT 光流跟踪

Bouguet 算法。每层：迭代求解 2×2 Lucas-Kanade 方程组；累积流向量 `g`，在金字塔层间×2 上采样。

- 图像块提取使用 `scipy.ndimage.map_coordinates`（对所有 N 个点并行向量化）。
- 前向-后向一致性检验过滤不可靠的跟踪点。
- 实现：[`slam_vo/features/klt_tracker.py`](slam_vo/features/klt_tracker.py)

### ③ rBRIEF ORB 描述子

- **方向估计**：强度质心法，半径 15 px 圆内计算。
- **描述子**：对旋转后的高斯采样点对进行 256 次二值测试，`np.packbits` 压缩为 32 字节。
- **匹配**：XOR + 查表法 popcount；Lowe 比值测试（0.75）+ 可选互检验。
- 实现：[`slam_vo/features/orb_descriptor.py`](slam_vo/features/orb_descriptor.py)、[`slam_vo/features/feature_matcher.py`](slam_vo/features/feature_matcher.py)

### VO 管线

| 模块 | 文件 |
|------|------|
| 状态机（未初始化→初始化→跟踪→丢失） | [`slam_vo/vo/tracker.py`](slam_vo/vo/tracker.py) |
| 局部地图 | [`slam_vo/vo/local_map.py`](slam_vo/vo/local_map.py) |
| 几何计算封装 | [`slam_vo/utils/geometry.py`](slam_vo/utils/geometry.py) |
| EuRoC 数据加载器 | [`slam_vo/datasets/euroc.py`](slam_vo/datasets/euroc.py) |

**初始化**：双帧本质矩阵 → `recoverPose` → 三角化 → 归一化（中位深度 = 10 m）。  
**跟踪**：KLT 跟踪活跃地图点 → PnP RANSAC → 保留内点 → 更新位姿。  
**关键帧**：跟踪比率 < 0.65 或平移超过阈值时插入；对待定种子点进行三角化。

---

## 对比测评：手写 KLT vs OpenCV

测试场景：640×480 合成纹理图像，每帧平移 8–12 px。

| 指标 | 手写 KLT | `cv2.calcOpticalFlowPyrLK` |
|------|:--------:|:---------------------------:|
| 终点平均误差 | < 2 px | — （参考基准） |
| 跟踪率 | 与 cv2 差异 < 15% | 参考基准 |
| 运行速度（Python+NumPy） | ~1200 ms/帧 | ~2.5 ms/帧 |

Python 实现比 C++ 慢约 500 倍，这是算法正确性演示的固有代价，符合预期。  
精度验证见 `tests/test_vo_pipeline.py::TestKltVsOpenCV`。

---

## 目录结构

```
slam_feature_vo/
├── configs/              # YAML 配置文件（EuRoC、摄像头）
├── slam_vo/
│   ├── features/         # ★ 手写特征算法
│   ├── vo/               # VO 管线
│   ├── datasets/         # EuRoC MAV 数据加载器
│   └── utils/            # 几何、评估、可视化、计时器
├── scripts/
│   ├── run_vo.py         # 在 EuRoC 数据集上运行 VO
│   ├── run_camera.py     # 在实时摄像头/视频上运行 VO
│   ├── evaluate.py       # 独立的 TUM 轨迹评估工具
│   └── visualize_results.py
└── tests/                # 单元测试 + 集成测试
```

---

## 许可证

MIT
