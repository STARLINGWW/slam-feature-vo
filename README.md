# slam_feature_vo

混合特征视觉里程计系统，用于研究生课程技术调研作业。

实现"实时前端跑传统特征 + 异步后端跑学习型回环检测"的混合架构：

- **前端**: 手写 FAST 角点检测、KLT 金字塔光流、ORB 描述子，PnP 位姿估计
- **后端**: 手写 SuperPoint 推理、LightGlue 匹配，kornia LoFTR 验证，位姿图优化

## 快速开始

```bash
conda activate slam_vo
python scripts/run_vo.py --config configs/euroc.yaml --seq MH_01_easy
```

## 评估

```bash
python scripts/evaluate.py --results results/euroc/MH_01_easy
```

## 目录结构

```
slam_feature_vo/
├── configs/        # 相机参数 + 算法超参
├── slam_vo/        # 主包
│   ├── features/   # 手写特征算法 (FAST/KLT/ORB/SuperPoint/LightGlue)
│   ├── vo/         # VO Pipeline
│   ├── loop/       # 回环检测
│   ├── datasets/   # EuRoC 数据加载
│   └── utils/      # 几何/可视化/评估/计时工具
├── tests/          # 单元测试
├── scripts/        # 运行脚本
└── docs/           # 算法笔记 + 实验记录
```

## 数据集

EuRoC MAV (ASL格式)，在 `configs/euroc.yaml` 中配置路径。
