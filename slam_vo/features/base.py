"""Abstract base classes for all hand-written feature modules."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseDetector(ABC):
    """特征检测器基类."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """检测关键点.

        Args:
            image: 灰度图像, shape (H, W), dtype uint8.

        Returns:
            角点坐标数组, shape (N, 2), 每行为 [x, y], dtype float32.
            未检测到时返回 shape (0, 2).
        """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.detect(image)


class BaseDescriptor(ABC):
    """特征描述子基类."""

    @abstractmethod
    def compute(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """计算描述子.

        Args:
            image: 灰度图像, shape (H, W), dtype uint8.
            keypoints: 关键点坐标, shape (N, 2), 每行为 [x, y].

        Returns:
            描述子数组, shape (N, D).
            二值描述子返回 dtype uint8 (packed bits) 或 bool.
        """

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        return self.compute(image, keypoints)


class BaseMatcher(ABC):
    """特征匹配器基类."""

    @abstractmethod
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """匹配两组描述子.

        Args:
            desc1: 第一帧描述子, shape (N, D).
            desc2: 第二帧描述子, shape (M, D).

        Returns:
            匹配索引对, shape (K, 2), 每行为 [idx_in_desc1, idx_in_desc2].
        """

    def __call__(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        return self.match(desc1, desc2)


class BaseTracker(ABC):
    """特征跟踪器基类."""

    @abstractmethod
    def track(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从上一帧跟踪特征点到当前帧.

        Args:
            prev_img: 上一帧灰度图, shape (H, W), dtype uint8.
            curr_img: 当前帧灰度图, shape (H, W), dtype uint8.
            prev_pts: 上一帧特征点坐标, shape (N, 2), dtype float32.

        Returns:
            curr_pts: 当前帧特征点坐标, shape (N, 2), dtype float32.
            status: 跟踪状态标志, shape (N,), dtype uint8, 1=成功 0=失败.
        """

    def __call__(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.track(prev_img, curr_img, prev_pts)
