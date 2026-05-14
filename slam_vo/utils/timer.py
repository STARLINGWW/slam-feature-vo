"""Unified timing utility for profiling SLAM modules."""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Context-manager timer that logs elapsed time in milliseconds.

    Usage:
        with Timer("FAST detection"):
            keypoints = detector.detect(image)
        # prints: [FAST detection] 2.34 ms
    """

    _stats: Dict[str, list] = defaultdict(list)

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._start: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        assert self._start is not None
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        Timer._stats[self.name].append(self.elapsed_ms)
        if self.verbose:
            logger.debug("[%s] %.2f ms", self.name, self.elapsed_ms)

    # ------------------------------------------------------------------
    # Class-level statistics
    # ------------------------------------------------------------------

    @classmethod
    def summary(cls) -> Dict[str, Dict[str, float]]:
        """Return mean/min/max/count for every named timer."""
        import numpy as np

        result = {}
        for name, times in cls._stats.items():
            arr = np.array(times)
            result[name] = {
                "mean_ms": float(arr.mean()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "count": len(arr),
            }
        return result

    @classmethod
    def reset(cls) -> None:
        cls._stats.clear()

    @classmethod
    def log_summary(cls) -> None:
        stats = cls.summary()
        for name, s in sorted(stats.items()):
            logger.info(
                "[Timer] %-30s  mean=%.2f ms  min=%.2f  max=%.2f  n=%d",
                name,
                s["mean_ms"],
                s["min_ms"],
                s["max_ms"],
                s["count"],
            )


@contextmanager
def timed(name: str, verbose: bool = True):
    """Alias for Timer as a standalone context manager."""
    t = Timer(name, verbose=verbose)
    with t:
        yield t
