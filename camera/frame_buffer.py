import threading
import time
import numpy as np


class FrameBuffer:
    """Thread-safe ring buffer that stores the latest camera frame."""

    def __init__(self, maxsize: int = 2):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._maxsize = maxsize
        self._buffer: list[np.ndarray] = []
        self._frame_count = 0
        self._last_put_time: float = 0.0

    def put(self, frame: np.ndarray) -> None:
        with self._condition:
            if len(self._buffer) >= self._maxsize:
                self._buffer.pop(0)
            self._buffer.append(frame)
            self._frame_count += 1
            self._last_put_time = time.time()
            self._condition.notify_all()

    def has_recent_frame(self, max_age: float = 3.0) -> bool:
        """Check if a frame was received within max_age seconds."""
        with self._lock:
            if not self._buffer or self._last_put_time == 0:
                return False
            return (time.time() - self._last_put_time) < max_age

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def wait_for_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        with self._condition:
            if not self._buffer:
                self._condition.wait(timeout=timeout)
            if self._buffer:
                return self._buffer[-1]
            return None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._frame_count = 0
