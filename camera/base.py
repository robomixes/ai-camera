from abc import ABC, abstractmethod
import numpy as np


class CameraBase(ABC):
    """Abstract base class for all camera backends."""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Returns a BGR numpy array, or None if no frame is available."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        ...
