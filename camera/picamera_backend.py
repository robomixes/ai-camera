import logging
import time
from camera.base import CameraBase

logger = logging.getLogger(__name__)


class PicameraBackend(CameraBase):
    """Camera backend for Raspberry Pi Camera (Picamera2)."""

    def __init__(self, width: int = 1280, height: int = 720):
        self._width = width
        self._height = height
        self._picam2 = None
        self._running = False

    def start(self) -> None:
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        self._running = True
        logger.info("Picamera2 started.")
        time.sleep(1)

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            try:
                del self._picam2
            except Exception:
                pass
            self._picam2 = None
        self._running = False
        logger.info("Picamera2 stopped.")

    def get_frame(self):
        if self._picam2 is None:
            return None
        frame_rgb = self._picam2.capture_array()
        # Picamera2 returns RGB888; convert to BGR for OpenCV
        return frame_rgb[:, :, ::-1]

    def is_running(self) -> bool:
        return self._running

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)
