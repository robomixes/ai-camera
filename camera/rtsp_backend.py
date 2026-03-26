import os
import threading
import time
import cv2
from camera.base import CameraBase
from camera.frame_buffer import FrameBuffer

# Suppress FFmpeg h264 decode warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"


class RTSPBackend(CameraBase):
    """Camera backend for RTSP/IP cameras using OpenCV."""

    def __init__(self, url: str, transport: str = "tcp",
                 width: int = 1280, height: int = 720):
        self._url = url
        self._transport = transport
        self._width = width
        self._height = height
        self._running = False
        self._thread: threading.Thread | None = None
        self._buffer = FrameBuffer(maxsize=1)
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait for the first frame (up to 5 seconds)
        frame = self._buffer.wait_for_frame(timeout=5.0)
        if frame is not None:
            self._height, self._width = frame.shape[:2]
            print(f"RTSP stream connected: {self._url} ({self._width}x{self._height})")
        else:
            print(f"Warning: No frame received yet from {self._url}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._buffer.clear()
        print("RTSP stream stopped.")

    def get_frame(self):
        return self._buffer.get_latest()

    def is_running(self) -> bool:
        return self._running

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def _capture_loop(self) -> None:
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            cap = self._open_capture()
            if cap is None:
                print(f"RTSP reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            self._cap = cap
            backoff = 1.0  # reset on successful connect

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print("RTSP frame read failed. Reconnecting...")
                    break
                self._buffer.put(frame)

            cap.release()
            self._cap = None

    def _open_capture(self) -> cv2.VideoCapture | None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self._transport}"
            "|stimeout;10000000"
            "|timeout;10000000"
        )

        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Failed to open RTSP stream: {self._url}")
            cap.release()
            return None
        return cap
