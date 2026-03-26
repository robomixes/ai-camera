import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    """Encode a BGR frame as JPEG bytes."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize frame to fit within max_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))
