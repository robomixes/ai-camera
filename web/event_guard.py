"""Shared gate for writing event images, used by every path that saves one.

There are two independent places that save an event image: the AI runner, and
face_recognition's own "best frame" buffer, which force-logs when a face track
goes stale. They were written separately, so a guard added to one silently did
not apply to the other — near-duplicate suppression worked for object events
and not for face events, which is the bulk of them.

Keeping the comparison here means a new save path gets the same treatment by
calling one function.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_signature: dict[str, np.ndarray] = {}


def frame_signature(frame) -> np.ndarray | None:
    """A tiny greyscale thumbnail for comparing one frame against another.

    32x32 discards sensor noise and compression artefacts, which differ between
    frames of a motionless scene, while keeping enough structure that a real
    change still registers.
    """
    import cv2
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    except Exception:
        return None


def is_near_duplicate(camera_id: str, frame, threshold: float) -> tuple[bool, float | None]:
    """Would this image look identical to the last one saved for this camera?

    Returns (skip, diff). Does not record anything — call record() only once the
    image has actually been written, so a save that was skipped for some other
    reason cannot become the baseline that suppresses later real events.
    """
    sig = frame_signature(frame)
    if sig is None:
        return False, None
    with _lock:
        prev = _last_signature.get(camera_id)
    if prev is None:
        return False, None
    try:
        diff = float(np.mean(np.abs(prev - sig)))
    except Exception:
        return False, None
    return (diff < threshold), diff


def record(camera_id: str, frame) -> None:
    """Remember a frame that was actually written."""
    sig = frame_signature(frame)
    if sig is None:
        return
    with _lock:
        _last_signature[camera_id] = sig


def reset(camera_id: str = "") -> None:
    with _lock:
        if camera_id:
            _last_signature.pop(camera_id, None)
        else:
            _last_signature.clear()
