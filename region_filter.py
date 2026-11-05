"""Ignore detections that fall inside regions the user has marked as noise.

Some parts of a fixed camera's view produce false detections no threshold can
remove. On this installation a gateway — a dark frame around a bright opening
with railings behind it — reads as a face at certain sun angles, and a stone
balustrade reads as a person under infrared. Both are at fixed positions and
neither is a real detection, ever.

Thresholds cannot separate them: the false matches land in the same confidence
band as genuine distant detections, and duplicate suppression cannot either,
because events minutes apart differ more than a distant person does. Excluding
the region is the only approach that removes the noise without also removing
real detections elsewhere in the frame.

Regions are stored as fractions of the frame so they survive a resolution
change — switching between the camera's sub and main streams would otherwise
silently move every region.
"""

import logging

logger = logging.getLogger(__name__)


def _regions():
    import config
    regions = getattr(config, "IGNORE_REGIONS", None) or []
    return regions if isinstance(regions, list) else []


def is_ignored(center_x: float, center_y: float, frame_w: int, frame_h: int) -> bool:
    """Is this detection's centre inside an ignored region?

    The centre is used rather than any overlap so that a person walking past an
    ignored region is still detected — only something whose middle sits inside
    it is dropped.
    """
    if not frame_w or not frame_h:
        return False
    fx = center_x / float(frame_w)
    fy = center_y / float(frame_h)
    for r in _regions():
        try:
            x = float(r["x"]); y = float(r["y"])
            w = float(r["w"]); h = float(r["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if x <= fx <= x + w and y <= fy <= y + h:
            return True
    return False


def filter_boxes(boxes, frame_w: int, frame_h: int):
    """Drop (x, y, w, h) boxes whose centre falls in an ignored region."""
    if not _regions():
        return boxes, 0
    kept = []
    dropped = 0
    for b in boxes:
        x, y, w, h = b[0], b[1], b[2], b[3]
        if is_ignored(x + w / 2.0, y + h / 2.0, frame_w, frame_h):
            dropped += 1
            continue
        kept.append(b)
    return kept, dropped
