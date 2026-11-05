# config.py

import os
import json
import logging
import tempfile
import threading

_logger = logging.getLogger(__name__)

# --- Runtime Overrides (persisted across restarts) ---
_CONFIG_DIR = os.path.join("data", "config") if os.path.isdir(os.path.join("data", "config")) else "."
_OVERRIDES_FILE = os.path.join(_CONFIG_DIR, "config_overrides.json")

# Serialises the read-merge-write in save_overrides(). The dashboard and the
# background rediscovery task can both persist settings.
_overrides_lock = threading.RLock()

# Keys the user has explicitly saved from the dashboard. Environment variables act
# as first-boot defaults only, so a saved setting is never silently reverted on restart.
_SAVED_KEYS = set()


def _load_overrides():
    """Load saved runtime settings from JSON file."""
    if os.path.exists(_OVERRIDES_FILE):
        try:
            with open(_OVERRIDES_FILE, 'r') as f:
                overrides = json.load(f)
            for key, value in overrides.items():
                globals()[key] = value
                _SAVED_KEYS.add(key)
            _logger.info(f"Loaded {len(overrides)} setting override(s) from {_OVERRIDES_FILE}")
        except Exception as e:
            _logger.warning(f"Could not load overrides: {e}")

def overrides_path() -> str:
    """Path of the overrides file actually in use (varies with the working directory)."""
    return _OVERRIDES_FILE


def save_overrides(settings: dict):
    """Save runtime settings to JSON file for persistence.

    Serialised and atomic. Two writers previously raced here: the read-merge
    then truncate-and-write is not atomic, so one save could silently drop
    another's changes, and a crash mid-write left invalid JSON. That is worse
    than it sounds — _load_overrides() swallows a parse error with a warning,
    so the next start would quietly revert EVERY persisted setting to its
    default with nothing in the UI to say so.
    """
    with _overrides_lock:
        existing = {}
        if os.path.exists(_OVERRIDES_FILE):
            try:
                with open(_OVERRIDES_FILE, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update(settings)

        # Write beside the target, flush all the way to disk, then swap it in:
        # os.replace is atomic, so a reader sees either the old file or the new
        # one and never a half-written one.
        directory = os.path.dirname(_OVERRIDES_FILE) or "."
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".overrides-", suffix=".tmp")
            with os.fdopen(fd, 'w') as f:
                json.dump(existing, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, _OVERRIDES_FILE)
            tmp_path = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    _logger.info(f"Saved {len(settings)} setting(s) to {_OVERRIDES_FILE}")

# --- Camera Identity & Location ---
CAMERA_ID = "CAM_001"
CAMERA_DESCRIPTION = "Front Gate - North Entrance"

# GPS Coordinates (Static location for this installation)
GPS_LATITUDE = 48.8584
GPS_LONGITUDE = 2.2945

# --- General Configuration ---
OUTPUT_DIR = "output_images"
ROI_OUTPUT_DIR = "roi_events"
LOG_DELAY_SECONDS = 5.0

# --- Database & Storage Settings ---
# In Docker, use data/db/ directory (mounted volume). Direct install uses project root.
_DB_DIR = os.path.join("data", "db") if os.path.isdir(os.path.join("data", "db")) else "."
DB_NAME = os.path.join(_DB_DIR, "detections_history.db")
EVENT_IMAGE_DIR = "event_images"

# Example: Filter for people/car detection only
DETECTION_CLASSES = ['person']

# Minimum YOLO confidence for a detection to count. Ultralytics defaults to 0.25,
# which lets low-confidence false positives through (railings, statues, shadows
# read as "person").
#
# Measured against this camera on an empty scene:
#   daylight:  the balustrade peaks at 39%  -> 0.50 was clear
#   night/IR:  the same balustrade peaks at 53%
#
# Infrared flattens texture and stonework reads far more like a person, so a
# threshold tuned in daylight lets night artefacts through. 0.60 clears the night
# peak with margin; genuine people measured 69-93% in daylight.
DETECTION_CONFIDENCE = 0.6

# Draw detection boxes on the images saved for events. Off by default — the boxes
# obscure faces, plates and other detail you want when reviewing an event. The live
# stream always shows boxes regardless of this setting.
SAVE_ANNOTATED_IMAGES = False

# --- Camera auto-rediscovery ---
# A camera that reboots can come back on a different DHCP lease; the capture
# backend would retry the old address forever and the system stays blind. When
# a camera has been silent long enough to rule out a reboot, scan its subnet
# for a host serving the same stream, verify it really is the same camera, then
# reattach and persist the new address.
#
# The scan range is derived from the camera's OWN configured address, never
# from a local interface: in Docker the container sits on a bridge network
# (172.x) while the camera is on the LAN, so local interfaces would send the
# search somewhere useless.
#
# Note the trade-off: verifying a candidate means connecting to it with the
# camera's credentials. Only hosts already listening on the RTSP port are ever
# tried, which on a home subnet is a handful, but a hostile device listening
# there could harvest them. Disable this, or pin AUTO_DISCOVERY_SUBNET, on a
# network you do not trust. A DHCP reservation on the router avoids the whole
# problem and is the better fix where you can set one.
AUTO_DISCOVERY_ENABLED = True
AUTO_DISCOVERY_AUTO_APPLY = True          # False: report a find, never rewrite config
AUTO_DISCOVERY_CHECK_INTERVAL_SECONDS = 15
AUTO_DISCOVERY_OFFLINE_GRACE_SECONDS = 120   # longer than a cold boot + reconnect ceiling
AUTO_DISCOVERY_FAST_GRACE_SECONDS = 30       # old address refuses the port: it is gone, not booting
AUTO_DISCOVERY_MIN_INTERVAL_SECONDS = 300    # first retry gap, doubling from here
AUTO_DISCOVERY_MAX_INTERVAL_SECONDS = 1800   # never slower than this, and never gives up
AUTO_DISCOVERY_SUBNET = ""                   # explicit CIDR; overrides the derived range
AUTO_DISCOVERY_SUBNET_PREFIX = 24
AUTO_DISCOVERY_MAX_HOSTS = 512               # refuse to sweep anything larger
AUTO_DISCOVERY_PORT_TIMEOUT = 0.5
AUTO_DISCOVERY_PORT_CONCURRENCY = 32
AUTO_DISCOVERY_VERIFY_TIMEOUT_SECONDS = 6.0
AUTO_DISCOVERY_MIN_FRAMES = 2                # a stream that opens then dies is not a camera
AUTO_DISCOVERY_REQUIRE_RESOLUTION_MATCH = True
AUTO_DISCOVERY_REQUIRE_SCENE_MATCH = True    # the check that rejects a stranger's camera
AUTO_DISCOVERY_SCENE_MAX_DIFF = 45.0         # generous: day/night IR shifts the scene a lot
AUTO_DISCOVERY_SCAN_TIMEOUT_SECONDS = 180    # a hung search must not wedge the feature

# Drop an event whose image is visually indistinguishable from the last one
# saved. A motionless scene that keeps re-triggering writes the same picture
# repeatedly; there is nothing new to look at and it buries real events.
# The number is the mean per-pixel difference (0-255) of a 32x32 greyscale
# thumbnail: lower is stricter, 0 disables by never matching.
DEDUPE_SIMILAR_EVENTS = True
# Measured twice, because the first measurement asked the wrong question.
# Frames captured two seconds apart in a motionless scene differ by at most
# 0.03 — but real events arrive 5-30s apart, and over that gap sun, shadow and
# auto-exposure drift the picture. Consecutive SAVED events actually differ by
# 0.32-0.99 (median 0.45), so a threshold of 0.3 suppressed nothing at all.
#
# 1.0 covers the observed duplicate band and still clears the ~1.4 that a person
# far from the camera produces. That margin is modest: this reduces repeat
# images of an unchanging scene, it does not replace excluding the region that
# keeps triggering in the first place.
DEDUPE_DIFF_THRESHOLD = 1.0
# --- Face Recognition Paths & Thresholds ---
# Location of the TFLite model
FACENET_MODEL_PATH = "facenet.tflite"

# Base directory where the known faces JSON and image files are stored
FACE_IMAGE_BASE_DIR = "people_search_queue/ready" 

# Location of the JSON file mapping names to image files (inside the base dir)
KNOWN_FACES_DB = os.path.join(FACE_IMAGE_BASE_DIR, "known_faces.json")

# Core Recognition Thresholds
# Haar cascade strictness. minNeighbors is how many overlapping detections a region
# needs before it counts as a face; OpenCV's default of 5 is permissive enough that
# stonework and railings register as faces on every frame.
#
# Measured over 30 live frames of a scene containing no faces:
#   minNeighbors=5  -> false face in 30/30 frames   (the old default)
#   minNeighbors=8  -> 1/30
#   minNeighbors=10 -> 0/30
#
# Raising the minimum face size also reaches zero, but only at ~12% of frame width,
# which is larger than a real face at street distance from this camera — it would
# reject genuine people. minNeighbors is the lever that does not cost reach, so the
# size ratio stays at the original 5%.
FACE_MIN_NEIGHBORS = 15
FACE_MIN_SIZE_RATIO = 0.05

# If the embedding distance is BELOW this, the face is called by name.
#
# Measured against this camera's own history:
#   genuine recognitions   Tala avg 0.734, anas avg 0.780 (mostly below 0.82)
#   a false match on a gateway, named as a known person   0.840 - 0.900
#
# At 0.9 the false matches sat just inside the threshold and were logged with a
# real person's name — 360 of them in one afternoon. 0.82 rejects that whole
# band while keeping typical genuine recognitions. The cost is that a marginal
# real frame is reported as Unknown rather than named, which is the better way
# to be wrong: a missed name is recoverable, a confidently wrong one is not.
RECOGNITION_THRESHOLD = 0.82

# Parts of the view that should never produce a detection, as fractions of the
# frame so they survive a resolution change:
#   [{"x": 0.84, "y": 0.20, "w": 0.16, "h": 0.22, "label": "gateway"}]
#
# For a fixed camera, some scene features register as detections at particular
# times of day and never stop: a gateway whose dark frame and bright opening
# read as a face, a balustrade that reads as a person under infrared. These
# land in the same confidence band as genuine distant detections, so no
# threshold separates them, and they arrive far enough apart that duplicate
# suppression cannot either. Excluding the region is the only approach that
# removes the noise without also discarding real detections elsewhere.
#
# A detection is dropped only when its CENTRE falls inside a region, so someone
# walking past one is still detected.
IGNORE_REGIONS = []
REJECTION_DISTANCE = 1.4     # If distance is ABOVE this, reject the detection as a non-face artifact.
INPUT_SIZE = (160, 160)      # FaceNet input size

# --- Multi-Frame Aggregation Constants ---
EMBEDDING_HISTORY_SIZE = 5     # Number of past embeddings to average for stability
MIN_IOU_THRESHOLD = 0.5         # Minimum IoU overlap required to consider a detection as a continuation of a tracked face

# --- NEW: Display Configuration ---
ENABLE_GUI_DISPLAY = True    # Set to False to run headless (no windows shown)

# config.py (Additions)

# --- Menu Settings ---
MENU_TIMEOUT_SECONDS = 25  # Number of seconds to wait before auto-selecting
MENU_DEFAULT_CHOICE = '5'  # The default option to pick on timeout (e.g., '5' for AI Analysis)

# --- Camera Backend ---
CAMERA_TYPE = "rtsp"       # "auto", "picamera", or "rtsp"
RTSP_URL = "rtsp://USERNAME:PASSWORD@CAMERA_IP/h264Preview_01_main"  # sub-stream for lower latency
RTSP_TRANSPORT = "tcp"     # "tcp" or "udp"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Web Server ---
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
STREAM_JPEG_QUALITY = 70
STREAM_MAX_WIDTH = 960
STREAM_TARGET_FPS = 10

# --- Default AI Mode (for single-camera setup) ---
DEFAULT_AI_MODE = "both"  # "yolo", "facenet", or "both"

# --- Smart Logging ---
DETECTION_COOLDOWN_SECONDS = 60.0  # Don't re-log the same detection for this many seconds

# --- Data Retention ---
DATA_RETENTION_DAYS = 30       # Auto-delete events older than this (0 = keep forever)
MIN_FREE_DISK_MB = 100         # Stop saving images if disk free < this

# --- ANPR (License Plate Recognition) ---
ANPR_ENABLED = False
ANPR_FRAME_INTERVAL = 5          # Run OCR every N frames
ANPR_MIN_VEHICLE_WIDTH = 100     # Min vehicle bbox width (px) before attempting OCR
ANPR_PLATE_COOLDOWN_SECONDS = 30 # Don't re-log same plate for this many seconds
ANPR_VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
ANPR_PLATE_MODEL_PATH = "yolov8n-plate.pt"  # Optional: dedicated plate detection model
PLATE_IMAGE_DIR = "plate_images"

# --- Alerts ---
ALERT_ENABLED = True
ALERT_EVENTS = ["unknown_face", "person_detected", "known_face", "watchlist_plate"]

# --- Multi-Camera Support ---
# If CAMERAS is empty, falls back to single-camera config above.
# Each entry: {"id": "...", "description": "...", "type": "rtsp", "url": "rtsp://...",
#              "transport": "tcp", "width": 1280, "height": 720,
#              "latitude": 0.0, "longitude": 0.0, "ai_mode": "both"}
CAMERAS = []
# Example:
# CAMERAS = [
#     {"id": "CAM_001", "description": "Front Gate", "type": "rtsp",
#      "url": "rtsp://USERNAME:PASSWORD@CAMERA_IP/h264Preview_01_main",
#      "transport": "tcp", "width": 1280, "height": 720,
#      "latitude": 48.8584, "longitude": 2.2945, "ai_mode": "both"},
#     {"id": "CAM_002", "description": "Back Door", "type": "rtsp",
#      "url": "rtsp://USERNAME:PASSWORD@CAMERA_IP/h264Preview_01_main",
#      "transport": "tcp", "width": 1280, "height": 720,
#      "latitude": 48.8585, "longitude": 2.2946, "ai_mode": "yolo"},
# ]

# --- Load overrides (file-based, then env vars) ---
_load_overrides()

# Environment variable overrides: defaults for a fresh deployment. A value the user
# has saved from the dashboard wins, otherwise a hardcoded env var in docker-compose.yml
# would silently revert that setting on every restart.
_ENV_MAP = {
    "CAMERA_TYPE": ("CAMERA_TYPE", str),
    "RTSP_URL": ("RTSP_URL", str),
    "RTSP_TRANSPORT": ("RTSP_TRANSPORT", str),
    "CAMERA_ID": ("CAMERA_ID", str),
    "CAMERA_DESCRIPTION": ("CAMERA_DESCRIPTION", str),
    "WEB_HOST": ("WEB_HOST", str),
    "WEB_PORT": ("WEB_PORT", int),
    "FRAME_WIDTH": ("FRAME_WIDTH", int),
    "FRAME_HEIGHT": ("FRAME_HEIGHT", int),
    "STREAM_TARGET_FPS": ("STREAM_TARGET_FPS", int),
    "STREAM_MAX_WIDTH": ("STREAM_MAX_WIDTH", int),
    "STREAM_JPEG_QUALITY": ("STREAM_JPEG_QUALITY", int),
    "DEFAULT_AI_MODE": ("DEFAULT_AI_MODE", str),
    "ANPR_ENABLED": ("ANPR_ENABLED", lambda v: v.lower() in ("true", "1", "yes")),
    "AUTO_DISCOVERY_ENABLED": ("AUTO_DISCOVERY_ENABLED", lambda v: v.lower() in ("true", "1", "yes")),
    "AUTO_DISCOVERY_SUBNET": ("AUTO_DISCOVERY_SUBNET", str),
}

for _env_key, (_config_key, _cast) in _ENV_MAP.items():
    if _config_key in _SAVED_KEYS:
        # Explicitly saved from the dashboard — keep the user's choice.
        continue
    _env_val = os.getenv(_env_key)
    if _env_val is not None:
        try:
            globals()[_config_key] = _cast(_env_val)
        except (ValueError, TypeError):
            pass