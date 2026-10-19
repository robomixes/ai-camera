# config.py

import os
import json
import logging

_logger = logging.getLogger(__name__)

# --- Runtime Overrides (persisted across restarts) ---
_CONFIG_DIR = os.path.join("data", "config") if os.path.isdir(os.path.join("data", "config")) else "."
_OVERRIDES_FILE = os.path.join(_CONFIG_DIR, "config_overrides.json")

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
    """Save runtime settings to JSON file for persistence."""
    # Read existing overrides and merge
    existing = {}
    if os.path.exists(_OVERRIDES_FILE):
        try:
            with open(_OVERRIDES_FILE, 'r') as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(settings)
    with open(_OVERRIDES_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
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
# read as "person"). Raise this if static objects keep getting detected.
DETECTION_CONFIDENCE = 0.5

# Draw detection boxes on the images saved for events. Off by default — the boxes
# obscure faces, plates and other detail you want when reviewing an event. The live
# stream always shows boxes regardless of this setting.
SAVE_ANNOTATED_IMAGES = False
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

RECOGNITION_THRESHOLD = 0.9  # If distance is BELOW this, it's a known person.
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
RTSP_URL = "rtsp://admin:AnAs1001kad!@192.168.1.2/h264Preview_01_sub"  # sub-stream for lower latency
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
#      "url": "rtsp://admin:pass@192.168.1.9/h264Preview_01_sub",
#      "transport": "tcp", "width": 1280, "height": 720,
#      "latitude": 48.8584, "longitude": 2.2945, "ai_mode": "both"},
#     {"id": "CAM_002", "description": "Back Door", "type": "rtsp",
#      "url": "rtsp://admin:pass@192.168.1.10/h264Preview_01_sub",
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