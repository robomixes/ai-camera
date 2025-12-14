# config.py

import os
# --- General Configuration ---
OUTPUT_DIR = "output_images"
ROI_OUTPUT_DIR = "roi_events"
LOG_DELAY_SECONDS = 5.0 

# Example: Filter for people/car detection only
DETECTION_CLASSES = ['person']
# --- Face Recognition Paths & Thresholds ---
# Location of the TFLite model
FACENET_MODEL_PATH = "facenet.tflite"

# Base directory where the known faces JSON and image files are stored
FACE_IMAGE_BASE_DIR = "people_search_queue/ready" 

# Location of the JSON file mapping names to image files (inside the base dir)
KNOWN_FACES_DB = os.path.join(FACE_IMAGE_BASE_DIR, "known_faces.json")

# Core Recognition Thresholds
RECOGNITION_THRESHOLD = 0.9  # If distance is BELOW this, it's a known person.
REJECTION_DISTANCE = 1.4     # If distance is ABOVE this, reject the detection as a non-face artifact.
INPUT_SIZE = (160, 160)      # FaceNet input size

# --- Multi-Frame Aggregation Constants ---
EMBEDDING_HISTORY_SIZE = 5     # Number of past embeddings to average for stability
MIN_IOU_THRESHOLD = 0.5         # Minimum IoU overlap required to consider a detection as a continuation of a tracked face

# --- NEW: Display Configuration ---
ENABLE_GUI_DISPLAY = False   # Set to False to run headless (no windows shown)

# config.py (Additions)

# --- Menu Settings ---
MENU_TIMEOUT_SECONDS = 25  # Number of seconds to wait before auto-selecting
MENU_DEFAULT_CHOICE = '7'  # The default option to pick on timeout (e.g., '1' for Camera Feed)