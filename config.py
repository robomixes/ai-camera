# config.py

# --- Application Configuration ---

# Directory where captured images and videos will be saved
OUTPUT_DIR = "detected_media"

# Directory where ROI-filtered detection images will be saved
ROI_OUTPUT_DIR = "ROI_detection"

# Minimum delay (in seconds) required between logging two consecutive events 
# in the ROI-filtered mode (Option 6). Set to 0 to log every frame.
LOG_DELAY_SECONDS = 25.0 # <--- NEW SETTING: 1.0 second delay by default

# --- AI Configuration ---

# List of YOLOv8 class names to detect.
# If this list is empty (default), ALL objects will be detected.
# Example: DETECTION_CLASSES = ['person', 'car', 'dog']
DETECTION_CLASSES = ['person']