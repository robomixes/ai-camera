# face_recognition.py (Version with Multi-Frame Temporal Averaging)

import cv2
import numpy as np
import json
import os
import sys
import datetime
import time # <--- THIS LINE IS THE FIX
from tensorflow.lite.python.interpreter import Interpreter 
import config 
import db_handler 

# --- Global Variables ---
FACE_DETECTOR = None 
FACENET_INTERPRETER = None
KNOWN_EMBEDDINGS = {} 
TRACKED_FACES = {} # Dictionary to store state: {track_id: TrackedFace object}
NEXT_TRACK_ID = 0  # Counter for assigning unique IDs

# ----------------------------------------------------------------------
# --- NEW: TrackedFace Class for State Management ---
# ----------------------------------------------------------------------

class TrackedFace:
    """Holds state information for a single person tracked across frames."""
    def __init__(self, track_id, bbox, initial_embedding):
        self.id = track_id
        # Bounding box: (x, y, w, h)
        self.bbox = bbox 
        # History queue of the last N embeddings
        self.embedding_history = [initial_embedding] 
        # Time of last update (for cleanup)
        self.last_update_time = time.time() 
        self.name = "Unknown"
        self.distance = float('inf')

    def update(self, new_bbox, new_embedding):
        """Updates the tracking state with a new frame's data."""
        self.bbox = new_bbox
        self.embedding_history.append(new_embedding)
        self.last_update_time = time.time()
        
        # Keep the history buffer size constant
        if len(self.embedding_history) > config.EMBEDDING_HISTORY_SIZE:
            self.embedding_history.pop(0)

    def get_aggregated_embedding(self):
        """Calculates the stable, aggregated embedding by averaging the history."""
        # Calculate the mean (average) across the time dimension (axis=0)
        return np.mean(self.embedding_history, axis=0)
        
    def get_display_bbox(self):
        """Returns the current bounding box."""
        return self.bbox

# ----------------------------------------------------------------------
# --- Utility Functions (IoU for Tracking) ---
# ----------------------------------------------------------------------

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes (x, y, w, h).
    Used to associate a new detection with an existing tracked face.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x_min, y_min, x_max, y_max)
    box1_min_x, box1_min_y = x1, y1
    box1_max_x, box1_max_y = x1 + w1, y1 + h1
    box2_min_x, box2_min_y = x2, y2
    box2_max_x, box2_max_y = x2 + w2, y2 + h2

    # Determine the (x, y)-coordinates of the intersection rectangle
    inter_min_x = max(box1_min_x, box2_min_x)
    inter_min_y = max(box1_min_y, box2_min_y)
    inter_max_x = min(box1_max_x, box2_max_x)
    inter_max_y = min(box1_max_y, box2_max_y)

    # Compute the area of intersection
    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the intersection over union
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# ----------------------------------------------------------------------
# --- Initialization & Database Loading (Omitted for brevity, assumed unchanged) ---
# ----------------------------------------------------------------------
# NOTE: The initialize_facenet_model, load_known_faces_from_images, 
# and initialize_system functions are assumed to be identical to the previous script.
# They must be included in the final file.

# ... (Original initialize_facenet_model, load_known_faces_from_images, initialize_system, recognize_face, etc.) ...
# Included below for completeness, but typically placed higher in the file.

def initialize_facenet_model():
    """Initializes the TFLite interpreter for FaceNet."""
    global FACENET_INTERPRETER
    try:
        FACENET_INTERPRETER = Interpreter(model_path=config.FACENET_MODEL_PATH)
        FACENET_INTERPRETER.allocate_tensors()
        print(f"FaceNet Model loaded from {config.FACENET_MODEL_PATH}.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load FaceNet model: {e}")
        return False

def load_known_faces_from_images():
    """Loads known face embeddings."""
    global KNOWN_EMBEDDINGS
    KNOWN_EMBEDDINGS.clear()
    try:
        if not os.path.exists(config.KNOWN_FACES_DB): return False 
        with open(config.KNOWN_FACES_DB, 'r') as f:
            face_defs = json.load(f)
        
        # (Embedding calculation logic omitted for brevity, but it's the same)
        for name, image_list in face_defs.items():
            person_embeddings = []
            for image_file in image_list:
                image_path = os.path.join(config.FACE_IMAGE_BASE_DIR, image_file)
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    embedding = get_face_embedding(img_bgr)
                    if embedding is not None: person_embeddings.append(embedding)
            
            if person_embeddings:
                KNOWN_EMBEDDINGS[name] = np.mean(person_embeddings, axis=0)

        print(f"Known faces loaded successfully: {len(KNOWN_EMBEDDINGS)} unique names.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to read/parse face definitions JSON: {e}")
        return False
        
def initialize_system():
    if not initialize_facenet_model(): return False
    global FACE_DETECTOR
    try:
        FACE_DETECTOR = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if FACE_DETECTOR.empty(): print("WARNING: Haar Cascade missing.")
    except: pass
    return load_known_faces_from_images()

def get_face_embedding(face_image):
    # ... (same as before) ...
    global FACENET_INTERPRETER
    if FACENET_INTERPRETER is None: return None
    normalized_input = normalize_face(face_image)
    input_details = FACENET_INTERPRETER.get_input_details()
    output_details = FACENET_INTERPRETER.get_output_details()
    FACENET_INTERPRETER.set_tensor(input_details[0]['index'], normalized_input)
    FACENET_INTERPRETER.invoke()
    return FACENET_INTERPRETER.get_tensor(output_details[0]['index'])[0]

def normalize_face(face_image):
    face_resized = cv2.resize(face_image, config.INPUT_SIZE)
    face_norm = (face_resized.astype(np.float32) - 127.5) / 127.5
    return np.expand_dims(face_norm, axis=0)

def calculate_distance(emb1, emb2):
    return np.sqrt(np.sum(np.square(emb1 - emb2)))

def recognize_face(embedding):
    min_distance = float('inf')
    best_match_name = "Unknown"
    if not KNOWN_EMBEDDINGS: return "No Database", 0.0
    for name, known_emb in KNOWN_EMBEDDINGS.items():
        distance = calculate_distance(embedding, known_emb)
        if distance < min_distance:
            min_distance = distance
            best_match_name = name
    return best_match_name, min_distance

# ----------------------------------------------------------------------
# --- NEW: Tracking and Cleanup Functions ---
# ----------------------------------------------------------------------

def clean_stale_tracks(max_age_seconds=2.0):
    """Removes tracked faces that haven't been seen for a while."""
    global TRACKED_FACES
    current_time = time.time()
    stale_ids = [
        t_id for t_id, t_face in TRACKED_FACES.items() 
        if current_time - t_face.last_update_time > max_age_seconds
    ]
    for t_id in stale_ids:
        del TRACKED_FACES[t_id]
        # print(f"Track {t_id} removed (stale).")

def associate_detections(current_boxes, current_embeddings):
    """Matches current detections to existing tracks or creates new ones."""
    global TRACKED_FACES, NEXT_TRACK_ID
    
    # List of new detections that haven't been matched yet
    unmatched_detections = list(zip(current_boxes, current_embeddings))
    
    # 1. Try to match current detections to existing tracks
    for t_id, t_face in TRACKED_FACES.items():
        best_iou = 0
        best_match_index = -1
        
        # Find the detection that overlaps the most with this tracked face's last known position
        for i, (bbox, _) in enumerate(unmatched_detections):
            iou = calculate_iou(t_face.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_index = i
        
        # If a good match is found:
        if best_iou >= config.MIN_IOU_THRESHOLD and best_match_index != -1:
            # Update the tracked face with the new detection data
            new_bbox, new_embedding = unmatched_detections[best_match_index]
            t_face.update(new_bbox, new_embedding)
            
            # Remove the detection from the unmatched list
            unmatched_detections.pop(best_match_index)
            
    # 2. Create new tracks for all remaining unmatched detections
    for new_bbox, new_embedding in unmatched_detections:
        new_track = TrackedFace(NEXT_TRACK_ID, new_bbox, new_embedding)
        TRACKED_FACES[NEXT_TRACK_ID] = new_track
        # print(f"New track created: {NEXT_TRACK_ID}")
        NEXT_TRACK_ID += 1

# ----------------------------------------------------------------------
# --- Recognition Loop Function (Updated) ---
# ----------------------------------------------------------------------

def run_facenet_recognition(frame, picam2_frame_size):
    """
    Main function for running multi-frame face detection and recognition.
    """
    annotated_frame = frame.copy()
    
    # ------------------------------------------------------------------
    # STEP 1: FACE DETECTION & EMBEDDING CALCULATION (STATELSS)
    # ------------------------------------------------------------------
    
    raw_face_boxes = []
    current_embeddings = []
    
    if FACE_DETECTOR and not FACE_DETECTOR.empty():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_face_boxes = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in raw_face_boxes:
            face_image = frame[y:y+h, x:x+w]
            embedding = get_face_embedding(face_image)
            if embedding is not None:
                current_embeddings.append(embedding)
            else:
                raw_face_boxes.remove((x,y,w,h)) # Remove if embedding failed
        
    # ------------------------------------------------------------------
    # STEP 2: TRACKING AND AGGREGATION (STATEFUL)
    # ------------------------------------------------------------------
    
    clean_stale_tracks()
    associate_detections(raw_face_boxes, current_embeddings)
    
    detected_faces = [] 
    
    # ------------------------------------------------------------------
    # STEP 3: RECOGNITION AND DRAWING (USING AGGREGATED EMBEDDINGS)
    # ------------------------------------------------------------------
    
    for t_id, t_face in TRACKED_FACES.items():
        # Only perform recognition if we have enough frames for a stable average
        if len(t_face.embedding_history) >= 2: # Start after 2 frames for initial stability
            
            # Use the averaged embedding for recognition!
            aggregated_embedding = t_face.get_aggregated_embedding()
            name, distance = recognize_face(aggregated_embedding)
            
            # Update the tracked face's known status
            t_face.name = name
            t_face.distance = distance
        
        # Check against the stored results
        name = t_face.name
        distance = t_face.distance
        x, y, w, h = t_face.get_display_bbox()


        # FILTER 1: Reject detection if distance is too high (non-face artifact)
        if distance > config.REJECTION_DISTANCE:
            # Note: We keep the track alive for a moment, but don't draw or log a bad result
            continue 

        # FILTER 2: Determine final status (Known or Unknown)
        is_known_person = False
        if distance < config.RECOGNITION_THRESHOLD:
            display_name = name
            color = (0, 255, 0) # Green (Known)
            is_known_person = True 
        elif name == "No Database":
             display_name = "No Database"
             color = (0, 0, 255) 
        else:
            display_name = "Unknown"
            color = (0, 0, 255) # Red (Unknown Face)

        
        # --- LOGGING ACTION ---
        if is_known_person:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"faces_known_T{t_id}_{display_name}_{timestamp_str}.jpg"
            full_image_path = os.path.join(db_handler.EVENT_IMAGE_DIR, image_filename)
            
            cv2.imwrite(full_image_path, annotated_frame) 
            
            db_handler.log_face_detection_event(
                name=display_name,
                distance=distance,
                image_filename=image_filename,
                is_known=is_known_person
            )
        
        # --- DRAWING ---
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        # Display the Track ID and the recognition result
        label = f"T{t_id}: {display_name} ({distance:.2f})"
        cv2.putText(annotated_frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detected_faces.append((display_name, 1.0 - distance))

    info_text = f"Tracks: {len(TRACKED_FACES)} | Detections: {len(raw_face_boxes)} | Valid: {len(detected_faces)}"
    cv2.putText(annotated_frame, info_text, (10, picam2_frame_size[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return annotated_frame, detected_faces
