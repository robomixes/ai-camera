# face_recognition.py (Final Version with Temporal Aggregation, Fast Cleanup, and Best Frame Logging)

import cv2
import numpy as np
import json
import os
import sys
import datetime
import time # Import 'time' for time.time() calls
from tensorflow.lite.python.interpreter import Interpreter 
import config 
import db_handler 

# --- Global Variables ---
FACE_DETECTOR = None 
FACENET_INTERPRETER = None
KNOWN_EMBEDDINGS = {} 
TRACKED_FACES = {} # Dictionary to store face state: {track_id: TrackedFace object}
NEXT_TRACK_ID = 0  # Counter for assigning unique IDs

# --- Global Logging Buffer ---
# Stores the highest-quality detection data and frame copy during the logging delay window.
# {name: {'last_log_time': time.time(), 'best_frame': None, 'best_quality': float, 'distance': float}}
LOGGING_BUFFER = {} 

# ----------------------------------------------------------------------
# --- TrackedFace Class for State Management ---
# ----------------------------------------------------------------------

class TrackedFace:
    """Holds state information for a single person tracked across frames."""
    def __init__(self, track_id, bbox, initial_embedding):
        self.id = track_id
        # Bounding box: (x, y, w, h)
        self.bbox = bbox 
        # History queue of the last N embeddings
        self.embedding_history = [initial_embedding] 
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
# --- Utility Functions (IoU, Embeddings, Distance) ---
# ----------------------------------------------------------------------

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_min_x, box1_min_y = x1, y1
    box1_max_x, box1_max_y = x1 + w1, y1 + h1
    box2_min_x, box2_min_y = x2, y2
    box2_max_x, box2_max_y = x2 + w2, y2 + h2

    inter_min_x = max(box1_min_x, box2_min_x)
    inter_min_y = max(box1_min_y, box2_min_y)
    inter_max_x = min(box1_max_x, box2_max_x)
    inter_max_y = min(box1_max_y, box2_max_y)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou
    
def normalize_face(face_image):
    """Resizes and normalizes the face image for FaceNet input."""
    face_resized = cv2.resize(face_image, config.INPUT_SIZE)
    face_norm = (face_resized.astype(np.float32) - 127.5) / 127.5
    return np.expand_dims(face_norm, axis=0)

def get_face_embedding(face_image):
    """Runs the FaceNet model to get a 128-dim embedding."""
    global FACENET_INTERPRETER
    if FACENET_INTERPRETER is None: return None
        
    normalized_input = normalize_face(face_image)

    input_details = FACENET_INTERPRETER.get_input_details()
    output_details = FACENET_INTERPRETER.get_output_details()

    FACENET_INTERPRETER.set_tensor(input_details[0]['index'], normalized_input)
    FACENET_INTERPRETER.invoke()

    embedding = FACENET_INTERPRETER.get_tensor(output_details[0]['index'])[0]
    return embedding

def calculate_distance(emb1, emb2):
    """Calculates L2 (Euclidean) distance between two embeddings."""
    return np.sqrt(np.sum(np.square(emb1 - emb2)))

def recognize_face(embedding):
    """Compares a new embedding against all known embeddings."""
    min_distance = float('inf')
    best_match_name = "Unknown"
    
    if not KNOWN_EMBEDDINGS:
        return "No Database", 0.0

    for name, known_emb in KNOWN_EMBEDDINGS.items():
        distance = calculate_distance(embedding, known_emb)
        
        if distance < min_distance:
            min_distance = distance
            best_match_name = name
    
    return best_match_name, min_distance

# ----------------------------------------------------------------------
# --- Initialization & Database Loading ---
# ----------------------------------------------------------------------

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
    """
    Loads known face definitions from the JSON and calculates average embeddings.
    """
    global KNOWN_EMBEDDINGS
    KNOWN_EMBEDDINGS.clear()
    
    try:
        if not os.path.exists(config.KNOWN_FACES_DB):
            print(f"WARNING: Known faces JSON file NOT FOUND at {config.KNOWN_FACES_DB}.")
            return False 

        with open(config.KNOWN_FACES_DB, 'r') as f:
            face_defs = json.load(f)
            if not isinstance(face_defs, dict):
                 raise TypeError("JSON file must be a dictionary.")

    except Exception as e:
        print(f"ERROR: Failed to read/parse face definitions JSON: {e}")
        return False
        
    print("Calculating average embeddings from known face images...")
    
    for name, image_list in face_defs.items():
        person_embeddings = []
        
        if not isinstance(image_list, list): continue
             
        for image_file in image_list:
            image_path = os.path.join(config.FACE_IMAGE_BASE_DIR, image_file)
            
            if not os.path.exists(image_path): continue

            try:
                img_bgr = cv2.imread(image_path)
                if img_bgr is None: continue
                
                embedding = get_face_embedding(img_bgr)
                if embedding is not None:
                    person_embeddings.append(embedding)

            except Exception:
                pass 
        
        if person_embeddings:
            avg_embedding = np.mean(person_embeddings, axis=0)
            KNOWN_EMBEDDINGS[name] = avg_embedding
            print(f"  -> Calculated average embedding for '{name}' from {len(person_embeddings)} images.")

    print(f"Known faces loaded successfully: {len(KNOWN_EMBEDDINGS)} unique names.")
    return True
    
def initialize_system():
    """Combines model initialization and database loading."""
    if not initialize_facenet_model():
        return False
        
    global FACE_DETECTOR
    try:
        FACE_DETECTOR = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if FACE_DETECTOR.empty():
            print("WARNING: Haar Cascade (detection step) not found. Face detection reliability may suffer.")
    except Exception as e:
        print(f"WARNING: Haar Cascade (detection step) could not be loaded: {e}")
        
    return load_known_faces_from_images()

# ----------------------------------------------------------------------
# --- Tracking and Cleanup Functions (Updated for Fast Cleanup and Exit Log) ---
# ----------------------------------------------------------------------

def clean_stale_tracks(max_age_seconds=0.5): # FIX 1: Reduced Track Age to 0.5s
    """
    Removes tracked faces that haven't been seen for a while, 
    and forces a log if a known face track is dying with buffered evidence.
    """
    global TRACKED_FACES, LOGGING_BUFFER
    current_time = time.time()
    stale_ids = [
        t_id for t_id, t_face in TRACKED_FACES.items() 
        if current_time - t_face.last_update_time > max_age_seconds
    ]
    
    for t_id in stale_ids:
        track_name = TRACKED_FACES[t_id].name
        
        # FIX 2: Check for buffered evidence before track deletion
        if track_name in LOGGING_BUFFER:
            entry = LOGGING_BUFFER[track_name]
            
            # If we have a 'best_frame' buffered (i.e., the person was recognized)
            # we must process the log NOW before the evidence is lost, overriding the throttle.
            if entry['best_frame'] is not None:
                 
                 # --- FORCE LOG THE BEST EVENT ON EXIT ---
                 
                 timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 image_filename = f"faces_final_{track_name}_{timestamp_str}.jpg"
                 full_image_path = os.path.join(db_handler.EVENT_IMAGE_DIR, image_filename)
                 
                 # Save the best frame found
                 cv2.imwrite(full_image_path, entry['best_frame'])
                 
                 # Log the event to the database
                 db_handler.log_face_detection_event(
                    name=track_name,
                    distance=entry['distance'],
                    image_filename=image_filename,
                    is_known=True
                 )
                 
                 # Clean the buffer entry as it has now been logged
                 del LOGGING_BUFFER[track_name] 

        # Finally, delete the track from the main tracking dictionary
        del TRACKED_FACES[t_id]


def associate_detections(current_boxes, current_embeddings):
    """Matches current detections to existing tracks or creates new ones."""
    global TRACKED_FACES, NEXT_TRACK_ID
    
    unmatched_detections = list(zip(current_boxes, current_embeddings))
    
    # 1. Try to match current detections to existing tracks
    for t_id, t_face in TRACKED_FACES.items():
        best_iou = 0
        best_match_index = -1
        
        for i, (bbox, _) in enumerate(unmatched_detections):
            iou = calculate_iou(t_face.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_index = i
        
        if best_iou >= config.MIN_IOU_THRESHOLD and best_match_index != -1:
            new_bbox, new_embedding = unmatched_detections[best_match_index]
            t_face.update(new_bbox, new_embedding)
            unmatched_detections.pop(best_match_index)
            
    # 2. Create new tracks for all remaining unmatched detections
    for new_bbox, new_embedding in unmatched_detections:
        new_track = TrackedFace(NEXT_TRACK_ID, new_bbox, new_embedding)
        TRACKED_FACES[NEXT_TRACK_ID] = new_track
        NEXT_TRACK_ID += 1

# ----------------------------------------------------------------------
# --- Deferred Logging Function (Called from ai-general.py loop) ---
# ----------------------------------------------------------------------

def process_deferred_logs():
    """
    Checks the buffer and logs the best image for faces whose logging window has passed.
    This handles the throttle (time-based) logging.
    """
    global LOGGING_BUFFER
    current_time = time.time()
    
    for name, entry in list(LOGGING_BUFFER.items()): # Iterate over a copy of keys
        
        # Check 1: Is the full logging delay met since the last log?
        if current_time - entry['last_log_time'] >= config.LOG_DELAY_SECONDS:
            
            # Check 2: Do we have a valid best frame captured during this window?
            if entry['best_frame'] is not None:
                
                # --- LOG AND SAVE THE BEST EVENT (Throttle Trigger) ---
                
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"faces_best_{name}_{timestamp_str}.jpg"
                full_image_path = os.path.join(db_handler.EVENT_IMAGE_DIR, image_filename)
                
                # Save the best frame found
                cv2.imwrite(full_image_path, entry['best_frame'])
                
                # Log the event to the database
                db_handler.log_face_detection_event(
                    name=name,
                    distance=entry['distance'],
                    image_filename=image_filename,
                    is_known=True
                )
                
                # --- RESET THE BUFFER ---
                
                # Start the next logging window now
                entry['last_log_time'] = current_time 
                # Clear the stored frame/quality so the system starts looking for a *new* best frame
                entry['best_frame'] = None
                entry['best_quality'] = float('inf') 
            
            else:
                # If the window is met, but no best_frame was found (e.g., person was recognized, 
                # but then recognition failed for the rest of the window), just reset the timer
                # to start a fresh window immediately.
                entry['last_log_time'] = current_time

# ----------------------------------------------------------------------
# --- Recognition Loop Function ---
# ----------------------------------------------------------------------

def run_facenet_recognition(frame, picam2_frame_size):
    """
    Runs multi-frame face detection, recognition, and fills the LOGGING_BUFFER.
    """
    annotated_frame = frame.copy()
    
    # ------------------------------------------------------------------
    # STEP 1: FACE DETECTION & EMBEDDING CALCULATION
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
                raw_face_boxes.remove((x,y,w,h))
        
    # ------------------------------------------------------------------
    # STEP 2: TRACKING AND AGGREGATION
    # ------------------------------------------------------------------
    
    clean_stale_tracks()
    associate_detections(raw_face_boxes, current_embeddings)
    
    detected_faces = [] 
    
    # ------------------------------------------------------------------
    # STEP 3: RECOGNITION, DRAWING, and BUFFERING
    # ------------------------------------------------------------------
    
    for t_id, t_face in TRACKED_FACES.items():
        
        # Only perform recognition if we have enough frames for a stable average
        if len(t_face.embedding_history) >= 2: 
            
            aggregated_embedding = t_face.get_aggregated_embedding()
            name, distance = recognize_face(aggregated_embedding)
            
            t_face.name = name
            t_face.distance = distance
        
        # Check against the stored results
        name = t_face.name
        distance = t_face.distance
        x, y, w, h = t_face.get_display_bbox()


        # FILTER 1: Reject detection if distance is too high (non-face artifact)
        if distance > config.REJECTION_DISTANCE:
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
        
        # ------------------------------------------------------------------
        # --- LOGGING BUFFER FILLING (Only if KNOWN) ---
        # ------------------------------------------------------------------
        if is_known_person:
            person_name = display_name
            current_quality = distance # Lower distance (closer match) = Higher Quality
            
            if person_name not in LOGGING_BUFFER:
                # Initialize the buffer entry for a new person
                LOGGING_BUFFER[person_name] = {
                    'last_log_time': time.time() - config.LOG_DELAY_SECONDS, # Allow immediate log on first appearance
                    'best_frame': None, 
                    'best_quality': float('inf'),
                    'distance': distance
                } 
                
            buffer_entry = LOGGING_BUFFER[person_name]
            
            # Check if current frame is the BEST one seen in the current (or new) window
            if current_quality < buffer_entry['best_quality']:
                
                # Store the best available data
                buffer_entry['best_quality'] = current_quality
                buffer_entry['distance'] = distance
                # Save a copy of the annotated frame as the evidence
                buffer_entry['best_frame'] = annotated_frame.copy() 
        
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