# face_recognition.py

import cv2
import numpy as np
import json
import os
import sys
import datetime
from tensorflow.lite.python.interpreter import Interpreter 
import config 
import db_handler # For logging events

# --- Global Variables ---
FACE_DETECTOR = None # Haar Cascade Detector
FACENET_INTERPRETER = None
KNOWN_EMBEDDINGS = {} # Dictionary to store calculated AVERAGE embeddings: {name: numpy_array}

# ----------------------------------------------------------------------
# --- Utility Functions ---
# ----------------------------------------------------------------------

def normalize_face(face_image):
    """Resizes and normalizes the face image for FaceNet input."""
    face_resized = cv2.resize(face_image, config.INPUT_SIZE)
    face_norm = (face_resized.astype(np.float32) - 127.5) / 127.5
    return np.expand_dims(face_norm, axis=0)

def get_face_embedding(face_image):
    """Runs the FaceNet model to get a 128-dim embedding."""
    global FACENET_INTERPRETER
    if FACENET_INTERPRETER is None:
        print("FaceNet interpreter not initialized.")
        return None
        
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
    Loads known face definitions from the JSON and calculates average embeddings 
    for all associated physical image files.
    """
    global KNOWN_EMBEDDINGS
    KNOWN_EMBEDDINGS.clear()
    
    # 1. Load Face Definitions from JSON
    try:
        if not os.path.exists(config.KNOWN_FACES_DB):
            print(f"WARNING: Known faces JSON file NOT FOUND at {config.KNOWN_FACES_DB}.")
            return False 

        with open(config.KNOWN_FACES_DB, 'r') as f:
            face_defs = json.load(f)
            if not isinstance(face_defs, dict):
                 raise TypeError("JSON file must be a dictionary mapping Name to a list of image paths.")

    except Exception as e:
        print(f"ERROR: Failed to read/parse face definitions JSON: {e}")
        return False
        
    # 2. Iterate through names and images to calculate embeddings
    print("Calculating average embeddings from known face images...")
    
    for name, image_list in face_defs.items():
        person_embeddings = []
        
        if not isinstance(image_list, list):
             print(f"WARNING: Skipping '{name}'. Value must be a list of image names.")
             continue
             
        for image_file in image_list:
            image_path = os.path.join(config.FACE_IMAGE_BASE_DIR, image_file)
            
            if not os.path.exists(image_path):
                print(f"WARNING: Image not found for '{name}': {image_path}. Skipping.")
                continue

            try:
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    print(f"WARNING: Could not load image file: {image_path}. Skipping.")
                    continue
                
                embedding = get_face_embedding(img_bgr)
                if embedding is not None:
                    person_embeddings.append(embedding)

            except Exception as e:
                print(f"ERROR: Failed to process image {image_file} for '{name}': {e}")
        
        # 3. Store the average embedding
        if person_embeddings:
            avg_embedding = np.mean(person_embeddings, axis=0)
            KNOWN_EMBEDDINGS[name] = avg_embedding
            print(f"  -> Calculated average embedding for '{name}' from {len(person_embeddings)} images.")
        else:
             print(f"  -> WARNING: No valid images found for '{name}'. Skipping.")

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
# --- Recognition Loop Function ---
# ----------------------------------------------------------------------

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

def run_facenet_recognition(frame, picam2_frame_size):
    """
    Main function for running face detection and FaceNet recognition.
    """
    annotated_frame = frame.copy()
    detected_faces = [] 

    # ------------------------------------------------------------------
    # STEP 1: FACE DETECTION (using Haar Cascade)
    # ------------------------------------------------------------------
    
    face_boxes = []
    if FACE_DETECTOR and not FACE_DETECTOR.empty():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.size > 0:
            face_boxes = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    else:
         cv2.putText(annotated_frame, "Detector Missing!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    for (x, y, w, h) in face_boxes:
        
        face_image = frame[y:y+h, x:x+w]
        
        # ------------------------------------------------------------------
        # STEP 2 & 3: EMBEDDING & RECOGNITION (FACENET)
        # ------------------------------------------------------------------
        embedding = get_face_embedding(face_image)
        name, distance = recognize_face(embedding)

        # ------------------------------------------------------------------
        # STEP 4: FILTERING, DRAWING, AND LOGGING
        # ------------------------------------------------------------------
        
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
             color = (0, 0, 255) # Red (Error)
        else:
            display_name = "Unknown"
            color = (0, 0, 255) # Red (Unknown Face)

        
        # --- LOGGING ACTION: Log event and save image only for KNOWN faces ---
        if is_known_person:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Image naming convention: faces_kown_Name_datatime.jpg
            image_filename = f"faces_known_{display_name}_{timestamp_str}.jpg"
            full_image_path = os.path.join(db_handler.EVENT_IMAGE_DIR, image_filename)
            
            # Save the annotated frame for context
            cv2.imwrite(full_image_path, annotated_frame) 
            
            # Log the event to the database
            db_handler.log_face_detection_event(
                name=display_name,
                distance=distance,
                image_filename=image_filename,
                is_known=is_known_person
            )
        
        # --- DRAWING ---
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        label = f"{display_name} ({distance:.2f})"
        cv2.putText(annotated_frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detected_faces.append((display_name, 1.0 - distance))

    info_text = f"Faces Detected: {len(face_boxes)} | Valid Detections: {len(detected_faces)} | DB: {len(KNOWN_EMBEDDINGS)}"
    cv2.putText(annotated_frame, info_text, (10, picam2_frame_size[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return annotated_frame, detected_faces