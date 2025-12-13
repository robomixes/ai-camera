# db_handler.py

import sqlite3
import datetime
import os
import config # Using config for potential future needs

DB_NAME = "ai_events.db"
# Define the path where the known face event images will be saved
EVENT_IMAGE_DIR = "known_face_events"

# Ensure the event directory exists
if not os.path.exists(EVENT_IMAGE_DIR):
    os.makedirs(EVENT_IMAGE_DIR)

def initialize_db():
    """
    Initializes the database and creates necessary tables.
    
    NOTE: SQLite does not allow direct column modification (ALTER TABLE). 
    If you change a column type (like min_distance from REAL to TEXT), 
    you must manually rename or delete the old table before running this function.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. YOLO/ROI Detection Events Table (Unchanged)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS roi_detections (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            data TEXT,
            roi TEXT,
            image_file TEXT
        )
    """)
    
    # 2. Face Recognition Events Table
    # min_distance is now explicitly TEXT as requested.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_recognition_events (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            name TEXT NOT NULL,
            confidence REAL NOT NULL, 
            is_known BOOLEAN NOT NULL,
            min_distance TEXT, 
            image_file TEXT,
            is_processed BOOLEAN DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()

def log_detection(detection_data, roi_area, image_filename):
    """Logs a general YOLO detection event (used by Option 6)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO roi_detections 
            (timestamp, data, roi, image_file)
            VALUES (?, ?, ?, ?)
        """, (timestamp, str(detection_data), str(roi_area), image_filename))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to log ROI detection event: {e}")
    finally:
        conn.close()


def log_face_detection_event(name, distance, image_filename, is_known):
    """
    Logs a face recognition event to the database.
    Converts 'distance' (float) to a TEXT string before saving.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Confidence is 1 - L2 distance (higher is better)
    confidence = 1.0 - distance 
    
    # --- CONVERT DISTANCE TO TEXT ---
    # Format the float as a string with high precision (10 decimal places)
    # This value will be saved into the min_distance TEXT column.
    distance_text = f"{distance:.10f}" 
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO face_recognition_events 
            (timestamp, name, confidence, is_known, min_distance, image_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, name, confidence, is_known, distance_text, image_filename))
        
        conn.commit()
        print(f"DATABASE: Logged face event for {name} (Conf: {confidence:.2f})")
        
    except sqlite3.Error as e:
        print(f"ERROR: Failed to log face recognition event: {e}")
        
    finally:
        conn.close()