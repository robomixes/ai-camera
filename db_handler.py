# db_handler.py - Handles SQLite logging with GPS, Camera ID, and Configurable Paths

import sqlite3
import os
import config  # Pulls DB_NAME, EVENT_IMAGE_DIR, CAMERA_ID, and GPS info

def initialize_db():
    """Creates the SQLite database and tables based on paths in config.py."""
    
    # Ensure the event image directory exists
    if not os.path.exists(config.EVENT_IMAGE_DIR):
        os.makedirs(config.EVENT_IMAGE_DIR)
        print(f"Created directory: {config.EVENT_IMAGE_DIR}")

    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    
    print(f"Initializing database: {config.DB_NAME}")

    # 1. Table for General YOLO Detections
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT,
            latitude REAL,
            longitude REAL,
            object_type TEXT,
            confidence REAL,
            roi_box TEXT,
            image_path TEXT
        )
    ''')

    # 2. Table for Face Recognition Events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT,
            latitude REAL,
            longitude REAL,
            person_name TEXT,
            distance TEXT,
            image_path TEXT,
            is_known INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database tables verified.")

def log_detection(detection_data, roi_area, image_filename):
    """Logs general object detections using config-defined paths and ID."""
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        
        for obj in detection_data:
            cursor.execute('''
                INSERT INTO detections (
                    camera_id, latitude, longitude, object_type, 
                    confidence, roi_box, image_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.CAMERA_ID, 
                config.GPS_LATITUDE, 
                config.GPS_LONGITUDE,
                obj['class'], 
                obj['confidence'], 
                str(roi_area), 
                image_filename
            ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging detection to DB: {e}")

def log_face_detection_event(name, distance, image_filename, is_known):
    """Logs a Face Recognition event using config-defined paths and ID."""
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        
        distance_text = f"{distance:.4f}"
                
        cursor.execute('''
            INSERT INTO face_events (
                camera_id, latitude, longitude, person_name, 
                distance, image_path, is_known
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            config.CAMERA_ID, 
            config.GPS_LATITUDE, 
            config.GPS_LONGITUDE,
            name, 
            distance_text, 
            image_filename, 
            1 if is_known else 0
        ))
        
        conn.commit()
        conn.close()
        print(f"Logged Face Event: {name} from {config.CAMERA_ID}")
    except Exception as e:
        print(f"Error logging face event to DB: {e}")

# Run initialization if script is executed directly
if __name__ == "__main__":
    initialize_db()