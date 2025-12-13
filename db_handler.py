# db_handler.py

import sqlite3
from datetime import datetime

DATABASE_NAME = 'roi_detection.db'
TABLE_NAME = 'roi_detections'

def initialize_database():
    """Ensures the SQLite database and table exist, including the new 'is_processed' flag."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # Create the table if it does not exist
        # NOTE: Adding the new 'is_processed' column set to 0 (False) by default
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                detection_list TEXT NOT NULL,
                roi_area TEXT,
                image_filename TEXT,
                is_processed INTEGER NOT NULL DEFAULT 0 
            )
        """)
        conn.commit()
        conn.close()
        print(f"Database '{DATABASE_NAME}' initialized successfully.")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")

def log_detection(detection_data, roi_area, image_filename):
    """
    Inserts a new record into the database table upon a filtered detection event.
    The new detection_data format is: [(label, confidence), (label, confidence), ...].
    
    Args:
        detection_data (list): List of tuples (object name, confidence) detected outside the ROI.
        roi_area (tuple): The (x, y, w, h) of the ROI that was ignored.
        image_filename (str): The path to the saved image associated with the event.
    """
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert list of tuples [(name, conf)] to a formatted string: "name (95%), name (98%)"
        if detection_data:
            detections_str = ", ".join([f"{name} ({int(conf*100)}%)" for name, conf in detection_data])
        else:
            detections_str = "None"
            
        roi_str = str(roi_area) if roi_area else "N/A"
        
        # is_processed defaults to 0 (False) when inserted
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (timestamp, detection_list, roi_area, image_filename)
            VALUES (?, ?, ?, ?)
        """, (timestamp, detections_str, roi_str, image_filename))
        
        conn.commit()
        conn.close()
        print(f"Database log created for: {detections_str}")
    except sqlite3.Error as e:
        print(f"Error logging detection: {e}")

# Call initialization on import so the database is ready when the main script runs
initialize_database()