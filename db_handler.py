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

def _build_date_clause(date_from: str, date_to: str) -> tuple[str, list]:
    """Build SQL WHERE clause for date filtering."""
    clauses = []
    params = []
    if date_from:
        clauses.append("timestamp >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        clauses.append("timestamp <= ?")
        params.append(f"{date_to} 23:59:59")
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


def get_recent_events(limit: int = 50, event_type: str = "all", offset: int = 0,
                      date_from: str = "", date_to: str = "") -> list:
    """Query recent detection and face events for the web dashboard."""
    events = []
    date_where, date_params = _build_date_clause(date_from, date_to)
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            cursor.execute(
                "SELECT id, timestamp, camera_id, object_type, confidence, image_path, "
                f"'detection' as event_type FROM detections WHERE 1=1{date_where} ORDER BY timestamp DESC",
                date_params
            )
            for row in cursor.fetchall():
                events.append(dict(row))

        if event_type in ("all", "face"):
            cursor.execute(
                "SELECT id, timestamp, camera_id, person_name, distance, image_path, is_known, "
                f"'face' as event_type FROM face_events WHERE 1=1{date_where} ORDER BY timestamp DESC",
                date_params
            )
            for row in cursor.fetchall():
                events.append(dict(row))

        conn.close()

        # Sort combined results by timestamp descending
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return events[offset:offset + limit]

    except Exception as e:
        print(f"Error querying events: {e}")
        return []


def get_event_count(event_type: str = "all", date_from: str = "", date_to: str = "") -> int:
    """Get total count of events."""
    total = 0
    date_where, date_params = _build_date_clause(date_from, date_to)
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            cursor.execute(f"SELECT COUNT(*) FROM detections WHERE 1=1{date_where}", date_params)
            total += cursor.fetchone()[0]

        if event_type in ("all", "face"):
            cursor.execute(f"SELECT COUNT(*) FROM face_events WHERE 1=1{date_where}", date_params)
            total += cursor.fetchone()[0]

        conn.close()
    except Exception as e:
        print(f"Error counting events: {e}")
    return total


def delete_all_events(event_type: str = "all", date_from: str = "", date_to: str = "") -> int:
    """Delete events in bulk, optionally filtered by type and date range. Also removes image files."""
    deleted = 0
    date_where, date_params = _build_date_clause(date_from, date_to)
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            # Get image paths before deleting
            cursor.execute(f"SELECT image_path FROM detections WHERE 1=1{date_where}", date_params)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.ROI_OUTPUT_DIR)
            cursor.execute(f"DELETE FROM detections WHERE 1=1{date_where}", date_params)
            deleted += cursor.rowcount

        if event_type in ("all", "face"):
            cursor.execute(f"SELECT image_path FROM face_events WHERE 1=1{date_where}", date_params)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.EVENT_IMAGE_DIR)
            cursor.execute(f"DELETE FROM face_events WHERE 1=1{date_where}", date_params)
            deleted += cursor.rowcount

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error bulk deleting events: {e}")
    return deleted


def _delete_image_file(image_path, base_dir):
    """Safely delete an event image file."""
    if not image_path:
        return
    try:
        full_path = os.path.join(base_dir, image_path)
        if os.path.exists(full_path):
            os.remove(full_path)
    except Exception:
        pass


def delete_events(ids: list) -> int:
    """Delete events by their composite ids (type_id format, e.g. 'detection_5', 'face_12'). Also removes image files."""
    deleted = 0
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        det_ids = []
        face_ids = []
        for eid in ids:
            parts = str(eid).split("_", 1)
            if len(parts) == 2:
                if parts[0] == "detection":
                    det_ids.append(int(parts[1]))
                elif parts[0] == "face":
                    face_ids.append(int(parts[1]))

        if det_ids:
            placeholders = ",".join("?" * len(det_ids))
            cursor.execute(f"SELECT image_path FROM detections WHERE id IN ({placeholders})", det_ids)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.ROI_OUTPUT_DIR)
            cursor.execute(f"DELETE FROM detections WHERE id IN ({placeholders})", det_ids)
            deleted += cursor.rowcount

        if face_ids:
            placeholders = ",".join("?" * len(face_ids))
            cursor.execute(f"SELECT image_path FROM face_events WHERE id IN ({placeholders})", face_ids)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.EVENT_IMAGE_DIR)
            cursor.execute(f"DELETE FROM face_events WHERE id IN ({placeholders})", face_ids)
            deleted += cursor.rowcount

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting events: {e}")
    return deleted


# Run initialization if script is executed directly
if __name__ == "__main__":
    initialize_db()