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

def log_detection(detection_data, roi_area, image_filename, camera_id="", latitude=None, longitude=None):
    """Logs general object detections."""
    cam_id = camera_id or config.CAMERA_ID
    lat = latitude if latitude is not None else config.GPS_LATITUDE
    lon = longitude if longitude is not None else config.GPS_LONGITUDE
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
                cam_id,
                lat,
                lon,
                obj['class'], 
                obj['confidence'], 
                str(roi_area), 
                image_filename
            ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging detection to DB: {e}")

def log_face_detection_event(name, distance, image_filename, is_known, camera_id="", latitude=None, longitude=None):
    """Logs a Face Recognition event."""
    cam_id = camera_id or config.CAMERA_ID
    lat = latitude if latitude is not None else config.GPS_LATITUDE
    lon = longitude if longitude is not None else config.GPS_LONGITUDE
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
            cam_id,
            lat,
            lon,
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
                      date_from: str = "", date_to: str = "", camera_id: str = "") -> list:
    """Query recent detection and face events for the web dashboard."""
    events = []
    date_where, date_params = _build_date_clause(date_from, date_to)
    cam_where = ""
    cam_params = []
    if camera_id:
        cam_where = " AND camera_id = ?"
        cam_params = [camera_id]
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            cursor.execute(
                "SELECT id, timestamp, camera_id, object_type, confidence, image_path, "
                f"'detection' as event_type FROM detections WHERE 1=1{date_where}{cam_where} ORDER BY timestamp DESC",
                date_params + cam_params
            )
            for row in cursor.fetchall():
                events.append(dict(row))

        if event_type in ("all", "face"):
            cursor.execute(
                "SELECT id, timestamp, camera_id, person_name, distance, image_path, is_known, "
                f"'face' as event_type FROM face_events WHERE 1=1{date_where}{cam_where} ORDER BY timestamp DESC",
                date_params + cam_params
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


def get_event_count(event_type: str = "all", date_from: str = "", date_to: str = "", camera_id: str = "") -> int:
    """Get total count of events."""
    total = 0
    date_where, date_params = _build_date_clause(date_from, date_to)
    cam_where = ""
    cam_params = []
    if camera_id:
        cam_where = " AND camera_id = ?"
        cam_params = [camera_id]
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            cursor.execute(f"SELECT COUNT(*) FROM detections WHERE 1=1{date_where}{cam_where}", date_params + cam_params)
            total += cursor.fetchone()[0]

        if event_type in ("all", "face"):
            cursor.execute(f"SELECT COUNT(*) FROM face_events WHERE 1=1{date_where}{cam_where}", date_params + cam_params)
            total += cursor.fetchone()[0]

        conn.close()
    except Exception as e:
        print(f"Error counting events: {e}")
    return total


def delete_all_events(event_type: str = "all", date_from: str = "", date_to: str = "", camera_id: str = "") -> int:
    """Delete events in bulk, optionally filtered by type, date range, and camera. Also removes image files."""
    deleted = 0
    date_where, date_params = _build_date_clause(date_from, date_to)
    cam_where = ""
    cam_params = []
    if camera_id:
        cam_where = " AND camera_id = ?"
        cam_params = [camera_id]
    all_params = date_params + cam_params
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if event_type in ("all", "detection"):
            cursor.execute(f"SELECT image_path FROM detections WHERE 1=1{date_where}{cam_where}", all_params)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.ROI_OUTPUT_DIR)
            cursor.execute(f"DELETE FROM detections WHERE 1=1{date_where}{cam_where}", all_params)
            deleted += cursor.rowcount

        if event_type in ("all", "face"):
            cursor.execute(f"SELECT image_path FROM face_events WHERE 1=1{date_where}{cam_where}", all_params)
            for row in cursor.fetchall():
                _delete_image_file(row["image_path"], config.EVENT_IMAGE_DIR)
            cursor.execute(f"DELETE FROM face_events WHERE 1=1{date_where}{cam_where}", all_params)
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


def get_today_stats() -> dict:
    """Get detection counts for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    stats = {"total_detections": 0, "total_faces": 0, "persons": 0, "known_faces": 0, "unknown_faces": 0}
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (f"{today} 00:00:00",))
        stats["total_detections"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ? AND object_type = 'person'",
                       (f"{today} 00:00:00",))
        stats["persons"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM face_events WHERE timestamp >= ?", (f"{today} 00:00:00",))
        stats["total_faces"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM face_events WHERE timestamp >= ? AND is_known = 1",
                       (f"{today} 00:00:00",))
        stats["known_faces"] = cursor.fetchone()[0]

        stats["unknown_faces"] = stats["total_faces"] - stats["known_faces"]

        conn.close()
    except Exception as e:
        print(f"Error getting today stats: {e}")
    return stats


def get_hourly_activity(hours: int = 24) -> list:
    """Get detection counts per hour for the last N hours."""
    from datetime import datetime, timedelta
    activity = []
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()

        now = datetime.now()
        for i in range(hours - 1, -1, -1):
            hour_start = (now - timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            start_str = hour_start.strftime("%Y-%m-%d %H:%M:%S")
            end_str = hour_end.strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ? AND timestamp < ?",
                           (start_str, end_str))
            det_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM face_events WHERE timestamp >= ? AND timestamp < ?",
                           (start_str, end_str))
            face_count = cursor.fetchone()[0]

            activity.append({
                "hour": hour_start.strftime("%H:%M"),
                "detections": det_count,
                "faces": face_count,
            })

        conn.close()
    except Exception as e:
        print(f"Error getting hourly activity: {e}")
    return activity


def get_top_objects(limit: int = 10) -> list:
    """Get most frequently detected object classes today."""
    today = datetime.now().strftime("%Y-%m-%d")
    objects = []
    try:
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT object_type, COUNT(*) as cnt FROM detections "
            "WHERE timestamp >= ? GROUP BY object_type ORDER BY cnt DESC LIMIT ?",
            (f"{today} 00:00:00", limit)
        )
        for row in cursor.fetchall():
            objects.append({"label": row[0], "count": row[1]})

        conn.close()
    except Exception as e:
        print(f"Error getting top objects: {e}")
    return objects


# need datetime import at module level
from datetime import datetime


# Run initialization if script is executed directly
if __name__ == "__main__":
    initialize_db()