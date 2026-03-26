import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse

import config
import db_handler
import face_recognition as face_rec
from camera import create_camera
from camera.base import CameraBase
from web.ai_runner import AIRunner
from web.auth import (initialize_auth, verify_login, change_password, create_session,
                       get_session_user, delete_session, check_rate_limit, record_failed_attempt, clear_attempts)
from web.camera_manager import CameraManager
from web.utils import encode_jpeg, resize_frame

_faces_lock = threading.Lock()


def _mask_url(url: str) -> str:
    """Mask password in RTSP URL. rtsp://admin:pass@host -> rtsp://admin:****@host"""
    if not url or "@" not in url:
        return url
    try:
        prefix, rest = url.split("@", 1)
        # Find the last : before @ which separates user:pass
        if ":" in prefix:
            scheme_user = prefix.rsplit(":", 1)[0]
            return f"{scheme_user}:****@{rest}"
    except Exception:
        pass
    return url


def _mask_sensitive(data: dict) -> dict:
    """Mask sensitive values in a dict for API responses."""
    masked = dict(data)
    if "url" in masked and masked["url"]:
        masked["url"] = _mask_url(masked["url"])
    if "value" in masked and isinstance(masked.get("description", ""), str):
        if "url" in masked.get("description", "").lower() or "rtsp" in str(masked.get("value", "")).lower():
            if isinstance(masked["value"], str) and "@" in masked["value"]:
                masked["value"] = _mask_url(masked["value"])
    return masked

# --- Globals set during lifespan ---
_manager: CameraManager | None = None

STATIC_DIR = Path(__file__).parent / "static"


def _get_cam(camera_id: str = ""):
    """Helper to get camera from manager."""
    return _manager.get_camera(camera_id) if _manager else None

def _get_runner(camera_id: str = ""):
    """Helper to get AI runner from manager."""
    return _manager.get_runner(camera_id) if _manager else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager

    # Startup
    initialize_auth()
    db_handler.initialize_db()

    _manager = CameraManager()
    _manager.start_all()

    yield

    # Shutdown
    if _manager:
        _manager.stop_all()


def create_app() -> FastAPI:
    app = FastAPI(title="AI Camera", lifespan=lifespan)

    # --- Auth Middleware ---
    PUBLIC_PATHS = {"/login", "/api/login", "/static/css/style.css", "/static/js/app.js"}

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path

            # Allow public paths
            if path in PUBLIC_PATHS or path.startswith("/static/"):
                return await call_next(request)

            # Check session cookie
            token = request.cookies.get("session")
            if token and get_session_user(token):
                return await call_next(request)

            # API requests get 401
            if path.startswith("/api/") or path.startswith("/stream"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            # HTML requests redirect to login
            return RedirectResponse("/login")

    app.add_middleware(AuthMiddleware)

    # --- Login/Logout ---
    LOGIN_PAGE = STATIC_DIR / "login.html"

    @app.get("/login")
    async def login_page():
        return HTMLResponse(LOGIN_PAGE.read_text())

    @app.post("/api/login")
    async def api_login(request: Request):
        client_ip = request.client.host if request.client else "unknown"

        # Rate limit check
        allowed, wait_seconds = check_rate_limit(client_ip)
        if not allowed:
            return JSONResponse(
                {"error": f"Too many login attempts. Try again in {wait_seconds}s."},
                status_code=429
            )

        body = await request.json()
        username = body.get("username", "")
        password = body.get("password", "")

        if verify_login(username, password):
            clear_attempts(client_ip)
            token = create_session(username)
            response = JSONResponse({"success": True, "username": username})
            response.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400)
            return response

        record_failed_attempt(client_ip)
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)

    @app.post("/api/logout")
    async def api_logout(request: Request):
        token = request.cookies.get("session")
        if token:
            delete_session(token)
        response = JSONResponse({"success": True})
        response.delete_cookie("session")
        return response

    @app.post("/api/change-password")
    async def api_change_password(request: Request):
        token = request.cookies.get("session")
        username = get_session_user(token) if token else None
        if not username:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)

        body = await request.json()
        old_pw = body.get("old_password", "")
        new_pw = body.get("new_password", "")

        success, message = change_password(username, old_pw, new_pw)
        if success:
            return {"message": message}
        return JSONResponse({"error": message}, status_code=400)

    # --- MJPEG Stream ---
    @app.get("/stream")
    async def mjpeg_stream(camera_id: str = ""):
        return StreamingResponse(
            _mjpeg_generator(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    async def _mjpeg_generator(camera_id: str = ""):
        target_fps = getattr(config, "STREAM_TARGET_FPS", 10)
        max_width = getattr(config, "STREAM_MAX_WIDTH", 960)
        jpeg_quality = getattr(config, "STREAM_JPEG_QUALITY", 70)
        interval = 1.0 / target_fps

        while True:
            runner = _get_runner(camera_id)
            frame, _ = runner.get_latest() if runner else (None, [])
            if frame is not None:
                small = resize_frame(frame, max_width)
                jpg = encode_jpeg(small, jpeg_quality)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                    b"\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(interval)

    # --- SSE Detection Events ---
    @app.get("/stream/events")
    async def sse_detections(request: Request, camera_id: str = ""):
        return EventSourceResponse(_detection_generator(request, camera_id))

    async def _detection_generator(request: Request, camera_id: str = ""):
        prev_detections = None

        try:
            while True:
                if await request.is_disconnected():
                    break
                runner = _get_runner(camera_id)
                if runner:
                    _, detections = runner.get_latest()
                    det_str = json.dumps(detections)
                    if det_str != prev_detections:
                        prev_detections = det_str
                        yield {
                            "event": "detections",
                            "data": json.dumps({
                                "detections": detections,
                                "fps": round(runner.fps, 1),
                                "mode": runner.mode,
                            })
                        }

                    # Emit alerts from the AI runner's smart logging
                    alerts = runner.get_pending_alerts()
                    for alert in alerts:
                        yield {
                            "event": "alert",
                            "data": json.dumps(alert)
                        }

                await asyncio.sleep(0.3)
        except (asyncio.CancelledError, Exception):
            pass

    # --- REST API ---
    @app.get("/api/status")
    async def status(camera_id: str = ""):
        cam = _get_cam(camera_id)
        runner = _get_runner(camera_id)
        # Check if camera actually has frames (not just thread running)
        has_signal = False
        if cam and cam.is_running():
            frame = cam.get_frame()
            has_signal = frame is not None
        return {
            "camera_connected": has_signal,
            "camera_type": config.CAMERA_TYPE,
            "frame_size": list(cam.frame_size) if cam and has_signal else None,
            "ai_mode": runner.mode if runner else None,
            "ai_fps": round(runner.fps, 1) if runner else 0,
            "cameras": _manager.list_cameras() if _manager else [],
            "multi_camera": _manager.is_multi if _manager else False,
        }

    @app.get("/api/cameras")
    async def list_cameras():
        return {"cameras": _manager.list_cameras() if _manager else []}

    @app.get("/api/cameras/config")
    async def get_cameras_config():
        """Get the raw camera configuration list."""
        cameras = config.CAMERAS if config.CAMERAS else []
        # If no CAMERAS list, show the current single-camera as a config entry
        if not cameras:
            cameras = [{
                "id": config.CAMERA_ID,
                "description": getattr(config, "CAMERA_DESCRIPTION", ""),
                "type": config.CAMERA_TYPE,
                "url": config.RTSP_URL,
                "transport": config.RTSP_TRANSPORT,
                "width": config.FRAME_WIDTH,
                "height": config.FRAME_HEIGHT,
                "latitude": getattr(config, "GPS_LATITUDE", 0),
                "longitude": getattr(config, "GPS_LONGITUDE", 0),
            }]
        return {"cameras": [_mask_sensitive(c) for c in cameras]}

    @app.post("/api/cameras/config")
    async def save_cameras_config(request: Request):
        """Save camera configuration. Requires restart to take effect."""
        body = await request.json()
        cameras = body.get("cameras", [])

        # Build lookup of existing cameras to preserve masked passwords
        # Read directly from overrides file to get the real (unmasked) URLs
        existing = {}
        try:
            if os.path.exists("config_overrides.json"):
                with open("config_overrides.json", 'r') as f:
                    saved = json.load(f)
                for c in saved.get("CAMERAS", []):
                    existing[c.get("id", "")] = c
                if not existing and saved.get("RTSP_URL"):
                    existing[saved.get("CAMERA_ID", config.CAMERA_ID)] = {"url": saved["RTSP_URL"]}
        except Exception:
            pass
        # Fallback to runtime config
        if not existing:
            for c in (config.CAMERAS or []):
                existing[c.get("id", "")] = c
        if not existing:
            existing[config.CAMERA_ID] = {"url": config.RTSP_URL}

        # Validate and restore masked passwords
        for i, cam in enumerate(cameras):
            if not cam.get("id"):
                return JSONResponse({"error": f"Camera {i+1} missing 'id'"}, status_code=400)

            # If URL contains masked password, restore from existing config
            url = cam.get("url", "")
            if "****" in url and cam["id"] in existing:
                cam["url"] = existing[cam["id"]].get("url", url)

            if not cam.get("url") and cam.get("type") == "rtsp":
                return JSONResponse({"error": f"Camera '{cam['id']}' missing 'url'"}, status_code=400)

            # Set defaults
            cam.setdefault("description", cam["id"])
            cam.setdefault("type", "rtsp")
            cam.setdefault("transport", "tcp")
            cam.setdefault("width", 1280)
            cam.setdefault("height", 720)
            cam.setdefault("latitude", 0.0)
            cam.setdefault("longitude", 0.0)

        # Save to config and persist
        config.CAMERAS = cameras
        overrides = {"CAMERAS": cameras}

        # Keep legacy single-camera fields in sync with the first camera
        if cameras:
            first = cameras[0]
            overrides["CAMERA_ID"] = first.get("id", "CAM_001")
            overrides["CAMERA_DESCRIPTION"] = first.get("description", "")
            overrides["CAMERA_TYPE"] = first.get("type", "rtsp")
            overrides["RTSP_URL"] = first.get("url", "")
            overrides["RTSP_TRANSPORT"] = first.get("transport", "tcp")
            overrides["FRAME_WIDTH"] = first.get("width", 1280)
            overrides["FRAME_HEIGHT"] = first.get("height", 720)
            overrides["GPS_LATITUDE"] = first.get("latitude", 0)
            overrides["GPS_LONGITUDE"] = first.get("longitude", 0)
            # Apply to runtime config too
            for k, v in overrides.items():
                if k != "CAMERAS":
                    setattr(config, k, v)

        config.save_overrides(overrides)

        return {"saved": len(cameras), "message": "Restart the server to apply changes."}

    @app.get("/api/snapshot")
    async def snapshot(camera_id: str = ""):
        cam = _get_cam(camera_id)
        frame = cam.get_frame() if cam else None
        if frame is None:
            return JSONResponse({"error": "No frame available"}, status_code=503)
        jpg = encode_jpeg(frame)
        return StreamingResponse(
            iter([jpg]),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/events")
    async def get_events(limit: int = 50, event_type: str = "all", offset: int = 0,
                         date_from: str = "", date_to: str = "", camera_id: str = ""):
        events = db_handler.get_recent_events(
            limit=limit, event_type=event_type, offset=offset,
            date_from=date_from, date_to=date_to, camera_id=camera_id
        )
        total = db_handler.get_event_count(event_type=event_type, date_from=date_from, date_to=date_to, camera_id=camera_id)
        return {"events": events, "total": total, "offset": offset, "limit": limit}

    @app.post("/api/events/delete")
    async def delete_events(request: Request):
        body = await request.json()
        ids = body.get("ids", [])
        if not ids:
            return JSONResponse({"error": "No ids provided"}, status_code=400)
        deleted = db_handler.delete_events(ids)
        return {"deleted": deleted}

    @app.post("/api/events/delete-all")
    async def delete_all_events(request: Request):
        body = await request.json()
        event_type = body.get("event_type", "all")
        date_from = body.get("date_from", "")
        date_to = body.get("date_to", "")
        deleted = db_handler.delete_all_events(event_type=event_type, date_from=date_from, date_to=date_to)
        return {"deleted": deleted}

    @app.get("/api/analytics/today")
    async def analytics_today():
        return db_handler.get_today_stats()

    @app.get("/api/analytics/hourly")
    async def analytics_hourly(hours: int = 24):
        return {"activity": db_handler.get_hourly_activity(hours=hours)}

    @app.get("/api/analytics/top-objects")
    async def analytics_top_objects(limit: int = 10):
        return {"objects": db_handler.get_top_objects(limit=limit)}

    @app.post("/api/ai/mode")
    async def set_ai_mode(request: Request):
        body = await request.json()
        mode = body.get("mode", "yolo")
        cam_id = body.get("camera_id", "")
        if mode not in ("yolo", "facenet", "both", "off"):
            return JSONResponse({"error": "Invalid mode"}, status_code=400)

        # Apply to specific camera or all cameras
        if cam_id:
            runner = _get_runner(cam_id)
            if runner:
                if mode == "off":
                    runner.stop()
                else:
                    runner.mode = mode
        elif _manager:
            for cid in _manager.camera_ids:
                runner = _manager.get_runner(cid)
                if runner:
                    if mode == "off":
                        runner.stop()
                    else:
                        runner.mode = mode
        return {"mode": mode}

    # --- Settings API ---
    SETTINGS_SCHEMA = {
        "CAMERA_ID": {"type": "string", "description": "Camera identifier for event logs", "category": "Camera"},
        "CAMERA_DESCRIPTION": {"type": "string", "description": "Camera location/description", "category": "Camera"},
        "GPS_LATITUDE": {"type": "float", "min": -90, "max": 90, "description": "Camera GPS latitude", "category": "Camera"},
        "GPS_LONGITUDE": {"type": "float", "min": -180, "max": 180, "description": "Camera GPS longitude", "category": "Camera"},
        "DEFAULT_AI_MODE": {"type": "string", "description": "Default AI mode: yolo, facenet, or both", "category": "Detection"},
        "LOG_DELAY_SECONDS": {"type": "float", "min": 0.5, "max": 60, "description": "Seconds between event logs", "category": "Detection"},
        "DETECTION_CLASSES": {"type": "list", "description": "Object classes to detect (comma-separated)", "category": "Detection"},
        "RECOGNITION_THRESHOLD": {"type": "float", "min": 0.1, "max": 2.0, "description": "Distance below which a face is 'known'", "category": "Face Recognition"},
        "REJECTION_DISTANCE": {"type": "float", "min": 0.5, "max": 3.0, "description": "Distance above which detection is rejected", "category": "Face Recognition"},
        "DETECTION_COOLDOWN_SECONDS": {"type": "float", "min": 5, "max": 600, "description": "Seconds before re-logging same detection", "category": "Detection"},
        "STREAM_JPEG_QUALITY": {"type": "int", "min": 10, "max": 100, "description": "JPEG quality for web stream", "category": "Stream"},
        "STREAM_MAX_WIDTH": {"type": "int", "min": 320, "max": 1920, "description": "Max width of streamed frames", "category": "Stream"},
        "STREAM_TARGET_FPS": {"type": "int", "min": 1, "max": 30, "description": "Target FPS for web stream", "category": "Stream"},
        "ALERT_ENABLED": {"type": "bool", "description": "Enable/disable alert notifications", "category": "Alerts"},
        "ALERT_EVENTS": {"type": "list", "description": "Alert types: unknown_face, person_detected, known_face", "category": "Alerts"},
    }

    READONLY_SETTINGS = {
        "CAMERA_TYPE": {"type": "string", "description": "Camera backend type", "category": "Camera"},
        "RTSP_URL": {"type": "string", "description": "RTSP stream URL", "category": "Camera"},
        "FRAME_WIDTH": {"type": "int", "description": "Camera frame width", "category": "Camera"},
        "FRAME_HEIGHT": {"type": "int", "description": "Camera frame height", "category": "Camera"},
        "WEB_HOST": {"type": "string", "description": "Web server host", "category": "Server"},
        "WEB_PORT": {"type": "int", "description": "Web server port", "category": "Server"},
        "FACENET_MODEL_PATH": {"type": "string", "description": "FaceNet model file", "category": "Face Recognition"},
        "DB_NAME": {"type": "string", "description": "Database filename", "category": "Storage"},
    }

    @app.get("/api/settings")
    async def get_settings():
        runtime = {}
        for key, schema in SETTINGS_SCHEMA.items():
            val = getattr(config, key, None)
            runtime[key] = {**schema, "value": val}

        readonly = {}
        for key, schema in READONLY_SETTINGS.items():
            val = getattr(config, key, None)
            entry = {**schema, "value": val}
            # Mask sensitive URLs
            if key == "RTSP_URL" and isinstance(val, str):
                entry["value"] = _mask_url(val)
            readonly[key] = entry

        return {"runtime": runtime, "readonly": readonly}

    @app.post("/api/settings")
    async def update_settings(request: Request):
        body = await request.json()
        updated = {}
        errors = {}

        for key, value in body.items():
            if key not in SETTINGS_SCHEMA:
                errors[key] = "Not a runtime-changeable setting"
                continue

            schema = SETTINGS_SCHEMA[key]
            try:
                # Type conversion
                if schema["type"] == "float":
                    value = float(value)
                elif schema["type"] == "int":
                    value = int(value)
                elif schema["type"] == "bool":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    else:
                        value = bool(value)
                elif schema["type"] == "string":
                    value = str(value).strip()
                elif schema["type"] == "list":
                    if isinstance(value, str):
                        value = [v.strip() for v in value.split(",") if v.strip()]

                # Range validation
                if "min" in schema and value < schema["min"]:
                    errors[key] = f"Must be >= {schema['min']}"
                    continue
                if "max" in schema and value > schema["max"]:
                    errors[key] = f"Must be <= {schema['max']}"
                    continue

                setattr(config, key, value)
                updated[key] = value

            except (ValueError, TypeError) as e:
                errors[key] = f"Invalid value: {e}"

        # Persist changes to disk
        if updated:
            config.save_overrides(updated)

        if errors:
            return JSONResponse({"updated": updated, "errors": errors}, status_code=207)
        return {"updated": updated}

    # --- Face Enrollment API ---
    faces_dir = Path(config.FACE_IMAGE_BASE_DIR)
    faces_dir.mkdir(parents=True, exist_ok=True)
    faces_json = Path(config.KNOWN_FACES_DB)

    def _read_faces_json() -> dict:
        if faces_json.exists():
            with open(faces_json, 'r') as f:
                return json.load(f)
        return {}

    def _write_faces_json(data: dict) -> None:
        with open(faces_json, 'w') as f:
            json.dump(data, f, indent=2)

    @app.get("/api/faces")
    async def list_faces():
        with _faces_lock:
            db = _read_faces_json()
        people = []
        for name, images in db.items():
            people.append({
                "name": name,
                "images": images,
                "image_count": len(images),
                "thumbnail": f"/images/faces/{images[0]}" if images else None,
            })
        return {"people": people}

    @app.post("/api/faces/enroll")
    async def enroll_face(name: str = Form(...), files: list[UploadFile] = File(...),
                          crop_faces: bool = Form(False)):
        import cv2
        import numpy as np

        if not name or not name.strip():
            return JSONResponse({"error": "Name is required"}, status_code=400)
        name = name.strip()

        # Load Haar Cascade for face cropping
        face_detector = None
        if crop_faces:
            cascade_path = "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_detector = cv2.CascadeClassifier(cascade_path)

        saved_files = []
        skipped = 0
        for i, file in enumerate(files):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            content = await file.read()

            if crop_faces and face_detector is not None:
                # Decode image and detect faces
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    skipped += 1
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                if len(faces) == 0:
                    skipped += 1
                    continue

                # Save each detected face as a separate cropped image
                for j, (x, y, w, h) in enumerate(faces):
                    # Add padding around the face (20%)
                    pad = int(max(w, h) * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(img.shape[1], x + w + pad)
                    y2 = min(img.shape[0], y + h + pad)
                    face_crop = img[y1:y2, x1:x2]

                    filename = f"enroll_{name}_{ts}_{i}_face{j}.jpg"
                    filepath = faces_dir / filename
                    cv2.imwrite(str(filepath), face_crop)
                    saved_files.append(filename)
            else:
                # Save original image as-is
                ext = Path(file.filename).suffix or ".jpg"
                filename = f"enroll_{name}_{ts}_{i}{ext}"
                filepath = faces_dir / filename
                with open(filepath, "wb") as f:
                    f.write(content)
                saved_files.append(filename)

        if not saved_files:
            return JSONResponse(
                {"error": f"No faces detected in uploaded images ({skipped} skipped)"},
                status_code=400
            )

        with _faces_lock:
            db = _read_faces_json()
            if name not in db:
                db[name] = []
            db[name].extend(saved_files)
            _write_faces_json(db)
            # Reload embeddings
            face_rec.load_known_faces_from_images()

        return {"name": name, "added": saved_files, "total_images": len(db[name])}

    @app.delete("/api/faces/{name}")
    async def delete_person(name: str):
        with _faces_lock:
            db = _read_faces_json()
            if name not in db:
                return JSONResponse({"error": f"Person '{name}' not found"}, status_code=404)

            # Delete image files
            for img in db[name]:
                img_path = faces_dir / img
                if img_path.exists():
                    img_path.unlink()

            del db[name]
            _write_faces_json(db)
            face_rec.load_known_faces_from_images()

        return {"deleted": name}

    @app.delete("/api/faces/{name}/image/{filename}")
    async def delete_person_image(name: str, filename: str):
        with _faces_lock:
            db = _read_faces_json()
            if name not in db:
                return JSONResponse({"error": f"Person '{name}' not found"}, status_code=404)
            if filename not in db[name]:
                return JSONResponse({"error": f"Image '{filename}' not found"}, status_code=404)

            db[name].remove(filename)
            img_path = faces_dir / filename
            if img_path.exists():
                img_path.unlink()

            # Remove person if no images left
            if not db[name]:
                del db[name]

            _write_faces_json(db)
            face_rec.load_known_faces_from_images()

        return {"name": name, "deleted_image": filename}

    # --- Serve event images ---
    event_img_dir = Path(config.EVENT_IMAGE_DIR)
    roi_img_dir = Path(config.ROI_OUTPUT_DIR)
    for d in (event_img_dir, roi_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    app.mount("/images/faces", StaticFiles(directory=str(faces_dir)), name="face_images")
    app.mount("/images/events", StaticFiles(directory=str(event_img_dir)), name="event_images")
    app.mount("/images/roi", StaticFiles(directory=str(roi_img_dir)), name="roi_images")

    # --- Serve frontend ---
    @app.get("/")
    async def index():
        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(index_file.read_text())

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
