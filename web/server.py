import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import config
import db_handler
import face_recognition as face_rec
from camera import create_camera
from camera.base import CameraBase
from web.ai_runner import AIRunner
from web.utils import encode_jpeg, resize_frame

_faces_lock = threading.Lock()

# --- Globals set during lifespan ---
_cam: CameraBase | None = None
_ai_runner: AIRunner | None = None

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cam, _ai_runner

    # Startup
    db_handler.initialize_db()

    _cam = create_camera(
        camera_type=config.CAMERA_TYPE,
        url=config.RTSP_URL,
        transport=config.RTSP_TRANSPORT,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
    )
    _cam.start()

    # Wait for first frame
    print("Web: Waiting for camera stream...")
    for _ in range(150):
        if _cam.get_frame() is not None:
            print("Web: Camera stream ready.")
            break
        await asyncio.sleep(0.1)
    else:
        print("Web: WARNING - No frames received after 15s.")

    _ai_runner = AIRunner(_cam, mode="yolo")
    _ai_runner.start()

    yield

    # Shutdown
    if _ai_runner:
        _ai_runner.stop()
    if _cam:
        _cam.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="AI Camera", lifespan=lifespan)

    # --- MJPEG Stream ---
    @app.get("/stream")
    async def mjpeg_stream():
        return StreamingResponse(
            _mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    async def _mjpeg_generator():
        target_fps = getattr(config, "STREAM_TARGET_FPS", 10)
        max_width = getattr(config, "STREAM_MAX_WIDTH", 960)
        jpeg_quality = getattr(config, "STREAM_JPEG_QUALITY", 70)
        interval = 1.0 / target_fps

        while True:
            frame, _ = _ai_runner.get_latest() if _ai_runner else (None, [])
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
    async def sse_detections(request: Request):
        return EventSourceResponse(_detection_generator(request))

    async def _detection_generator(request: Request):
        prev_detections = None
        while True:
            if await request.is_disconnected():
                break
            if _ai_runner:
                _, detections = _ai_runner.get_latest()
                # Only send when detections change
                det_str = json.dumps(detections)
                if det_str != prev_detections:
                    prev_detections = det_str
                    yield {
                        "event": "detections",
                        "data": json.dumps({
                            "detections": detections,
                            "fps": round(_ai_runner.fps, 1),
                            "mode": _ai_runner.mode,
                        })
                    }
            await asyncio.sleep(0.3)

    # --- REST API ---
    @app.get("/api/status")
    async def status():
        return {
            "camera_connected": _cam.is_running() if _cam else False,
            "camera_type": config.CAMERA_TYPE,
            "frame_size": list(_cam.frame_size) if _cam else None,
            "ai_mode": _ai_runner.mode if _ai_runner else None,
            "ai_fps": round(_ai_runner.fps, 1) if _ai_runner else 0,
        }

    @app.get("/api/snapshot")
    async def snapshot():
        frame = _cam.get_frame() if _cam else None
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
                         date_from: str = "", date_to: str = ""):
        events = db_handler.get_recent_events(
            limit=limit, event_type=event_type, offset=offset,
            date_from=date_from, date_to=date_to
        )
        total = db_handler.get_event_count(event_type=event_type, date_from=date_from, date_to=date_to)
        return {"events": events, "total": total, "offset": offset, "limit": limit}

    @app.post("/api/events/delete")
    async def delete_events(request: Request):
        body = await request.json()
        ids = body.get("ids", [])
        if not ids:
            return JSONResponse({"error": "No ids provided"}, status_code=400)
        deleted = db_handler.delete_events(ids)
        return {"deleted": deleted}

    @app.post("/api/ai/mode")
    async def set_ai_mode(request: Request):
        body = await request.json()
        mode = body.get("mode", "yolo")
        if mode not in ("yolo", "facenet", "both", "off"):
            return JSONResponse({"error": "Invalid mode"}, status_code=400)
        if _ai_runner:
            if mode == "off":
                _ai_runner.stop()
            else:
                _ai_runner.mode = mode
        return {"mode": mode}

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
