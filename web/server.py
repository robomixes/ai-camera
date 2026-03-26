import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import config
import db_handler
from camera import create_camera
from camera.base import CameraBase
from web.ai_runner import AIRunner
from web.utils import encode_jpeg, resize_frame

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

    # --- Serve event images ---
    import os
    event_img_dir = Path(config.EVENT_IMAGE_DIR)
    roi_img_dir = Path(config.ROI_OUTPUT_DIR)
    for d in (event_img_dir, roi_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    app.mount("/images/events", StaticFiles(directory=str(event_img_dir)), name="event_images")
    app.mount("/images/roi", StaticFiles(directory=str(roi_img_dir)), name="roi_images")

    # --- Serve frontend ---
    @app.get("/")
    async def index():
        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(index_file.read_text())

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
