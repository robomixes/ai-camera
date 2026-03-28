# AI-Camera

## AI-Powered Multi-Camera Surveillance System

AI-Camera is an open-source Python surveillance system with real-time AI detection, face recognition, license plate reading, and a web dashboard. Runs on Raspberry Pi and PC.

---

## Features

- **Object Detection** — YOLOv8 real-time detection (person, vehicle, etc.)
- **Face Recognition** — FaceNet with multi-frame aggregation for accuracy
- **License Plate Recognition (ANPR)** — EasyOCR with optional YOLO plate detection model
- **Multi-Camera** — manage multiple RTSP/IP cameras from one dashboard
- **Web Dashboard** — live MJPEG stream, analytics, event browser, notifications
- **Face Enrollment** — upload photos with auto-crop face detection
- **Plate Watchlist** — alert on specific license plates
- **Smart Logging** — per-detection cooldown, no event spam
- **Notifications** — browser notifications, sound alerts, toast popups
- **Multi-User RBAC** — admin, operator, viewer roles
- **Settings Management** — runtime-configurable from dashboard
- **Event Export** — CSV/JSON download with filters
- **Health Monitoring** — `/api/health` endpoint for uptime checks
- **Docker Support** — single command deployment

---

## Quick Start

### Option 1: Direct Install (Recommended for Pi)

**Windows:**
```bash
git clone https://github.com/robomixes/ai-camera.git
cd ai-camera
setup.bat
venv\Scripts\activate
python run_web.py
```

**Linux / Raspberry Pi:**
```bash
git clone https://github.com/robomixes/ai-camera.git
cd ai-camera
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python run_web.py
```

### Option 2: Docker

```bash
git clone https://github.com/robomixes/ai-camera.git
cd ai-camera
docker-compose up -d
```

Camera can be configured either way:
- **Before starting:** set `RTSP_URL` in `docker-compose.yml` or as env var
- **After starting:** login to dashboard > Parameters > Camera Management > add camera URL > save > `docker-compose restart`

### Access Dashboard

Open **http://localhost:8080**

Default login: `admin` / `admin` (change immediately)

---

## Configuration

### Camera Setup

**Single camera** — edit `config.py`:
```python
CAMERA_TYPE = "rtsp"
RTSP_URL = "rtsp://admin:password@192.168.1.9/h264Preview_01_sub"
```

**Multiple cameras** — configure from Dashboard > Parameters > Camera Management, or edit `config.py`:
```python
CAMERAS = [
    {"id": "CAM_001", "description": "Front Gate", "type": "rtsp",
     "url": "rtsp://admin:pass@192.168.1.9/h264Preview_01_sub",
     "ai_mode": "both"},
    {"id": "CAM_002", "description": "Back Door", "type": "rtsp",
     "url": "rtsp://admin:pass@192.168.1.10/h264Preview_01_sub",
     "ai_mode": "yolo"},
]
```

### Environment Variables (Docker)

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_TYPE` | `auto` | `auto`, `rtsp`, or `picamera` |
| `RTSP_URL` | `` | RTSP stream URL |
| `WEB_PORT` | `8080` | Dashboard port |
| `DEFAULT_AI_MODE` | `yolo` | `yolo`, `facenet`, or `both` |
| `ANPR_ENABLED` | `false` | Enable plate recognition |
| `STREAM_TARGET_FPS` | `10` | Web stream FPS |

### Runtime Settings

All settings configurable from Dashboard > Parameters > Runtime Settings:
- Detection classes, thresholds, cooldowns
- Stream quality (FPS, resolution, JPEG quality)
- Face recognition thresholds
- ANPR settings
- Alert configuration
- Data retention policy

---

## AI Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **YOLO** | Object detection only | General surveillance |
| **FaceNet** | Face recognition only | Access control |
| **Both** | YOLO + FaceNet combined | Full security |

**ANPR** runs as a toggle alongside any mode — auto-adds vehicle classes to YOLO when enabled.

---

## User Roles

| Role | Stream | Events | Faces/Plates | Settings | Users |
|------|:------:|:------:|:------------:|:--------:|:-----:|
| Admin | Yes | Yes | Yes | Yes | Yes |
| Operator | Yes | Yes | Yes | No | No |
| Viewer | Yes | Yes | No | No | No |

---

## API Documentation

Interactive Swagger docs available at `/docs` (requires login).

Key endpoints:
- `GET /stream` — MJPEG live stream
- `GET /stream/events` — SSE detection events
- `GET /api/health` — health check (public)
- `GET /api/events` — query events
- `GET /api/events/export` — download CSV/JSON
- `POST /api/faces/enroll` — enroll face
- `GET /api/plates/watchlist` — plate watchlist
- `POST /api/ai/mode` — switch AI mode

---

## Architecture

```
Camera (RTSP/Picamera) → Frame Buffer → AI Runner → Web Dashboard
                                            |
                                    YOLO + FaceNet + ANPR
                                            |
                                    SQLite Events DB
                                            |
                                    Alerts + Notifications
```

---

## Requirements

- Python 3.11+
- OpenCV, NumPy, Ultralytics (YOLOv8)
- FastAPI, Uvicorn
- EasyOCR (for ANPR)
- TensorFlow Lite (for FaceNet, Pi only)
- Bcrypt (authentication)

---

## Raspberry Pi Notes

- Use sub-stream (`h264Preview_01_sub`) for lower resolution/bandwidth
- Set `STREAM_TARGET_FPS=5` and `STREAM_MAX_WIDTH=640` for Pi performance
- Install Pi-specific deps: `pip install -r requirements-pi.txt`
- FaceNet uses TFLite for efficient ARM inference
- "Both" mode is slower — use YOLO or FaceNet individually on Pi

---

## License

Open source. See repository for details.
