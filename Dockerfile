# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download EasyOCR models so first run doesn't wait
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" 2>/dev/null || true

# ===== Stage 2: Runtime =====
FROM python:3.11-slim

# System dependencies for OpenCV + Tesseract fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
# Copy EasyOCR models from builder
COPY --from=builder /root/.EasyOCR /root/.EasyOCR

WORKDIR /app

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p logs event_images roi_events plate_images output_images \
    people_search_queue/ready data

# Expose web port
EXPOSE 8080

# Environment defaults
ENV CAMERA_TYPE=auto \
    RTSP_URL="" \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OPENCV_FFMPEG_LOGLEVEL=quiet

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

CMD ["python", "run_web.py"]
