import threading
import time
import os
from datetime import datetime
import numpy as np
import cv2

import config
import ai_features
import db_handler
import face_recognition as face_rec
from camera.base import CameraBase


class AIRunner:
    """Background thread that runs AI detection on camera frames and logs events."""

    def __init__(self, cam: CameraBase, mode: str = "yolo"):
        self._cam = cam
        self._mode = mode
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_detections: list = []
        self._fps: float = 0.0
        self._last_log_time: float = 0.0

    def start(self) -> None:
        if self._mode == "facenet":
            if not face_rec.initialize_system():
                print("Warning: FaceNet initialization failed. Falling back to YOLO.")
                self._mode = "yolo"

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"AI Runner started (mode: {self._mode})")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("AI Runner stopped.")

    def get_latest(self) -> tuple[np.ndarray | None, list]:
        with self._lock:
            return self._latest_frame, self._latest_detections.copy()

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value in ("facenet", "both") and self._mode not in ("facenet", "both"):
            if not face_rec.initialize_system():
                print("Warning: FaceNet initialization failed.")
                return
        self._mode = value
        print(f"AI Runner mode switched to: {value}")

    def _run_loop(self) -> None:
        frame_count = 0
        fps_start = time.time()

        while self._running:
            frame_bgr = self._cam.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            try:
                if self._mode == "yolo":
                    annotated, detections = self._run_yolo(frame_bgr)
                elif self._mode == "facenet":
                    annotated, detections = self._run_facenet(frame_bgr)
                elif self._mode == "both":
                    annotated, detections = self._run_both(frame_bgr)
                else:
                    annotated, detections = frame_bgr, []

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_detections = detections

                # Log events (throttled)
                if detections and (time.time() - self._last_log_time >= config.LOG_DELAY_SECONDS):
                    self._log_event(annotated, detections)
                    self._last_log_time = time.time()

                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    with self._lock:
                        self._fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                print(f"AI Runner error: {e}")
                time.sleep(0.1)

    def _run_yolo(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        analyzed_frame, detected_data = ai_features.run_yolov8_detection(
            frame_bgr, self._cam.frame_size,
            roi=None, classes_filter=config.DETECTION_CLASSES
        )
        detections = []
        for item in detected_data:
            if isinstance(item, tuple) and len(item) == 2:
                detections.append({"label": item[0], "confidence": round(item[1], 3)})
        return analyzed_frame, detections

    def _run_facenet(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        analyzed_frame, detected_data = face_rec.run_facenet_recognition(
            frame_bgr, self._cam.frame_size
        )
        face_rec.process_deferred_logs()

        # detected_data is list of (display_name, 1.0 - distance) tuples
        detections = []
        for item in detected_data:
            if isinstance(item, dict):
                detections.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                name = item[0]
                confidence = round(item[1], 3)  # 1.0 - distance (higher = better match)
                detections.append({"name": name, "confidence": confidence})
        return analyzed_frame, detections

    def _run_both(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        """Run YOLO and FaceNet on the same frame, merge results."""
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        # YOLO on clean frame — returns annotated frame with object boxes
        yolo_frame, yolo_dets = ai_features.run_yolov8_detection(
            frame_bgr.copy(), self._cam.frame_size,
            roi=None, classes_filter=config.DETECTION_CLASSES
        )

        # FaceNet on clean frame — returns annotated frame with face boxes
        face_frame, face_data = face_rec.run_facenet_recognition(
            frame_bgr.copy(), self._cam.frame_size
        )
        face_rec.process_deferred_logs()

        # Simple merge: draw face_frame on top of yolo_frame using addWeighted
        # This preserves both sets of annotations clearly
        combined = cv2.addWeighted(yolo_frame, 0.5, face_frame, 0.5, 0)
        # Re-overlay annotations by keeping the brighter pixels from either frame
        # (annotations are drawn in bright colors on dark background)
        combined = np.maximum(yolo_frame, face_frame)

        # Merge detections
        detections = []
        for item in yolo_dets:
            if isinstance(item, tuple) and len(item) == 2:
                detections.append({"type": "yolo", "label": item[0], "confidence": round(item[1], 3)})

        for item in face_data:
            if isinstance(item, dict):
                item["type"] = "face"
                detections.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                detections.append({"type": "face", "name": item[0], "confidence": round(item[1], 3)})

        return combined, detections

    def _log_event(self, frame: np.ndarray, detections: list) -> None:
        """Save an event image and log to database."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Skip black/empty frames
            if frame is None or frame.mean() < 1.0:
                return

            # Save snapshot
            img_dir = config.ROI_OUTPUT_DIR
            os.makedirs(img_dir, exist_ok=True)
            image_filename = f"det_{timestamp}.jpg"
            cv2.imwrite(os.path.join(img_dir, image_filename), frame)

            # Log YOLO detections
            yolo_dets = [d for d in detections if d.get("type") == "yolo" or ("label" in d and "type" not in d)]
            for det in yolo_dets:
                label = det.get("label", "unknown")
                conf = det.get("confidence", 0)
                db_handler.log_detection(
                    detection_data=[{"class": label, "confidence": conf}],
                    roi_area=None,
                    image_filename=image_filename
                )

            # Log face detections (in "both" mode, face_rec.process_deferred_logs handles its own logging,
            # but we also log faces that appear in the detections list)
            face_dets = [d for d in detections if d.get("type") == "face" or "name" in d]
            for det in face_dets:
                name = det.get("name", "Unknown")
                confidence = det.get("confidence", 0)
                distance = 1.0 - confidence  # convert back: confidence = 1.0 - distance
                is_known = name != "Unknown" and distance < config.RECOGNITION_THRESHOLD
                db_handler.log_face_detection_event(
                    name=name,
                    distance=distance,
                    image_filename=image_filename,
                    is_known=is_known
                )

        except Exception as e:
            print(f"AI Runner logging error: {e}")
