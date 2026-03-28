"""License Plate Reader using EasyOCR with optional YOLO plate detection."""
import logging
import os
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateReader:
    """Reads license plates from vehicle crop images using EasyOCR."""

    def __init__(self):
        self._loaded = False
        self._reader = None
        self._plate_model = None

    def load(self) -> bool:
        """Initialize EasyOCR and optionally load plate detection model."""
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._loaded = True
            logger.info("EasyOCR plate reader loaded.")
        except ImportError:
            logger.error("easyocr not installed. Run: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            return False

        # Try to load dedicated plate detection model
        self._load_plate_model()
        return True

    def _load_plate_model(self):
        """Load YOLO plate detection model if available."""
        import config
        model_path = getattr(config, "ANPR_PLATE_MODEL_PATH", "yolov8n-plate.pt")
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self._plate_model = YOLO(model_path)
                logger.info(f"Plate detection model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load plate detection model: {e}")
                self._plate_model = None
        else:
            logger.info(f"No plate detection model at {model_path} — using crop fallback")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def read_plate(self, vehicle_crop: np.ndarray) -> dict | None:
        """
        Read license plate from a vehicle crop image.
        If plate detection model is available, detects exact plate region first.
        Otherwise falls back to cropping lower 40% of vehicle.
        Returns {"text": "ABC1234", "confidence": 0.85} or None.
        """
        if not self._loaded or self._reader is None:
            return None

        try:
            h, w = vehicle_crop.shape[:2]
            if h < 20 or w < 40:
                return None

            # Step 1: Find the plate region
            plate_region = self._detect_plate_region(vehicle_crop)

            # Step 2: Preprocess for OCR
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            # Resize to standard height for consistent OCR
            ph, pw = gray.shape
            if ph < 50:
                scale = 100 / max(ph, 1)
                gray = cv2.resize(gray, (int(pw * scale), 100))
            gray = cv2.equalizeHist(gray)

            # Step 3: Run EasyOCR
            results = self._reader.readtext(gray, detail=1, paragraph=False)

            # Step 4: Find best plate-like text
            best = None
            best_conf = 0.0
            for bbox, text, conf in results:
                cleaned = self._clean_plate_text(text)
                if self._is_valid_plate(cleaned) and conf > 0.3 and conf > best_conf:
                    best = cleaned
                    best_conf = conf

            if best:
                return {"text": best, "confidence": round(best_conf, 3)}

        except Exception as e:
            logger.error(f"Plate read error: {e}")

        return None

    def _detect_plate_region(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Detect exact plate region using YOLO model, or fallback to lower 40% crop."""
        h, w = vehicle_crop.shape[:2]

        # Try YOLO plate detection if model available
        if self._plate_model is not None:
            try:
                results = self._plate_model(vehicle_crop, verbose=False, conf=0.3)
                for r in results:
                    if len(r.boxes) > 0:
                        # Take highest confidence plate detection
                        best_box = r.boxes[0]
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                        # Add small padding
                        pad = 5
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        plate = vehicle_crop[y1:y2, x1:x2]
                        if plate.size > 0:
                            return plate
            except Exception as e:
                logger.debug(f"Plate detection failed, using fallback: {e}")

        # Fallback: crop lower 40% of vehicle
        return vehicle_crop[int(h * 0.6):, :]

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Clean OCR result to plate-like characters."""
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    @staticmethod
    def _is_valid_plate(text: str) -> bool:
        """Check if text looks like a license plate."""
        if not text or len(text) < 4 or len(text) > 10:
            return False
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        return has_letter and has_digit
