"""License Plate Reader using Tesseract OCR."""
import logging
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateReader:
    """Reads license plates from vehicle crop images using Tesseract OCR."""

    def __init__(self):
        self._loaded = False
        self._pytesseract = None

    def load(self) -> bool:
        """Initialize Tesseract. Returns True if available."""
        try:
            import pytesseract
            # Verify tesseract binary is accessible
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self._loaded = True
            logger.info("Tesseract OCR loaded successfully.")
            return True
        except ImportError:
            logger.error("pytesseract not installed. Run: pip install pytesseract")
            return False
        except Exception as e:
            logger.error(f"Tesseract binary not found: {e}. Install: apt install tesseract-ocr")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def read_plate(self, vehicle_crop: np.ndarray) -> dict | None:
        """
        Read license plate from a vehicle crop image.
        Returns {"text": "ABC1234", "confidence": 0.85} or None.
        """
        if not self._loaded or self._pytesseract is None:
            return None

        try:
            h, w = vehicle_crop.shape[:2]
            if h < 20 or w < 40:
                return None

            # Crop lower 40% of vehicle (where plate usually is)
            plate_region = vehicle_crop[int(h * 0.6):, :]

            # Preprocess for OCR
            processed = self._preprocess(plate_region)

            # Run Tesseract
            custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            # Get detailed data for confidence
            data = self._pytesseract.image_to_data(
                processed, config=custom_config, output_type=self._pytesseract.Output.DICT
            )

            # Extract text and confidence
            texts = []
            confidences = []
            for i, conf in enumerate(data["conf"]):
                conf = int(conf)
                if conf > 30:  # minimum confidence threshold
                    text = data["text"][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf)

            if not texts:
                return None

            plate_text = self._clean_plate_text("".join(texts))
            if not self._is_valid_plate(plate_text):
                return None

            avg_conf = sum(confidences) / len(confidences) / 100.0

            return {"text": plate_text, "confidence": round(avg_conf, 2)}

        except Exception as e:
            logger.error(f"Plate read error: {e}")
            return None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess plate region for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to standard height for consistent OCR
        h, w = gray.shape
        if h < 50:
            scale = 100 / h
            gray = cv2.resize(gray, (int(w * scale), 100))

        # Histogram equalization for contrast
        gray = cv2.equalizeHist(gray)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological close to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh

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
        # Must have at least one letter and one digit
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        return has_letter and has_digit
