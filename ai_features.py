import cv2
import numpy as np
from ultralytics import YOLO
# Removed import config, assuming it's imported in ai-general.py only

# Initialize YOLOv8 model once when the module is loaded
try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    print("YOLOv8n model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8n model: {e}")
    YOLO_MODEL = None


# --- Helper function to check if a point is inside the ROI rectangle ---
def is_inside_roi(center_x, center_y, roi):
    """Checks if a point (center_x, center_y) is inside the ROI rectangle (x, y, w, h)."""
    if roi is None or len(roi) != 4:
        return False
    
    roi_x, roi_y, roi_w, roi_h = roi
    
    return (roi_x <= center_x <= roi_x + roi_w) and \
           (roi_y <= center_y <= roi_y + roi_h)


def run_yolov8_detection(frame, frame_size, roi=None, classes_filter=None):
    """
    Performs general object detection and conditionally filters results based on the ROI 
    and the list of desired classes.
    
    Returns:
        np.array: The frame with filtered bounding boxes and labels drawn.
        list: A list of unique detected object data: [(label, confidence), ...]. <-- UPDATED RETURN
    """
    if YOLO_MODEL is None:
        cv2.putText(frame, 'YOLOv8 Failed to Load', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame, []

    # --- Filter YOLO results using the 'classes' argument if classes_filter is provided ---
    if classes_filter:
        model_names = YOLO_MODEL.names
        filter_ids = [k for k, v in model_names.items() if v in classes_filter]
        results = YOLO_MODEL(frame, stream=True, verbose=False, classes=filter_ids) 
    else:
        results = YOLO_MODEL(frame, stream=True, verbose=False) 
    
    # List to store tuples of (label, confidence) for objects outside the ROI
    filtered_detected_data = []

    annotated_frame = frame.copy()

    # Scale text/line thickness based on frame width
    h_frame, w_frame = frame.shape[:2]
    scale = max(w_frame / 640, 0.5)  # normalize to 640px baseline
    font_scale = 0.4 * scale
    thickness = max(1, int(1 * scale))
    circle_r = max(3, int(4 * scale))

    # Define colors for drawing
    FILTERED_COLOR = (0, 255, 0) # Green for unfiltered (tracked)
    IGNORED_COLOR = (0, 0, 255)  # Red for ignored (inside ROI)

    for r in results:
        names = r.names
        
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            class_id = int(box.cls[0])
            label = names[class_id]
            conf = float(box.conf[0]) # Ensure confidence is float for logging

            
            # --- ROI CHECK (Only applies if roi is provided) ---
            line_h = int(18 * scale)  # line height for stacked text

            if roi is not None and is_inside_roi(center_x, center_y, roi):
                color = IGNORED_COLOR
                cv2.circle(annotated_frame, (center_x, center_y), circle_r, color, -1)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, "IGNORED", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            else:
                color = FILTERED_COLOR
                filtered_detected_data.append((label, conf))
                cv2.circle(annotated_frame, (center_x, center_y), circle_r, color, -1)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - line_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                cv2.putText(annotated_frame, f"{conf:.0%}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        break

    # Draw the ROI boundary only if it was passed (i.e., filtered mode)
    if roi is not None:
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), thickness + 1)
        cv2.putText(annotated_frame, "ROI (IGNORED)", (roi_x + 5, roi_y + int(20 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

    # Info text removed — dashboard sidebar shows detections
    
    return annotated_frame, filtered_detected_data # <--- RETURN LIST OF TUPLES