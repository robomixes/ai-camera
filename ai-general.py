import cv2
from picamera2 import Picamera2
import time
from datetime import datetime
import os
import sys
import numpy as np 
# --- For Timed Menu ---
import select 

# --- Imports (Ensure config, ai_features, and db_handler exist) ---
import config 
import ai_features
import db_handler
import face_recognition 
# --------------------------------

# --- Configuration & Folder Setup ---
ROI_CONFIG_FILE = "roi_config.txt" 

# Create necessary output directories (using config paths)
DIRS_TO_CREATE = [
    config.OUTPUT_DIR, 
    config.ROI_OUTPUT_DIR, 
    config.FACE_IMAGE_BASE_DIR, 
    db_handler.EVENT_IMAGE_DIR  
]

for d in DIRS_TO_CREATE:
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            print(f"Error creating directory {d}: {e}")
            sys.exit(1) 

# --- Global variable to store the selected ROI ---
GLOBAL_ROI = None 

# ----------------------------------------------------------------------
## 💾 Functions: Load and Save ROI (Persistence) 
# ----------------------------------------------------------------------
def load_roi():
    """Loads ROI coordinates from the config file at startup."""
    global GLOBAL_ROI
    try:
        if os.path.exists(ROI_CONFIG_FILE):
            with open(ROI_CONFIG_FILE, 'r') as f:
                coords = f.read().strip().split(',')
                if len(coords) == 4:
                    GLOBAL_ROI = tuple(map(int, coords))
                    print(f"Loaded persistent ROI: {GLOBAL_ROI}")
                    return GLOBAL_ROI
        
        GLOBAL_ROI = None
        return None
    except Exception as e:
        print(f"Error loading ROI from file: {e}. Resetting ROI to None.")
        GLOBAL_ROI = None
        return None

def save_roi(roi):
    """Saves the current ROI coordinates to the config file."""
    try:
        if roi is None:
            if os.path.exists(ROI_CONFIG_FILE):
                os.remove(ROI_CONFIG_FILE)
            print("ROI cleared and configuration file removed.")
            return

        roi_str = f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}"
        with open(ROI_CONFIG_FILE, 'w') as f:
            f.write(roi_str)
        print(f"ROI saved persistently to {ROI_CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving ROI to file: {e}")
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
## 📷 Camera/Capture Functions 
# ----------------------------------------------------------------------
def initialize_camera():
    """Initializes and returns the Picamera2 object."""
    try:
        picam2 = Picamera2()
        frame_size = (1280, 720)
        config_picam = picam2.create_video_configuration(main={"size": frame_size, "format": "RGB888"})
        picam2.configure(config_picam)
        picam2.start()
        print("Camera started.")
        time.sleep(1)
        return picam2, frame_size
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        if "Pipeline handler in use by another process" in str(e):
            print("TIP: The camera is likely in use by another program. Please kill that process or reboot the Pi.")
        
        return None, None 
        
def capture_single_image(picam2, frame_size):
    """Captures a single image."""
    print("Capturing a single image...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(config.OUTPUT_DIR, f"image_{timestamp}.jpg")
    
    frame = picam2.capture_array()
    frame_bgr = frame[:, :, ::-1] 
    
    success = cv2.imwrite(output_filename, frame_bgr)
    
    if success:
        print(f"✅ Image saved to: {output_filename}")
    else:
        print(f"❌ Error saving image to: {output_filename}")
        
    if config.ENABLE_GUI_DISPLAY:
        cv2.imshow('Image Captured', frame_bgr)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()

def capture_timed_images(picam2, frame_size, interval_seconds=1.0):
    """Captures an image every 'interval_seconds'."""
    print(f"\n--- Timed Capture Started ---")
    if config.ENABLE_GUI_DISPLAY:
        print("Press the **q** key while the preview window is focused to **STOP**.")
    else:
        print("Running headless. Press Ctrl+C to STOP.")

    last_capture_time = time.time() - interval_seconds
    capture_count = 0

    try:
        while True:
            current_time = time.time()
            frame = picam2.capture_array()
            frame_bgr = frame[:, :, ::-1]
            
            if config.ENABLE_GUI_DISPLAY:
                cv2.imshow('Timed Capture - Press q to STOP', frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13: 
                    print("\nStop key pressed. Stopping timed capture...")
                    break
            else:
                 time.sleep(0.01) # Small delay to prevent 100% CPU utilization in headless mode
                
            if current_time - last_capture_time >= interval_seconds:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = os.path.join(config.OUTPUT_DIR, f"timed_img_{timestamp}_{capture_count:04d}.jpg")
                
                success = cv2.imwrite(output_filename, frame_bgr)
                if success:
                    print(f"Captured: {output_filename}")
                    capture_count += 1
                else:
                    print(f"Error saving timed image: {output_filename}")
                    
                last_capture_time = current_time
                
    except KeyboardInterrupt:
        print("\nTimed capture interrupted by Ctrl+C.")
    except Exception as e:
        print(f"An unexpected error occurred during timed capture: {e}")

    if config.ENABLE_GUI_DISPLAY:
        cv2.destroyAllWindows()
    print(f"Timed capture finished. {capture_count} images saved.")

def record_video(picam2, frame_size):
    """Records video until a key is pressed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(config.OUTPUT_DIR, f"video_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"Error: VideoWriter could not be opened for file {output_filename}.")
        return

    print("\n--- Recording Started ---")
    if config.ENABLE_GUI_DISPLAY:
        print("Press the **q** key or the **Enter** key to **STOP** recording.")
    else:
        print("Running headless. Press Ctrl+C to STOP.")
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = frame[:, :, ::-1] 
            
            out.write(frame_bgr)
            
            if config.ENABLE_GUI_DISPLAY:
                cv2.imshow('Recording - Press q or Enter to STOP', frame_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13:
                    print("\nStop key pressed. Stopping recording...")
                    break
            else:
                 time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nRecording interrupted by Ctrl+C.")
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}")

    out.release()
    if config.ENABLE_GUI_DISPLAY:
        cv2.destroyAllWindows()
    
    print(f"Recording finished. Video saved to: {output_filename}")

def select_roi(picam2):
    """
    Opens a live preview and allows the user to select a Region of Interest (ROI).
    """
    global GLOBAL_ROI
    
    if not config.ENABLE_GUI_DISPLAY:
        print("\nERROR: ROI selection requires ENABLE_GUI_DISPLAY=True in config.py.")
        return
        
    print("\n--- ROI Selection Started ---")
    print("Drag a rectangle on the video window and press **ENTER** or **SPACE** to confirm.")
    
    try:
        print("Stabilizing camera feed...")
        for _ in range(5):
            picam2.capture_array() 
            time.sleep(0.1) 
            
        frame_rgb = picam2.capture_array()
        frame_bgr = frame_rgb[:, :, ::-1] 
    except Exception as e:
        print(f"Error capturing stable frame for ROI selection: {e}")
        return

    # cv2.selectROI is a blocking function that opens a GUI window.
    roi = cv2.selectROI("Select ROI", frame_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x, y, w, h = roi
    
    if w > 0 and h > 0:
        GLOBAL_ROI = roi
        save_roi(GLOBAL_ROI)
        print(f"✅ ROI selected: x={x}, y={y}, w={w}, h={h}")
        
        confirm_frame = frame_bgr.copy()
        cv2.rectangle(confirm_frame, (x, y), (x + w, y + h), (255, 255, 0), 3) 
        cv2.putText(confirm_frame, "SELECTED ROI (CONFIRMED)", (x + 5, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('ROI Confirmed', confirm_frame)
        cv2.waitKey(2000) 
        
    else:
        GLOBAL_ROI = None
        save_roi(GLOBAL_ROI)
        print("❌ ROI selection cancelled or invalid. Cleared saved ROI.")

    cv2.destroyAllWindows()
# ----------------------------------------------------------------------


## 🧠 AI Analysis Loop (YOLO) 
# ----------------------------------------------------------------------

def run_ai_analysis(picam2, frame_size, use_roi=False):
    """Runs a continuous loop applying YOLOv8 general object detection."""
    global GLOBAL_ROI
    
    print("\n--- YOLOv8 Analysis Started ---")
    if config.ENABLE_GUI_DISPLAY:
        print("Press the **q** key while the video window is focused to **STOP**.")
    else:
        print("Running headless. Press Ctrl+C to STOP.")
    
    roi_to_use = GLOBAL_ROI if use_roi else None
    detection_classes = config.DETECTION_CLASSES
    
    last_log_time = time.time() - config.LOG_DELAY_SECONDS
    
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = frame_rgb[:, :, ::-1] 
            
            analyzed_frame, detected_data = ai_features.run_yolov8_detection(
                frame_bgr, frame_size, roi=roi_to_use, classes_filter=detection_classes
            )
            
            # --- ACTIONABLE LOGGING & IMAGE CAPTURE (Only in Filtered Mode) ---
            if use_roi and detected_data and (time.time() - last_log_time >= config.LOG_DELAY_SECONDS):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"roi_{timestamp}.jpg" 
                full_image_path = os.path.join(config.ROI_OUTPUT_DIR, image_filename)
                
                cv2.imwrite(full_image_path, analyzed_frame)
                print(f"!!! EVENT !!! Image saved: {full_image_path}")
                
                db_handler.log_detection(
                    detection_data=detected_data, 
                    roi_area=GLOBAL_ROI, 
                    image_filename=image_filename
                )
                
                last_log_time = time.time()
                
            # Display frame
            if config.ENABLE_GUI_DISPLAY:
                cv2.imshow('YOLOv8 Detection - Press q to STOP', analyzed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13:
                    print("\nStop key pressed. Stopping analysis...")
                    break
            else:
                 time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nYOLOv8 analysis interrupted by Ctrl+C.")
    except Exception as e:
        print(f"An unexpected error occurred during YOLOv8 analysis: {e}")

    if config.ENABLE_GUI_DISPLAY:
        cv2.destroyAllWindows()
    print("YOLOv8 Analysis finished.")

# ----------------------------------------------------------------------
## 🚨 Face Recognition Loop (Option 7) using FaceNet
# ----------------------------------------------------------------------

def run_facenet_analysis(picam2, frame_size):
    """
    Runs a continuous loop performing Multi-Frame Face Detection and Recognition.
    """
    print("\n--- FaceNet Recognition Analysis Started (Multi-Frame) ---")
    
    if not face_recognition.initialize_system():
        print("FATAL: Failed to initialize FaceNet system. Check model and image paths.")
        return

    print("Using Multi-Frame Aggregation (History Size: {}) for stability.".format(config.EMBEDDING_HISTORY_SIZE))
    if config.ENABLE_GUI_DISPLAY:
        print("Press the **q** key while the video window is focused to **STOP**.")
    else:
        print("Running headless. Press Ctrl+C to STOP.")
    
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = frame_rgb[:, :, ::-1] 
            
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8)
            
            # 1. Run the combined detection and temporal recognition pipeline
            analyzed_frame, detected_data = face_recognition.run_facenet_recognition(
                frame_bgr, 
                frame_size 
            )
            
            # 2. Check the buffer and log any events whose throttle time has passed
            face_recognition.process_deferred_logs()
            
            # Display frame
            if config.ENABLE_GUI_DISPLAY:
                cv2.imshow('FaceNet Recognition - Press q to STOP', analyzed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13:
                    print("\nStop key pressed. Stopping analysis...")
                    break
            else:
                 time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nFaceNet Analysis interrupted by Ctrl+C.")
    except Exception as e:
        print(f"An unexpected error occurred during FaceNet Analysis: {e}")

    if config.ENABLE_GUI_DISPLAY:
        cv2.destroyAllWindows()
    print("FaceNet Recognition Analysis finished.")

# ----------------------------------------------------------------------
## 💻 Timed Menu Function (Linux/Unix based)
# ----------------------------------------------------------------------

def get_menu_choice_with_timeout(timeout, default_choice):
    """
    Prompts the user for a menu choice, defaulting to a specified value
    after a timeout (non-blocking implementation using select).
    """
    
    print(f"\nWaiting for selection... Auto-selecting Option {default_choice} in {timeout} seconds.")
    
    # Check if there is data waiting on standard input (sys.stdin)
    try:
        # select.select waits until input is available or timeout occurs
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    except select.error as e:
        # Handle cases where select is interrupted or fails (e.g., non-TTY environment)
        print(f"Warning: Select error ({e}). Using default choice.")
        return default_choice
    except Exception:
         # Generic catch-all
         print("Warning: Timed input failed. Using default choice.")
         return default_choice
    
    if rlist:
        # Input is available, read the line and return it
        user_input = sys.stdin.readline().strip()
        print(f"-> Manual choice selected: {user_input}")
        return user_input
    else:
        # Timeout occurred, no input was received
        print(f"\n--- Timeout reached. Auto-selecting Option {default_choice}. ---")
        return default_choice

# ----------------------------------------------------------------------
## 🧠 Main User Interaction Loop 
# ----------------------------------------------------------------------

def main():
    """Presents the menu and executes one action, then exits the script."""
    
    db_handler.initialize_db()
    load_roi()
    
    global GLOBAL_ROI
    
    print("\n--- Camera Action Selection ---")
    print(
        "What would you like to do?\n"
        "1. Capture a **single image**\n"
        "2. Capture an **image every second**\n"
        "3. **Record a video**\n"
        "4. **Set Region of Interest (ROI)**\n"
        "5. **Run Live AI Analysis (Full Frame)**\n"
        "6. **Run Live AI Analysis (Filtered by ROI)**\n"
        "7. **Run Live Face Recognition (FaceNet)**\n"
        "8. **Exit**\n" 
    )
    
    # Display current configuration status
    roi_status = f"Current ROI: {GLOBAL_ROI}" if GLOBAL_ROI else "Current ROI: None (Full frame)"
    display_status = "Enabled (Showing GUI)" if config.ENABLE_GUI_DISPLAY else "Disabled (Headless/No GUI)"
    print(f"\n{roi_status}")
    print(f"GUI Display: {display_status}")
    
    # Print prompt before calling timed function
    prompt = f"Enter your choice ({config.MENU_DEFAULT_CHOICE} is default): "
    sys.stdout.write(prompt)
    sys.stdout.flush() 

    # Get the choice using the timed function
    choice = get_menu_choice_with_timeout(
        timeout=config.MENU_TIMEOUT_SECONDS, 
        default_choice=config.MENU_DEFAULT_CHOICE
    ).strip()

    if choice == '8':
        print("Exiting program.")
        sys.exit(99) 
    
    # Initialize camera for the chosen action
    picam2, frame_size = None, None
    if choice in ('1', '2', '3', '4', '5', '6', '7'):
        # Check if GUI is needed for ROI selection
        if choice == '4' and not config.ENABLE_GUI_DISPLAY:
             print("\nERROR: Cannot set ROI (Option 4) when ENABLE_GUI_DISPLAY is False.")
             sys.exit(1)
             
        picam2, frame_size = initialize_camera()
        
        if picam2 is None: 
            print("Failed to start camera. Exiting.")
            sys.exit(1)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


    # --- Execute Actions ---
    try:
        if choice == '4':
            select_roi(picam2)
        elif choice == '1':
            capture_single_image(picam2, frame_size)
        elif choice == '2':
            capture_timed_images(picam2, frame_size, interval_seconds=1.0)
        elif choice == '3':
            record_video(picam2, frame_size)
        elif choice == '5': 
            run_ai_analysis(picam2, frame_size, use_roi=False)
        elif choice == '6': 
            run_ai_analysis(picam2, frame_size, use_roi=True)
        elif choice == '7':
            run_facenet_analysis(picam2, frame_size)
        else:
            print(f"Option {choice} is not recognized.")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    finally:
        print("Stopping camera...")
        if picam2 is not None:
            picam2.stop()
            try:
                del picam2 
            except Exception:
                 pass 
        
        cv2.destroyAllWindows()
        
    print("\n--- Action Finished ---\n")
    sys.exit(0)


if __name__ == "__main__":
    main()