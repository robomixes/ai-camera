import cv2
from picamera2 import Picamera2
import time
from datetime import datetime
import os
import sys

# --- Configuration ---
OUTPUT_DIR = "recorded"
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}")
        # Exit if output directory cannot be created
        sys.exit(1) 

# --- Functions (Initialize, Capture, Record) remain the same ---

def initialize_camera():
    """Initializes and returns the Picamera2 object with a video configuration."""
    try:
        picam2 = Picamera2()
        frame_size = (1280, 720)
        config = picam2.create_video_configuration(main={"size": frame_size, "format": "RGB888"})
        picam2.configure(config)
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
    output_filename = os.path.join(OUTPUT_DIR, f"image_{timestamp}.jpg")
    
    frame = picam2.capture_array()
    frame_bgr = frame[:, :, ::-1] # RGB to BGR
    
    success = cv2.imwrite(output_filename, frame_bgr)
    
    if success:
        print(f"✅ Image saved to: {output_filename}")
    else:
        print(f"❌ Error saving image to: {output_filename}")
        
    cv2.imshow('Image Captured', frame_bgr)
    cv2.waitKey(2000) 
    cv2.destroyAllWindows()


def capture_timed_images(picam2, frame_size, interval_seconds=1.0):
    """Captures an image every 'interval_seconds'."""
    print(f"\n--- Timed Capture Started ---")
    print(f"Capturing an image every **{interval_seconds}** second(s).")
    print("Press the **q** key while the preview window is focused to **STOP**.")
    print("Or press **Ctrl+C** in the terminal to stop.")
    
    last_capture_time = time.time() - interval_seconds
    capture_count = 0

    try:
        while True:
            current_time = time.time()
            frame = picam2.capture_array()
            frame_bgr = frame[:, :, ::-1]
            
            cv2.imshow('Timed Capture - Press q to STOP', frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 13: 
                print("\nStop key pressed. Stopping timed capture...")
                break
                
            if current_time - last_capture_time >= interval_seconds:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = os.path.join(OUTPUT_DIR, f"timed_img_{timestamp}_{capture_count:04d}.jpg")
                
                success = cv2.imwrite(output_filename, frame_bgr)
                if success:
                    print(f"Captured: {output_filename}")
                    capture_count += 1
                else:
                    print(f"Error saving timed image: {output_filename}")
                    
                last_capture_time = current_time
                
    except KeyboardInterrupt:
        print("\n**Ctrl+C** detected. Stopping timed capture...")
    except Exception as e:
        print(f"An unexpected error occurred during timed capture: {e}")

    cv2.destroyAllWindows()
    print(f"Timed capture finished. {capture_count} images saved.")


def record_video(picam2, frame_size):
    """Records video until a key is pressed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"Error: VideoWriter could not be opened for file {output_filename}.")
        return

    print("\n--- Recording Started ---")
    print(f"Saving video to: {output_filename}")
    print("Press the **q** key or the **Enter** key while the video window is focused to **STOP** recording.")
    print("Or press **Ctrl+C** in the terminal to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = frame[:, :, ::-1] 
            
            out.write(frame_bgr)
            cv2.imshow('Recording - Press q or Enter to STOP', frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 13:
                print("\nStop key pressed. Stopping recording...")
                break
            
    except KeyboardInterrupt:
        print("\n**Ctrl+C** detected. Stopping recording...")
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}")

    out.release()
    cv2.destroyAllWindows()
    
    print(f"Recording finished. Video saved to: {output_filename}")


def main():
    """Presents the menu and executes one action, then exits the script."""
    print("\n--- Camera Action Selection ---")
    print(
        "What would you like to do?\n"
        "1. Capture a **single image** (display)\n"
        "2. Capture an **image every second** (image every second example image 10)\n"
        "3. **Record a video** (video)\n"
        "4. **Exit**\n"
    )
    
    # Read raw user input
    try:
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()
    except EOFError:
        choice = '4' # Handle if input is closed unexpectedly

    if choice == '4':
        # Exit with status 1 to tell the shell script to stop looping
        print("Exiting program.")
        sys.exit(1)
    
    if choice in ('1', '2', '3'):
        picam2, frame_size = initialize_camera()
        if picam2 is None:
            print("Failed to start camera. Exiting to reset hardware.")
            # Exit with status 0 to tell the shell script to try again
            sys.exit(0)

        try:
            if choice == '1':
                capture_single_image(picam2, frame_size)
            elif choice == '2':
                capture_timed_images(picam2, frame_size, interval_seconds=1.0)
            elif choice == '3':
                record_video(picam2, frame_size)
        finally:
            print("Stopping camera...")
            picam2.stop()
            # Explicitly delete and close resources
            del picam2 
            cv2.destroyAllWindows()
            
        print("\n--- Action Finished ---\n")
        # Exit with status 0 to tell the shell script to continue the loop
        sys.exit(0)
        
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        # Exit with status 0 to show the menu again
        sys.exit(0)


if __name__ == "__main__":
    main()