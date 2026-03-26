import cv2
import time
from datetime import datetime
import os
import sys
import config as app_config
from camera import create_camera

# --- Configuration ---
OUTPUT_DIR = "recorded"
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}")
        sys.exit(1)

# --- Functions ---

def capture_single_image(cam):
    """Captures a single image."""
    print("Capturing a single image...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"image_{timestamp}.jpg")

    frame_bgr = cam.get_frame()
    if frame_bgr is None:
        print("Error: Could not capture frame.")
        return

    success = cv2.imwrite(output_filename, frame_bgr)

    if success:
        print(f"Image saved to: {output_filename}")
    else:
        print(f"Error saving image to: {output_filename}")

    cv2.imshow('Image Captured', frame_bgr)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def capture_timed_images(cam, interval_seconds=1.0):
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
            frame_bgr = cam.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

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


def record_video(cam):
    """Records video until a key is pressed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(output_filename, fourcc, fps, cam.frame_size)

    if not out.isOpened():
        print(f"Error: VideoWriter could not be opened for file {output_filename}.")
        return

    print("\n--- Recording Started ---")
    print(f"Saving video to: {output_filename}")
    print("Press the **q** key or the **Enter** key while the video window is focused to **STOP** recording.")
    print("Or press **Ctrl+C** in the terminal to stop.")

    try:
        while True:
            frame_bgr = cam.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

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

    try:
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()
    except EOFError:
        choice = '4'

    if choice == '4':
        print("Exiting program.")
        sys.exit(1)

    if choice in ('1', '2', '3'):
        cam = create_camera(
            camera_type=app_config.CAMERA_TYPE,
            url=app_config.RTSP_URL,
            transport=app_config.RTSP_TRANSPORT,
            width=app_config.FRAME_WIDTH,
            height=app_config.FRAME_HEIGHT,
        )
        cam.start()

        if not cam.is_running():
            print("Failed to start camera. Exiting to reset hardware.")
            sys.exit(0)

        try:
            if choice == '1':
                capture_single_image(cam)
            elif choice == '2':
                capture_timed_images(cam, interval_seconds=1.0)
            elif choice == '3':
                record_video(cam)
        finally:
            print("Stopping camera...")
            cam.stop()
            cv2.destroyAllWindows()

        print("\n--- Action Finished ---\n")
        sys.exit(0)

    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        sys.exit(0)


if __name__ == "__main__":
    main()
