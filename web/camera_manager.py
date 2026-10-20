"""Manages multiple cameras with their AI runners."""
import asyncio
import threading
import time
from camera import create_camera
from camera.base import CameraBase
from web.ai_runner import AIRunner
import logging
import config

logger = logging.getLogger(__name__)


class CameraInstance:
    """Holds a camera + its AI runner + config."""

    def __init__(self, cam_config: dict):
        self.config = cam_config
        self.id = cam_config["id"]
        self.description = cam_config.get("description", self.id)
        self.cam: CameraBase | None = None
        self.runner: AIRunner | None = None

    def start(self) -> bool:
        self.cam = create_camera(
            camera_type=self.config.get("type", "rtsp"),
            url=self.config.get("url", ""),
            transport=self.config.get("transport", "tcp"),
            width=self.config.get("width", 1280),
            height=self.config.get("height", 720),
        )
        self.cam.start()

        # Wait for first frame
        logger.info(f"[{self.id}] Waiting for camera stream...")
        for _ in range(150):
            if self.cam.get_frame() is not None:
                logger.info(f"[{self.id}] Camera stream ready.")
                break
            time.sleep(0.1)
        else:
            logger.warning(f"[{self.id}] WARNING - No frames received after 15s.")

        ai_mode = self.config.get("ai_mode", getattr(config, "DEFAULT_AI_MODE", "yolo"))
        self.runner = AIRunner(self.cam, mode=ai_mode, camera_id=self.id)
        self.runner.start()
        return True

    def stop(self):
        if self.runner:
            self.runner.stop()
            self.runner = None
        import time
        time.sleep(0.5)  # let AI thread fully exit before releasing camera
        if self.cam:
            self.cam.stop()
            self.cam = None


class CameraManager:
    """Manages multiple camera instances."""

    def __init__(self):
        self._instances: dict[str, CameraInstance] = {}
        # Guards every mutation of _instances. Reentrant because restart_camera
        # calls remove_camera + add_camera, and reload_cameras calls
        # remove_camera. Without it a reload racing a config save could leave a
        # CameraInstance overwritten but never stopped, orphaning a capture
        # thread and an AIRunner that keep writing detections forever.
        self._lock = threading.RLock()
        # Held for the whole of a reconfigure. Background rediscovery acquires
        # this non-blocking and skips its turn if a user save holds it, so an
        # automatic action never competes with a deliberate one.
        self._reconfigure_lock = threading.RLock()

    @property
    def reconfigure_lock(self) -> threading.RLock:
        return self._reconfigure_lock

    def _get_camera_configs(self) -> list[dict]:
        """Get camera configs — from CAMERAS list or fallback to single-camera fields."""
        if config.CAMERAS:
            return config.CAMERAS

        # Fallback: build single-camera config from legacy fields
        return [{
            "id": config.CAMERA_ID,
            "description": getattr(config, "CAMERA_DESCRIPTION", config.CAMERA_ID),
            "type": config.CAMERA_TYPE,
            "url": config.RTSP_URL,
            "transport": config.RTSP_TRANSPORT,
            "width": config.FRAME_WIDTH,
            "height": config.FRAME_HEIGHT,
            "latitude": getattr(config, "GPS_LATITUDE", 0),
            "longitude": getattr(config, "GPS_LONGITUDE", 0),
            "ai_mode": getattr(config, "DEFAULT_AI_MODE", "yolo"),
        }]

    def start_all(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        configs = self._get_camera_configs()

        if len(configs) <= 1:
            # Single camera — start directly
            for cam_cfg in configs:
                self._start_one(cam_cfg)
            return

        # Multiple cameras — start in parallel
        logger.info(f"Starting {len(configs)} cameras in parallel...")
        with ThreadPoolExecutor(max_workers=len(configs)) as executor:
            futures = {
                executor.submit(self._start_one, cfg): cfg["id"]
                for cfg in configs
            }
            for future in as_completed(futures):
                cam_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to start camera '{cam_id}': {e}")

    def _start_one(self, cam_cfg: dict):
        cam_id = cam_cfg["id"]
        try:
            instance = CameraInstance(cam_cfg)
            instance.start()
            with self._lock:
                # Replacing an entry without stopping it would strand its
                # capture and AI threads with no owner.
                displaced = self._instances.get(cam_id)
                self._instances[cam_id] = instance
            if displaced is not None:
                logger.warning(f"Camera '{cam_id}' replaced a running instance — stopping the old one")
                try:
                    displaced.stop()
                except Exception as e:
                    logger.error(f"Error stopping displaced camera '{cam_id}': {e}")
            logger.info(f"Camera '{cam_id}' started.")
        except Exception as e:
            logger.error(f"Failed to start camera '{cam_id}': {e}")

    def stop_all(self):
        with self._lock:
            snapshot = list(self._instances.items())
            self._instances.clear()
        for cam_id, instance in snapshot:
            try:
                instance.stop()
                logger.info(f"Camera '{cam_id}' stopped.")
            except Exception as e:
                logger.error(f"Error stopping camera '{cam_id}': {e}")

    def get_instance(self, camera_id: str = "") -> CameraInstance | None:
        """Get a camera instance by ID, or the first one if ID is empty."""
        with self._lock:
            if not camera_id and self._instances:
                return next(iter(self._instances.values()))
            return self._instances.get(camera_id)

    def get_camera(self, camera_id: str = "") -> CameraBase | None:
        inst = self.get_instance(camera_id)
        return inst.cam if inst else None

    def get_runner(self, camera_id: str = "") -> AIRunner | None:
        inst = self.get_instance(camera_id)
        return inst.runner if inst else None

    def list_cameras(self) -> list[dict]:
        """Return status info for all cameras."""
        result = []
        with self._lock:
            snapshot = list(self._instances.items())
        for cam_id, inst in snapshot:
            runner = inst.runner
            cam = inst.cam
            has_signal = cam is not None and (
                hasattr(cam, 'has_signal') and cam.has_signal() or
                (not hasattr(cam, 'has_signal') and cam.is_running() and cam.get_frame() is not None)
            )
            result.append({
                "id": cam_id,
                "description": inst.description,
                "connected": has_signal,
                "frame_size": list(cam.frame_size) if cam else None,
                "ai_mode": runner.mode if runner else None,
                "ai_fps": round(runner.fps, 1) if runner else 0,
                "type": inst.config.get("type", ""),
            })
        return result

    def add_camera(self, cam_config: dict) -> bool:
        """Start a new camera at runtime."""
        cam_id = cam_config.get("id")
        if not cam_id:
            return False
        with self._lock:
            if cam_id in self._instances:
                logger.warning(f"Camera '{cam_id}' already exists. Remove first.")
                return False
        try:
            self._start_one(cam_config)
            return True
        except Exception as e:
            logger.error(f"Failed to add camera '{cam_id}': {e}")
            return False

    def remove_camera(self, cam_id: str) -> bool:
        """Stop and remove a camera at runtime."""
        with self._lock:
            inst = self._instances.pop(cam_id, None)
        if not inst:
            return False
        try:
            inst.stop()
            logger.info(f"Camera '{cam_id}' removed.")
            return True
        except Exception as e:
            logger.error(f"Error removing camera '{cam_id}': {e}")
            return False

    def restart_camera(self, cam_id: str) -> bool:
        """Restart a single camera."""
        with self._lock:
            inst = self._instances.get(cam_id)
        if not inst:
            return False
        cam_config = inst.config
        self.remove_camera(cam_id)
        return self.add_camera(cam_config)

    def reload_cameras(self):
        """Reload cameras from config — stop removed, start new, restart changed.

        Blocking: starting a camera waits up to 15s for its first frame. Call
        it off the event loop.
        """
        with self._reconfigure_lock:
            self._reload_cameras_locked()

    def _reload_cameras_locked(self):
        new_configs = self._get_camera_configs()
        new_ids = {c["id"] for c in new_configs}
        with self._lock:
            current_ids = set(self._instances.keys())

        # Stop cameras that were removed
        for cam_id in current_ids - new_ids:
            self.remove_camera(cam_id)
            logger.info(f"Camera '{cam_id}' removed (no longer in config)")

        # Start new cameras or restart changed ones
        from concurrent.futures import ThreadPoolExecutor, as_completed
        to_start = []
        for cam_cfg in new_configs:
            cam_id = cam_cfg["id"]
            with self._lock:
                existing = self._instances.get(cam_id)
            if existing is None:
                # New camera
                to_start.append(cam_cfg)
            else:
                # Check if config changed (URL, type, etc.)
                old_cfg = existing.config
                if cam_cfg.get("url") != old_cfg.get("url") or cam_cfg.get("type") != old_cfg.get("type"):
                    self.remove_camera(cam_id)
                    to_start.append(cam_cfg)
                    logger.info(f"Camera '{cam_id}' config changed — restarting")

        if to_start:
            if len(to_start) == 1:
                self._start_one(to_start[0])
            else:
                with ThreadPoolExecutor(max_workers=len(to_start)) as executor:
                    futures = {executor.submit(self._start_one, cfg): cfg["id"] for cfg in to_start}
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Failed to start camera: {e}")

        with self._lock:
            active = len(self._instances)
        logger.info(f"Camera reload complete. {active} camera(s) active.")

    @property
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._instances.keys())

    @property
    def is_multi(self) -> bool:
        with self._lock:
            return len(self._instances) > 1
