"""Manages multiple cameras with their AI runners."""
import asyncio
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
        if self.cam:
            self.cam.stop()
            self.cam = None


class CameraManager:
    """Manages multiple camera instances."""

    def __init__(self):
        self._instances: dict[str, CameraInstance] = {}

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
            self._instances[cam_id] = instance
            logger.info(f"Camera '{cam_id}' started.")
        except Exception as e:
            logger.error(f"Failed to start camera '{cam_id}': {e}")

    def stop_all(self):
        for cam_id, instance in self._instances.items():
            try:
                instance.stop()
                logger.info(f"Camera '{cam_id}' stopped.")
            except Exception as e:
                logger.error(f"Error stopping camera '{cam_id}': {e}")
        self._instances.clear()

    def get_instance(self, camera_id: str = "") -> CameraInstance | None:
        """Get a camera instance by ID, or the first one if ID is empty."""
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
        for cam_id, inst in self._instances.items():
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

    @property
    def camera_ids(self) -> list[str]:
        return list(self._instances.keys())

    @property
    def is_multi(self) -> bool:
        return len(self._instances) > 1
