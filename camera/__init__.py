import platform
from camera.base import CameraBase


def create_camera(camera_type: str = "auto", **kwargs) -> CameraBase:
    """
    Factory function to create the appropriate camera backend.

    Args:
        camera_type: "auto", "picamera", or "rtsp"
        **kwargs: Passed to the backend constructor.
            For picamera: width, height
            For rtsp: url, transport, width, height
    """
    if camera_type == "rtsp":
        from camera.rtsp_backend import RTSPBackend
        url = kwargs.get("url", "")
        if not url:
            raise ValueError("RTSP_URL is required when CAMERA_TYPE is 'rtsp'")
        return RTSPBackend(
            url=url,
            transport=kwargs.get("transport", "tcp"),
            width=kwargs.get("width", 1280),
            height=kwargs.get("height", 720),
        )

    if camera_type == "picamera":
        from camera.picamera_backend import PicameraBackend
        return PicameraBackend(
            width=kwargs.get("width", 1280),
            height=kwargs.get("height", 720),
        )

    # auto-detect
    if platform.machine() in ("aarch64", "armv7l"):
        try:
            from camera.picamera_backend import PicameraBackend
            print("Auto-detected ARM platform, using Picamera2 backend.")
            return PicameraBackend(
                width=kwargs.get("width", 1280),
                height=kwargs.get("height", 720),
            )
        except ImportError:
            print("Picamera2 not available, falling back to RTSP backend.")

    # fallback to RTSP on non-ARM or if Picamera2 unavailable
    url = kwargs.get("url", "")
    if not url:
        raise ValueError(
            "Could not auto-detect camera. Set CAMERA_TYPE and RTSP_URL in config.py"
        )
    print("Using RTSP backend.")
    from camera.rtsp_backend import RTSPBackend
    return RTSPBackend(
        url=url,
        transport=kwargs.get("transport", "tcp"),
        width=kwargs.get("width", 1280),
        height=kwargs.get("height", 720),
    )
