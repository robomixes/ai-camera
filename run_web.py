"""Standalone entry point for the AI Camera web dashboard."""
import logging
import signal
import sys
import uvicorn
from app_logging import setup_logging
from web.server import create_app
import config

# Initialize logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

app = create_app()


def _shutdown_handler(signum, frame):
    """Handle SIGTERM/SIGINT for clean shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, shutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    host = getattr(config, "WEB_HOST", "0.0.0.0")
    port = getattr(config, "WEB_PORT", 8080)
    logger.info(f"Starting AI Camera Dashboard at http://localhost:{port}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown complete.")
