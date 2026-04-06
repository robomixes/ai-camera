"""Standalone entry point for the AI Camera web dashboard."""
import logging
import sys
import uvicorn
from app_logging import setup_logging
from web.server import create_app
import config

# Initialize logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

# Suppress noisy uvicorn shutdown errors
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

app = create_app()

if __name__ == "__main__":
    host = getattr(config, "WEB_HOST", "0.0.0.0")
    port = getattr(config, "WEB_PORT", 8080)
    logger.info(f"Starting Privora AI Camera at http://localhost:{port}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="error")
    except (KeyboardInterrupt, SystemExit):
        pass
    logger.info("Shutdown complete.")
