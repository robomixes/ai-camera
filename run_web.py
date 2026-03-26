"""Standalone entry point for the AI Camera web dashboard."""
import sys
import uvicorn
from web.server import create_app
import config

app = create_app()

if __name__ == "__main__":
    host = getattr(config, "WEB_HOST", "0.0.0.0")
    port = getattr(config, "WEB_PORT", 8080)
    print(f"\nStarting AI Camera Dashboard at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
