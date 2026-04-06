#!/bin/bash
# Privora AI Camera - Setup Script (Linux/Raspberry Pi)

set -e

echo "=== Privora AI Camera Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install base dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "Raspberry Pi detected. Installing Pi-specific dependencies..."
    pip install -r requirements-pi.txt
fi

# Create runtime directories
echo "Creating directories..."
mkdir -p logs event_images roi_events plate_images output_images people_search_queue/ready

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the dashboard:"
echo "  source venv/bin/activate"
echo "  python run_web.py"
echo ""
echo "Then open http://localhost:8080"
echo "Default login: admin / admin"
echo ""
