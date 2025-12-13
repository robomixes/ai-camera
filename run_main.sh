#!/bin/bash

# --- Shell Script to run the Python camera app in a loop (Direct Execution) ---

# Define the absolute path to the Python interpreter that contains tflite-runtime.
# This bypasses the unreliable 'source activate' command.
PYTHON_EXECUTABLE="/home/admin/env/bin/python"
PYTHON_SCRIPT="ai-general.py"

# Check if the Python executable exists before running
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "FATAL ERROR: Python executable not found at $PYTHON_EXECUTABLE"
    echo "Please verify your virtual environment path."
    exit 1
fi

# Define the full command
PYTHON_COMMAND="$PYTHON_EXECUTABLE $PYTHON_SCRIPT"

# Loop continues as long as the python script exits with status 0
while $PYTHON_COMMAND
do
    echo "--- Rerunning application to ensure clean camera state ---"
    # Wait a moment before relaunching, just in case
    sleep 0.5 
done

# The 'deactivate' command is not needed here as we never activated the environment.
echo "Application loop terminated."