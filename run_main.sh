#!/bin/bash

# --- Shell Script to run the Python camera app in a loop ---

# Make sure this command points to the correct file name!
PYTHON_COMMAND="python3 ai-general.py"

# Loop continues as long as the python script exits with status 0
while $PYTHON_COMMAND
do
    echo "--- Rerunning application to ensure clean camera state ---"
    # Wait a moment before relaunching, just in case
    sleep 0.5 
done

echo "Application loop terminated."