"""
ML-FORGE Runner Script

This script ensures the proper environment setup before running the ML-FORGE application.
It adds necessary paths to sys.path to make imports work and then launches the Streamlit app.
"""

import os
import sys
import subprocess
import platform

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Ensure the src directory is in sys.path
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Launch the Streamlit app
if __name__ == "__main__":
    print("Starting ML-FORGE application...")
    print(f"Python path: {sys.path}")
    
    # Use the appropriate command based on the platform
    if platform.system() == "Windows":
        subprocess.run(["streamlit", "run", os.path.join(root_dir, "app.py")])
    else:
        subprocess.run(["streamlit", "run", os.path.join(root_dir, "app.py")], shell=True)
