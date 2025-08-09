"""
ML-FORGE: Machine Learning Expert System
Main application entry point

This file serves as the central entry point for the ML-FORGE application.
It imports all necessary modules from the new project structure.
"""

import os
import sys

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app module
import streamlit as st

# Run the app directly
if __name__ == "__main__":
    import subprocess
    import platform
    
    # Use the appropriate command based on the platform
    if platform.system() == "Windows":
        subprocess.run(["streamlit", "run", "app.py"])
    else:
        subprocess.run(["streamlit", "run", "app.py"], shell=True)
