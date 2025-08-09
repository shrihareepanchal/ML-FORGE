import streamlit as st
import os
import sys
import importlib

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Ensure the src directory is in sys.path
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Define necessary packages to check before importing the main app
required_packages = [
    'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
    'tensorflow', 'sklearn', 'xgboost', 'lightgbm'
]

# Verify that all required packages are available
missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)

# If there are missing packages, display an error and exit
if missing_packages:
    st.error(f"Unable to start the application due to missing packages: {', '.join(missing_packages)}")
    st.info("Please make sure all dependencies are properly installed.")
    st.code("pip install -r requirements.txt")
    st.stop()

# Only import the main app if all dependencies are available
try:
    from app import main
    # Run the main application
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"Error loading the application: {str(e)}")
    st.info("Please check the logs for more details.")
    st.stop()
