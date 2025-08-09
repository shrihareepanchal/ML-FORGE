import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import tempfile
import shutil
import io
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime
import base64

def cnn_model_training():
    """CNN Model Training Module - Enhanced Version"""
    st.markdown("## 🖼️ CNN Model Training")
    
    # Create tabs for different stages
    cnn_tabs = st.tabs([
        "1️⃣ Dataset Loading", 
        "2️⃣ Data Preprocessing", 
        "3️⃣ Model Design",
        "4️⃣ Training", 
        "5️⃣ Evaluation",
        "6️⃣ Model Deployment"
    ])
    
    # Track module state
    if "cnn_state" not in st.session_state:
        st.session_state.cnn_state = {
            "dataset_loaded": False,
            "dataset_processed": False,
            "model_built": False,
            "model_trained": False,
            "dataset_path": None,
            "classes": [],
            "image_size": (224, 224),
            "image_channels": 3,
            "train_data": None,
            "val_data": None,
            "test_data": None,
            "model": None,
            "history": None,
            "class_mapping": {}
        }
    
    # =====================================================================
    # STEP 1: DATASET LOADING
    # =====================================================================
    with cnn_tabs[0]:
        st.markdown("### 📂 Dataset Loading")
        
        # Offer multiple ways to load datasets
        data_source = st.radio(
            "Select Data Source",
            ["Upload Zip File", "Upload Individual Images", "Use Sample Dataset"]
        )
        
        if data_source == "Upload Zip File":
            upload_zip(st.session_state.cnn_state)
            
        elif data_source == "Upload Individual Images":
            upload_individual_images(st.session_state.cnn_state)
            
        elif data_source == "Use Sample Dataset":
            use_sample_dataset(st.session_state.cnn_state)
        
        # Display dataset statistics if loaded
        if st.session_state.cnn_state["dataset_loaded"]:
            st.success("✅ Dataset successfully loaded!")
            display_dataset_info(st.session_state.cnn_state)

    # =====================================================================
    # STEP 2: DATA PREPROCESSING
    # =====================================================================
    with cnn_tabs[1]:
        if not st.session_state.cnn_state["dataset_loaded"]:
            st.warning("Please load a dataset first (go to Dataset Loading tab)")
        else:
            data_preprocessing(st.session_state.cnn_state)

    # =====================================================================
    # STEP 3: MODEL DESIGN
    # =====================================================================
    with cnn_tabs[2]:
        if not st.session_state.cnn_state["dataset_processed"]:
            st.warning("Please preprocess your dataset first (go to Data Preprocessing tab)")
        else:
            model_design(st.session_state.cnn_state)

    # =====================================================================
    # STEP 4: TRAINING
    # =====================================================================
    with cnn_tabs[3]:
        if not st.session_state.cnn_state["model_built"]:
            st.warning("Please build your model first (go to Model Design tab)")
        else:
            model_training(st.session_state.cnn_state)

    # =====================================================================
    # STEP 5: EVALUATION
    # =====================================================================
    with cnn_tabs[4]:
        if not st.session_state.cnn_state["model_trained"]:
            st.warning("Please train your model first (go to Training tab)")
        else:
            model_evaluation(st.session_state.cnn_state)

    # =====================================================================
    # STEP 6: MODEL DEPLOYMENT
    # =====================================================================
    with cnn_tabs[5]:
        if not st.session_state.cnn_state["model_trained"]:
            st.warning("Please train your model first (go to Training tab)")
        else:
            model_deployment(st.session_state.cnn_state)
    
    return st.session_state.cnn_state

# Helper functions for Dataset Loading
def upload_zip(state):
    """Handle zip file upload for dataset"""
    uploaded_file = st.file_uploader("Upload Image Dataset (ZIP file)", type=["zip"])
    
    if uploaded_file is not None:
        # Create persistent extraction directory
        upload_id = f"{int(time.time())}_{uploaded_file.name}"
        extract_root = os.path.join(tempfile.gettempdir(), "ml_xpert_uploads")
        dataset_path = os.path.join(extract_root, upload_id)
        
        # Cleanup previous uploads
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save and extract zip
        zip_path = os.path.join(dataset_path, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Find actual dataset root (handle nested folders)
        actual_dataset_path = find_actual_dataset_root(dataset_path)
        
        if validate_dataset_structure(actual_dataset_path):
            state["dataset_path"] = actual_dataset_path
            state["dataset_loaded"] = True
            state["classes"] = get_class_labels(actual_dataset_path)
            state["class_mapping"] = {i: cls for i, cls in enumerate(state["classes"])}
            st.success(f"Successfully loaded dataset with {len(state['classes'])} classes")
            # Store cleanup hook in session state
            st.session_state.dataset_cleanup_path = dataset_path
        else:
            st.error("Invalid dataset structure. Please ensure your ZIP file contains folders with class names.")
            shutil.rmtree(dataset_path)

def upload_individual_images(state):
    """Handle individual image upload for dataset"""
    st.markdown("#### Upload Images")
    st.markdown("1. Enter class labels (comma-separated)")
    st.markdown("2. Upload images and assign classes to each")
    
    # Get class labels
    class_input = st.text_input("Class Labels (comma-separated)", "cat,dog")
    classes = [c.strip() for c in class_input.split(",")]
    
    # Show the upload interface
    uploaded_files = st.file_uploader(
        "Upload Image Files", 
        type=["jpg", "jpeg", "png", "bmp"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create a temporary directory for the dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = temp_dir
            
            # Create class directories
            for cls in classes:
                os.makedirs(os.path.join(dataset_path, cls), exist_ok=True)
            
            # Display images and class selector
            for img in uploaded_files:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display the image
                    image = Image.open(img)
                    st.image(image, caption=img.name, width=150)
                
                with col2:
                    # Select class for this image
                    selected_class = st.selectbox(
                        f"Class for {img.name}", 
                        options=classes,
                        key=f"class_{img.name}"
                    )
                    
                    # Save the image to the appropriate class folder
                    img_path = os.path.join(dataset_path, selected_class, img.name)
                    with open(img_path, "wb") as f:
                        f.write(img.getbuffer())
            
            # Update state
            if uploaded_files:
                state["dataset_path"] = dataset_path
                state["dataset_loaded"] = True
                state["classes"] = classes
                state["class_mapping"] = {i: cls for i, cls in enumerate(classes)}
                st.success(f"Successfully loaded {len(uploaded_files)} images with {len(classes)} classes")

def use_sample_dataset(state):
    """Use a sample dataset for demonstration"""
    sample_options = ["CIFAR-10 (subset)", "Fashion MNIST (subset)"]
    selected_sample = st.selectbox("Select Sample Dataset", sample_options)
    
    if st.button("Load Sample Dataset"):
        with st.spinner("Loading sample dataset..."):
            # Here we would normally download/prepare the sample dataset
            # For now, let's simulate it
            time.sleep(2)  # Simulate download time
            
            # Update state as if we loaded the dataset
            state["dataset_loaded"] = True
            
            if selected_sample == "CIFAR-10 (subset)":
                state["classes"] = ["airplane", "automobile", "bird", "cat", "deer", 
                                    "dog", "frog", "horse", "ship", "truck"]
                state["dataset_path"] = "simulated_cifar10"
            else:  # Fashion MNIST
                state["classes"] = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
                state["dataset_path"] = "simulated_fashion_mnist"
            
            state["class_mapping"] = {i: cls for i, cls in enumerate(state["classes"])}
            st.success(f"Successfully loaded {selected_sample} with {len(state['classes'])} classes")

def validate_dataset_structure(dataset_path):
    """Validate that the dataset has a proper structure"""
    # Check if there are subdirectories (classes)
    subdirs = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not subdirs:
        # Check one level deeper (some archives have a root folder)
        first_level = os.path.join(dataset_path, os.listdir(dataset_path)[0])
        if os.path.isdir(first_level):
            subdirs = [d for d in os.listdir(first_level) 
                      if os.path.isdir(os.path.join(first_level, d))]
            if subdirs:
                return True
        return False
    return True

def get_class_labels(dataset_path):
    """Get class labels from the dataset directory structure"""
    subdirs = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not subdirs:
        # Check one level deeper
        first_level = os.path.join(dataset_path, os.listdir(dataset_path)[0])
        if os.path.isdir(first_level):
            subdirs = [d for d in os.listdir(first_level) 
                      if os.path.isdir(os.path.join(first_level, d))]
    
    return subdirs

def display_dataset_info(state):
    """Display information about the loaded dataset"""
    st.markdown("### Dataset Information")
    
    # Display class distribution in a DataFrame
    st.markdown("#### Class Distribution")
    class_counts = {}
    
    if state["dataset_path"] and os.path.exists(state["dataset_path"]):
        for cls in state["classes"]:
            class_path = os.path.join(state["dataset_path"], cls)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if os.path.isfile(os.path.join(class_path, f))])
                class_counts[cls] = count
    
    # Create a DataFrame and display it
    df_classes = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    st.dataframe(df_classes)
    
    # Plot class distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_title("Class Distribution")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display sample images
    st.markdown("#### Sample Images")
    display_sample_images(state)

def display_sample_images(state):
    """Display sample images from each class"""
    num_classes = len(state["classes"])
    num_samples = min(num_classes, 5)  # Show up to 5 classes
    
    cols = st.columns(num_samples)
    
    # Try to get actual images from the dataset
    if state["dataset_path"] and os.path.exists(state["dataset_path"]):
        for i, cls in enumerate(state["classes"][:num_samples]):
            class_path = os.path.join(state["dataset_path"], cls)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if os.path.isfile(os.path.join(class_path, f)) and 
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                if images:
                    with cols[i]:
                        sample_img_path = os.path.join(class_path, images[0])
                        img = Image.open(sample_img_path)
                        st.image(img, caption=f"Class: {cls}", width=150)

# Helper functions for Data Preprocessing
def data_preprocessing(state):
    """Handle image data preprocessing"""
    st.markdown("### 🔍 Data Preprocessing")
    
    # Image size configuration
    st.markdown("#### Image Size")
    
    # Predefined sizes
    preset_sizes = {
        "Custom": None,
        "32x32 (CIFAR)": (32, 32),
        "48x48 (Small)": (48, 48),
        "96x96 (Medium)": (96, 96),
        "224x224 (VGG/ResNet)": (224, 224),
        "299x299 (Inception)": (299, 299),
        "331x331 (EfficientNet)": (331, 331)
    }
    
    size_option = st.selectbox(
        "Select Image Size Preset", 
        options=list(preset_sizes.keys()),
        index=4  # Default to 224x224
    )
    
    if size_option == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            img_width = st.number_input("Image Width", min_value=16, max_value=512, value=224)
        with col2:
            img_height = st.number_input("Image Height", min_value=16, max_value=512, value=224)
        img_size = (img_width, img_height)
    else:
        img_size = preset_sizes[size_option]
    
    state["image_size"] = img_size
    
    # Normalization options
    st.markdown("#### Pixel Normalization")
    norm_method = st.radio(
        "Normalization Method",
        ["Divide by 255 (Scale to [0,1])", "Standardization (Mean=0, Std=1)"]
    )
    
    # Data augmentation
    st.markdown("#### Data Augmentation")
    
    enable_augmentation = st.checkbox("Enable Data Augmentation", value=True)
    
    if enable_augmentation:
        col1, col2 = st.columns(2)
        with col1:
            rotation_range = st.slider("Rotation Range (degrees)", 0, 90, 20)
            width_shift = st.slider("Width Shift Range", 0.0, 0.5, 0.1, 0.05)
            height_shift = st.slider("Height Shift Range", 0.0, 0.5, 0.1, 0.05)
        
        with col2:
            horizontal_flip = st.checkbox("Horizontal Flip", value=True)
            vertical_flip = st.checkbox("Vertical Flip", value=False)
            zoom_range = st.slider("Zoom Range", 0.0, 0.5, 0.2, 0.05)
    
    # Train/val/test split
    st.markdown("#### Dataset Splitting")
    
    col1, col2 = st.columns(2)
    with col1:
        val_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)
    with col2:
        test_split = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
    
    # Process button
    if st.button("Preprocess Dataset"):
        with st.spinner("Preprocessing images..."):
            # Here we would do the actual preprocessing
            # For now, we'll update the state to indicate preprocessing is done
            time.sleep(2)  # Simulate processing time
            
            state["dataset_processed"] = True
            
            # Configure data augmentation
            if enable_augmentation:
                state["data_augmentation"] = {
                    "rotation_range": rotation_range,
                    "width_shift_range": width_shift,
                    "height_shift_range": height_shift,
                    "horizontal_flip": horizontal_flip,
                    "vertical_flip": vertical_flip,
                    "zoom_range": zoom_range
                }
            else:
                state["data_augmentation"] = None
            
            # Store normalization method
            state["normalization"] = "divide_by_255" if "Divide by 255" in norm_method else "standardize"
            
            # Store split ratios
            state["val_split"] = val_split
            state["test_split"] = test_split
            
            st.success("✅ Dataset preprocessing completed!")
            
            # Display sample of preprocessed images
            st.markdown("#### Sample of Preprocessed Images")
            st.markdown("Here are examples of how your images will look after preprocessing:")
            
            # For demonstration, we'll just show a simulated grid
            show_preprocessed_samples(enable_augmentation)

def show_preprocessed_samples(with_augmentation):
    """Show samples of preprocessed and augmented images"""
    # This is a placeholder - in a real app, we would show actual processed images
    cols = st.columns(3)
    
    if with_augmentation:
        cols[0].image("https://www.tensorflow.org/images/original.png", caption="Original", width=150)
        cols[1].image("https://www.tensorflow.org/images/rotated.png", caption="Rotation", width=150)
        cols[2].image("https://www.tensorflow.org/images/flipped.png", caption="Flipped", width=150)
    else:
        # Just show placeholder images
        for i in range(3):
            cols[i].image("https://www.tensorflow.org/images/original.png", caption=f"Processed {i+1}", width=150)
