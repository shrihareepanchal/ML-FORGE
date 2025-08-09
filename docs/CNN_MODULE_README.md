# ML-FORGE CNN Module Documentation

## Overview

The ML-FORGE CNN Module is a comprehensive deep learning solution for image classification tasks. The module provides a streamlined workflow for dataset loading, preprocessing, model design, training, evaluation, and deployment of Convolutional Neural Networks (CNNs).

## Features

### 1. Dataset Loading
- Upload image datasets via zip files containing folders for each class
- Upload individual images and assign classes manually
- Use sample datasets for demonstration purposes
- Support for common image formats (JPG, PNG, BMP)

### 2. Data Preprocessing
- Image resizing with common presets (32x32, 48x48, 96x96, 224x224, 299x299, 331x331)
- Multiple normalization options (divide by 255, standardization)
- Data augmentation with adjustable parameters:
  - Rotation range
  - Width and height shifts
  - Horizontal and vertical flips
  - Zoom range
- Custom train/validation/test split ratios

### 3. Model Design and Customization
- Pre-built architectures:
  - LeNet-5
  - AlexNet
  - VGG-like
  - Transfer learning options (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
- Custom CNN architecture:
  - Configurable number of convolutional blocks
  - Adjustable filters, kernel sizes, and pooling types
  - Batch normalization options
  - Customizable fully connected layers

### 4. Training Configuration
- Batch size selection
- Learning rate adjustment
- Optimizer options (Adam, SGD, RMSprop)
- Early stopping with adjustable patience
- Best model checkpointing
- Learning rate schedulers

### 5. Training Visualization
- Real-time loss and accuracy plots
- Training and validation metrics tracking

### 6. Evaluation and Results
- Comprehensive metrics:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix with class labels
  - Class-wise performance analytics
- Sample predictions visualization
- Confidence score displays

### 7. Model Deployment
- Model saving in multiple formats:
  - Keras H5 (.h5)
  - TensorFlow SavedModel (.pb)
  - ONNX Format (.onnx)
- Architecture summary export
- Training configuration export

### 8. Inference
- Upload new images for prediction
- View multiple class predictions with confidence scores
- Visual representation of prediction results

## Getting Started

1. Navigate to the Deep Learning Model Training section
2. Select "Convolutional Neural Network (CNN)"
3. Follow the step-by-step tabs to complete your CNN workflow

## Technical Details

The CNN module is built on TensorFlow/Keras and integrates seamlessly with the ML-FORGE application. The module supports both custom CNN architectures and transfer learning with pre-trained models.

### Dependencies
- TensorFlow ≥ 2.12.0
- Streamlit ≥ 1.22.0
- Numpy ≥ 1.24.3
- Pillow ≥ 9.5.0
- Matplotlib ≥ 3.7.1
- Pandas ≥ 1.5.3

## Usage Examples

### Image Classification Workflow

1. **Load a dataset**:
   - Upload a zip file containing folders with class names
   - Or upload individual images and assign classes manually

2. **Preprocess the data**:
   - Select image size (e.g., 224x224 for VGG16)
   - Choose normalization method
   - Enable augmentation for small datasets

3. **Design the model**:
   - Select a pre-built architecture or create a custom CNN
   - Configure the layers, activation functions, and regularization

4. **Train the model**:
   - Set training parameters like batch size and learning rate
   - Enable callbacks like early stopping and model checkpointing
   - Monitor training progress with real-time metrics

5. **Evaluate performance**:
   - View comprehensive metrics and visualizations
   - Analyze the confusion matrix and class-wise performance

6. **Deploy and use the model**:
   - Save the model for future use
   - Run inference on new images

## Best Practices

- Use data augmentation for small datasets
- Start with pre-built architectures before customizing
- Monitor validation metrics to prevent overfitting
- Use early stopping to optimize training time
- For transfer learning, try freezing base model layers first
