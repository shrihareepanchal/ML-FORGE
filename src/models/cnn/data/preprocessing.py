import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
import logging
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split

class ImagePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Initialize the image preprocessor with configuration parameters.
        
        Args:
            target_size: Target size for resizing images (height, width)
            batch_size: Batch size for data loading
            validation_split: Fraction of data to use for validation
        """
        self.logger = logging.getLogger(__name__)
        self.target_size = target_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.target_size)
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
            
    def create_data_generator(
        self,
        image_paths: List[str],
        labels: List[int],
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow data generator for the images.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            is_training: Whether this is for training or validation
            
        Returns:
            TensorFlow Dataset object
        """
        try:
            # Create dataset from image paths and labels
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            
            # Map preprocessing function
            dataset = dataset.map(
                lambda x, y: (self.load_and_preprocess_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Shuffle and batch
            if is_training:
                dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(self.batch_size)
            
            # Prefetch for better performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error creating data generator: {str(e)}")
            raise
            
    def prepare_dataset(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            
        Returns:
            Tuple of (training_dataset, validation_dataset)
        """
        try:
            # Split data into training and validation sets
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels,
                test_size=self.validation_split,
                random_state=42
            )
            
            # Create datasets
            train_dataset = self.create_data_generator(
                train_paths, train_labels, is_training=True
            )
            val_dataset = self.create_data_generator(
                val_paths, val_labels, is_training=False
            )
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise
            
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image as numpy array
        """
        try:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                
            # Random rotation
            angle = np.random.uniform(-15, 15)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # Random brightness adjustment
            image = tf.image.random_brightness(image, 0.2)
            
            return image.numpy()
            
        except Exception as e:
            self.logger.error(f"Error augmenting image: {str(e)}")
            raise 