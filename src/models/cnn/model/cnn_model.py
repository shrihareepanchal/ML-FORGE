import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging
import os

class CNNModel:
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        model_name: str = "cnn_model"
    ):
        """
        Initialize the CNN model with specified parameters.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            model_name: Name for the model
        """
        self.logger = logging.getLogger(__name__)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        try:
            # Input layer
            inputs = layers.Input(shape=self.input_shape)
            
            # First convolutional block
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Second convolutional block
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Third convolutional block
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Flatten and dense layers
            x = layers.Flatten()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
            
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10,
        callbacks: Optional[list] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the CNN model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        try:
            # Default callbacks if none provided
            if callbacks is None:
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f"{self.model_name}_best.h5",
                        monitor='val_accuracy',
                        save_best_only=True
                    )
                ]
            
            # Train the model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path where the model should be saved
        """
        try:
            self.model.save(save_path)
            self.logger.info(f"Model saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    @classmethod
    def load_model(cls, model_path: str) -> 'CNNModel':
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded CNNModel instance
        """
        try:
            model = tf.keras.models.load_model(model_path)
            cnn_model = cls(
                input_shape=model.input_shape[1:],
                num_classes=model.output_shape[-1],
                model_name=model.name
            )
            cnn_model.model = model
            return cnn_model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, image: tf.Tensor) -> np.ndarray:
        """
        Make predictions on a single image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Predicted class probabilities
        """
        try:
            # Ensure image has correct shape
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
                
            # Make prediction
            predictions = self.model.predict(image)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise 