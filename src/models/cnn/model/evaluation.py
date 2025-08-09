import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CNNEvaluator:
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the CNN evaluator.
        
        Args:
            model: Trained Keras model to evaluate
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        
    def evaluate_model(
        self,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset: Test dataset to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Evaluate the model
            results = self.model.evaluate(test_dataset, verbose=0)
            
            # Create metrics dictionary
            metrics = {
                'loss': results[0],
                'accuracy': results[1]
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def get_predictions(
        self,
        dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and true labels from a dataset.
        
        Args:
            dataset: Dataset to get predictions for
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        try:
            predictions = []
            true_labels = []
            
            for images, labels in dataset:
                # Get model predictions
                pred = self.model.predict(images)
                predictions.extend(np.argmax(pred, axis=1))
                true_labels.extend(labels.numpy())
                
            return np.array(predictions), np.array(true_labels)
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {str(e)}")
            raise
            
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate various classification metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, average='weighted'),
                'recall': recall_score(true_labels, predictions, average='weighted'),
                'f1': f1_score(true_labels, predictions, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: str = None
    ) -> None:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            save_path: Optional path to save the plot
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=True,
                yticklabels=True
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def generate_classification_report(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        target_names: List[str] = None
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            target_names: Optional list of class names
            
        Returns:
            Classification report as string
        """
        try:
            report = classification_report(
                true_labels,
                predictions,
                target_names=target_names,
                digits=4
            )
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            raise
            
    def plot_training_history(
        self,
        history: tf.keras.callbacks.History,
        save_path: str = None
    ) -> None:
        """
        Plot training history metrics.
        
        Args:
            history: Training history object
            save_path: Optional path to save the plot
        """
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot loss
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
            raise 