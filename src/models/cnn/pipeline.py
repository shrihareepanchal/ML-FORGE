import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .data.secure_upload import SecureFileHandler
from .data.preprocessing import ImagePreprocessor
from .model.cnn_model import CNNModel
from .model.evaluation import CNNEvaluator
from .utils.reporting import ReportGenerator

class CNNPipeline:
    def __init__(
        self,
        input_shape: tuple = (224, 224, 3),
        num_classes: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        report_dir: str = "reports",
        email_config: Optional[Dict] = None
    ):
        """
        Initialize the CNN pipeline.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            report_dir: Directory to save reports
            email_config: Optional email configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Initialize components
        self.secure_handler = SecureFileHandler()
        self.preprocessor = ImagePreprocessor(
            target_size=input_shape[:2],
            batch_size=batch_size,
            validation_split=validation_split
        )
        self.model = CNNModel(
            input_shape=input_shape,
            num_classes=num_classes
        )
        self.evaluator = CNNEvaluator(self.model.model)
        self.report_generator = ReportGenerator(
            report_dir=report_dir,
            email_config=email_config
        )
        
    def process_encrypted_data(
        self,
        encrypted_data: bytes,
        output_path: str
    ) -> str:
        """
        Process encrypted data and return the decrypted file path.
        
        Args:
            encrypted_data: Encrypted data to process
            output_path: Path where decrypted data should be saved
            
        Returns:
            Path to the decrypted file
        """
        try:
            return self.secure_handler.decrypt_file(encrypted_data, output_path)
        except Exception as e:
            self.logger.error(f"Error processing encrypted data: {str(e)}")
            raise
            
    def prepare_dataset(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> tuple:
        """
        Prepare training and validation datasets.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        try:
            return self.preprocessor.prepare_dataset(image_paths, labels)
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise
            
    def train_model(
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
            return self.model.train(
                train_dataset,
                val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate_model(
        self,
        test_dataset: tf.data.Dataset,
        save_plots: bool = True
    ) -> Dict:
        """
        Evaluate the model and generate visualizations.
        
        Args:
            test_dataset: Test dataset to evaluate on
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Get predictions and true labels
            predictions, true_labels = self.evaluator.get_predictions(test_dataset)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(predictions, true_labels)
            
            # Generate classification report
            classification_report = self.evaluator.generate_classification_report(
                predictions,
                true_labels
            )
            
            # Generate and save plots if requested
            plots = []
            if save_plots:
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                
                # Save confusion matrix
                cm_path = plots_dir / "confusion_matrix.png"
                self.evaluator.plot_confusion_matrix(
                    predictions,
                    true_labels,
                    save_path=str(cm_path)
                )
                plots.append(str(cm_path))
                
            return {
                'metrics': metrics,
                'classification_report': classification_report,
                'plots': plots
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def generate_and_send_report(
        self,
        evaluation_results: Dict,
        training_history: Dict,
        model_info: Dict,
        recipient_email: Optional[str] = None
    ) -> str:
        """
        Generate and optionally send an evaluation report.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            training_history: Dictionary containing training history
            model_info: Dictionary containing model information
            recipient_email: Optional email address to send report to
            
        Returns:
            Path to the generated report file
        """
        try:
            # Generate report
            report_path = self.report_generator.generate_report(
                metrics=evaluation_results['metrics'],
                classification_report=evaluation_results['classification_report'],
                model_info=model_info,
                training_history=training_history,
                plots=evaluation_results['plots']
            )
            
            # Send report if email provided
            if recipient_email:
                self.report_generator.send_report(
                    report_path,
                    recipient_email
                )
                
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating and sending report: {str(e)}")
            raise
            
    def run_pipeline(
        self,
        encrypted_data: bytes,
        image_paths: List[str],
        labels: List[int],
        test_dataset: tf.data.Dataset,
        epochs: int = 10,
        save_plots: bool = True,
        recipient_email: Optional[str] = None
    ) -> Dict:
        """
        Run the complete CNN pipeline.
        
        Args:
            encrypted_data: Encrypted data to process
            image_paths: List of paths to image files
            labels: List of corresponding labels
            test_dataset: Test dataset for evaluation
            epochs: Number of training epochs
            save_plots: Whether to save evaluation plots
            recipient_email: Optional email address to send report to
            
        Returns:
            Dictionary containing pipeline results
        """
        try:
            # Process encrypted data
            decrypted_path = self.process_encrypted_data(
                encrypted_data,
                "decrypted_data"
            )
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_dataset(
                image_paths,
                labels
            )
            
            # Train model
            history = self.train_model(
                train_dataset,
                val_dataset,
                epochs=epochs
            )
            
            # Evaluate model
            evaluation_results = self.evaluate_model(
                test_dataset,
                save_plots=save_plots
            )
            
            # Prepare model info
            model_info = {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'batch_size': self.batch_size,
                'validation_split': self.validation_split
            }
            
            # Generate and send report
            report_path = self.generate_and_send_report(
                evaluation_results,
                history.history,
                model_info,
                recipient_email
            )
            
            return {
                'evaluation_results': evaluation_results,
                'training_history': history.history,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            raise 