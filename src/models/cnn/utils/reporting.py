import os
from typing import Dict, List, Optional
import logging
import datetime
from datetime import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path

class ReportGenerator:
    def __init__(
        self,
        report_dir: str = "reports",
        email_config: Optional[Dict] = None
    ):
        """
        Initialize the report generator.
        
        Args:
            report_dir: Directory to save reports
            email_config: Optional email configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.report_dir = report_dir
        self.email_config = email_config or {}
        
        # Create report directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
        
    def generate_report(
        self,
        metrics: Dict[str, float],
        classification_report: str,
        model_info: Dict,
        training_history: Dict,
        plots: List[str]
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            classification_report: Classification report string
            model_info: Dictionary containing model information
            training_history: Dictionary containing training history
            plots: List of paths to generated plots
            
        Returns:
            Path to the generated report file
        """
        try:
            # Create report filename with timestamp
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"cnn_evaluation_report_{timestamp}.html"
            report_path = os.path.join(self.report_dir, report_filename)
            
            # Generate HTML report
            html_content = self._generate_html_report(
                metrics,
                classification_report,
                model_info,
                training_history,
                plots
            )
            
            # Save report
            with open(report_path, 'w') as f:
                f.write(html_content)
                
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
            
    def _generate_html_report(
        self,
        metrics: Dict[str, float],
        classification_report: str,
        model_info: Dict,
        training_history: Dict,
        plots: List[str]
    ) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            classification_report: Classification report string
            model_info: Dictionary containing model information
            training_history: Dictionary containing training history
            plots: List of paths to generated plots
            
        Returns:
            HTML content as string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CNN Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin: 10px 0; }}
                .plot {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background-color: #f5f5f5; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>CNN Model Evaluation Report</h1>
            <p>Generated on: {dt.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Model Information</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in model_info.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Evaluation Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {''.join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Classification Report</h2>
                <pre>{classification_report}</pre>
            </div>
            
            <div class="section">
                <h2>Training History</h2>
                <table>
                    <tr><th>Epoch</th><th>Training Loss</th><th>Validation Loss</th><th>Training Accuracy</th><th>Validation Accuracy</th></tr>
                    {''.join(f"<tr><td>{i+1}</td><td>{training_history['loss'][i]:.4f}</td><td>{training_history['val_loss'][i]:.4f}</td><td>{training_history['accuracy'][i]:.4f}</td><td>{training_history['val_accuracy'][i]:.4f}</td></tr>" for i in range(len(training_history['loss'])))}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {''.join(f'<div class="plot"><img src="{plot}" alt="Plot"></div>' for plot in plots)}
            </div>
        </body>
        </html>
        """
        return html
        
    def send_report(
        self,
        report_path: str,
        recipient_email: str,
        subject: str = "CNN Model Evaluation Report"
    ) -> None:
        """
        Send the generated report via email.
        
        Args:
            report_path: Path to the report file
            recipient_email: Email address of the recipient
            subject: Email subject line
        """
        try:
            if not self.email_config:
                self.logger.warning("Email configuration not provided. Skipping email sending.")
                return
                
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('sender_email')
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            body = "Please find attached the CNN model evaluation report."
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report
            with open(report_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='html')
                attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=os.path.basename(report_path)
                )
                msg.attach(attachment)
                
            # Send email
            with smtplib.SMTP(
                self.email_config.get('smtp_server'),
                self.email_config.get('smtp_port')
            ) as server:
                server.starttls()
                server.login(
                    self.email_config.get('username'),
                    self.email_config.get('password')
                )
                server.send_message(msg)
                
            self.logger.info(f"Report sent successfully to {recipient_email}")
            
        except Exception as e:
            self.logger.error(f"Error sending report: {str(e)}")
            raise 