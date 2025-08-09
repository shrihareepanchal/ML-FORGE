"""
Standalone CNN pipeline module for image classification
"""

# Import only the standalone pipeline
from src.models.cnn.app_integration import standalone_cnn_pipeline

__all__ = [
    'standalone_cnn_pipeline'
] 