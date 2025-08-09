# ML-FORGE: Machine Learning Explorer & Reporting Tool

ML-FORGE is a comprehensive machine learning application designed to make the data science workflow more accessible and efficient. It provides a user-friendly interface for data processing, model training, and evaluation, all in a streamlined environment.

## Features

- **Data Loading & Processing**: Support for various data formats with automated preprocessing options
- **Exploratory Data Analysis (EDA)**: Interactive visualizations to understand your data
- **Machine Learning Model Training**: Train multiple ML models with customizable parameters
- **Deep Learning Model Training**: Build neural networks with TensorFlow
- **CNN Image Classification**: End-to-end image processing and CNN model training
- **Model Evaluation**: Comprehensive model performance metrics
- **Chat Assistant**: Built-in chat interface powered by Llama 3.2 for ML guidance
- **Exportable Reports**: Generate PDF reports of your ML experiments

## Project Structure

```
ML-FORGE/
│
├── app.py                 # Main Streamlit application
├── main.py                # Entry point for the application
├── requirements.txt       # Project dependencies
│
├── data/                  # Data storage
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data and reports
│
├── docs/                  # Documentation
│   ├── CNN_MODULE_README.md
│   ├── DEPLOYMENT.md
│   └── llama_setup_guide.md
│
├── images/                # Static images used in the app
│   ├── ml_xpert_logo.png
│   └── no_data_image.png
│
└── src/                   # Source code
    ├── core/              # Core application logic
    │   └── preprocess_update.py
    │
    ├── models/            # ML model implementations
    │   ├── ml_model_training.py
    │   ├── evaluate_section.py
    │   ├── cnn_module.py
    │   ├── cnn_module_part2.py
    │   └── cnn_module_part3.py
    │
    ├── utils/             # Utility functions
    │
    └── visualization/     # Data visualization components
```

## Requirements

```
streamlit>=1.30.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.13.0
tensorflow>=2.12.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ml-forge.git
   cd ml-forge
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

After launching the app, navigate through the sidebar to access different sections:

1. **Data Loading**: Upload your dataset
2. **Preprocessing**: Clean and transform your data
3. **EDA**: Explore your data with visualizations
4. **Model Training**: Train ML models on your data
5. **Evaluation**: Evaluate model performance
6. **Report Generation**: Create comprehensive reports

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Streamlit team for their amazing framework
- The open-source ML community for inspiration
