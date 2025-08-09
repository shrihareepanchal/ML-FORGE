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

## Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
xgboost
lightgbm
joblib
openpyxl
plotly
reportlab
llama-cpp-python (optional, for Chat Assistant feature)
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ml-forge.git
   cd ml-forge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For the Chat Assistant feature, install llama-cpp-python:
   ```bash
   pip install llama-cpp-python
   ```
   
   For GPU acceleration (recommended):
   ```bash
   pip install llama-cpp-python --prefer-binary --extra-index-url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
   ```

4. Download a Llama model:
   - Get a GGUF format Llama model (recommended: llama-3.2-8b-instruct.Q4_K_M.gguf)
   - Place it in a directory of your choice
   - See [llama_setup_guide.md](llama_setup_guide.md) for detailed instructions

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Navigate through the different sections:
   - Start by uploading your dataset in the **Data Loading** section
   - Process your data using the **Data Processing** tools
   - Explore your data with the **EDA** tools
   - Train models using either **ML Model Training** or **DL Model Training** sections
   - Evaluate your models' performance in the **Evaluate** section
   - Generate reports in the **Report** section
   - Ask questions about ML concepts in the **ML Chat Assistant** section

## Chat Assistant

The ML Chat Assistant feature provides an interactive way to learn about machine learning concepts through natural language conversation. It uses a locally running Llama 3.2 model to answer your ML-related questions.

### Setup

1. Install llama-cpp-python (see Installation section above)
2. Download a Llama 3.2 model in GGUF format
3. In the Chat Assistant section, configure the model settings:
   - Set the path to your model file
   - Adjust parameters like GPU layers, context length, and batch size
   - Click "Load Model" to initialize the chat

### Usage Tips

- Ask specific questions about machine learning concepts
- Use the chat to guide your understanding of model selection and evaluation
- Clear the chat history when starting a new topic
- If you encounter memory issues, reduce the context length or batch size

## CNN Image Classification

The CNN section allows you to:

1. Upload and preprocess image datasets
2. Visualize sample images from your dataset
3. Train CNN models with customizable architectures
4. Evaluate model performance on test data
5. Make predictions on new images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please contact:
- Email: your.email@example.com
- GitHub: [yourusername](https://github.com/yourusername)

## Screenshots

Here are some screenshots of ML-FORGE in action:

### Data Loading and Preprocessing
![Data Loading](images/data_loading.png)

### Model Training
![Model Training](images/model_training.png)

### Model Evaluation
![Model Evaluation](images/model_evaluation.png)

### Analysis Report
![Analysis Report](images/analysis_report.png)

## Deployment

ML-FORGE can be deployed using:

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [AWS](https://aws.amazon.com)
- [Google Cloud Platform](https://cloud.google.com)

See the deployment section below for detailed instructions.
