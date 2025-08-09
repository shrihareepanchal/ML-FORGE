# 🚀 ML-FORGE: Machine Learning Explorer & Reporting Tool

ML-FORGE is a comprehensive machine learning application designed to make the data science workflow more accessible and efficient. It provides a user-friendly interface for data processing, model training, and evaluation, all in a streamlined environment.

### 🎯 Why ML-FORGE?

🔧 **No-code preprocessing**  
📊 **Visual EDA**  
🧠 **ML/DL model training**  
📈 **Interactive evaluation**  
📄 **One-click PDF reporting**  
💬 **Built-in ML assistant powered by LLaMA 3.2**

---

## ✨ Key Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🗃️ **Data Upload & Preprocessing**     | Upload CSVs, handle missing values, encode categories & normalize features |
| 📊 **Exploratory Data Analysis** | Visualize your data using scatter, heatmap, violin, and more                |
| ⚙️ **Model Training**           | Train models like KNN, SVM, RF, GBM, ElasticNet with custom hyperparameters |
| 🧠 **Deep Learning Support**     | Build ANN/RNN/CNN using TensorFlow or PyTorch                               |
| 🖼️ **CNN Image Classifier**     | Upload image data and train CNN pipelines end-to-end                        |
| 📈 **Performance Evaluation**    | See precision, recall, ROC, F1, AUC, and feature importances                |
| 📄 **Automated Reports**         | Export PDF reports with metrics, visuals, and configuration summaries       |
| 🤖 **AI Assistant (LLaMA 3.2)**  | Ask ML questions inside the app powered by local or API-based LLMs         |

---
## 🗂️ Project Structure

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

## 📦 Requirements

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

## 🚀 Installation

1. Clone the repository
   ```
   git clone https://github.com/shrihareepanchal/ml-forge.git
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

## 🧭 How to Use

Navigate through the sidebar:

- 📂 Data Upload – Load CSV or image datasets
- 🧹 Preprocessing – Clean and transform your data
- 📊 EDA – Explore data through visualizations
- 🤖 Model Training – Train ML/DL models
- 📈 Evaluation – Assess model performance
- 📄 Reporting – Export detailed PDF reports

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Streamlit team for their amazing framework
- The open-source ML community for inspiration
