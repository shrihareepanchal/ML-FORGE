# Model evaluation and inference
import datetime
import tempfile
from datetime import datetime as dt

def model_evaluation(state):
    """Handle model evaluation and visualization"""
    st.markdown("### 📊 Model Evaluation")
    
    # Create example evaluation metrics (in a real app, these would come from actual model evaluation)
    if "eval_metrics" not in state:
        # Create simulated evaluation metrics for demonstration
        create_sample_evaluation_metrics(state)
    
    # Display evaluation metrics in multiple tabs
    eval_tabs = st.tabs(["Performance Metrics", "Confusion Matrix", "Sample Predictions"])
    
    with eval_tabs[0]:
        display_performance_metrics(state)
    
    with eval_tabs[1]:
        display_confusion_matrix(state)
    
    with eval_tabs[2]:
        display_sample_predictions(state)
    
    # Option to generate evaluation report
    if st.button("Generate Evaluation Report"):
        with st.spinner("Generating comprehensive evaluation report..."):
            # Simulate report generation
            time.sleep(2)
            
            # In a real implementation, this would generate a detailed report
            report_html = create_evaluation_report(state)
            
            # Provide download link
            st.markdown("### 📥 Download Report")
            st.download_button(
                label="Download Evaluation Report (HTML)",
                data=report_html,
                file_name=f"cnn_evaluation_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

def create_sample_evaluation_metrics(state):
    """Create sample evaluation metrics for demonstration"""
    num_classes = len(state["classes"])
    
    # Classification metrics
    if num_classes == 2:  # Binary classification
        metrics = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "auc": 0.95
        }
    else:  # Multi-class classification
        metrics = {
            "accuracy": 0.88,
            "macro_precision": 0.86,
            "macro_recall": 0.87,
            "macro_f1": 0.86,
            "weighted_f1": 0.88
        }
    
    # Confusion matrix - random values for demonstration
    cm = np.random.randint(0, 30, size=(num_classes, num_classes))
    # Make diagonal values higher (correct predictions)
    for i in range(num_classes):
        cm[i, i] = np.random.randint(70, 100)
    
    # Sample predictions - random values for demonstration
    num_samples = 10
    sample_indices = np.random.randint(0, 100, size=num_samples)
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Predicted probabilities - random values, but make predicted class probability higher
    pred_probs = np.random.rand(num_samples, num_classes) * 0.3
    pred_class = np.random.randint(0, num_classes, size=num_samples)
    for i in range(num_samples):
        pred_probs[i, pred_class[i]] += 0.7
    
    # Normalize probabilities to sum to 1
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
    
    # Store all metrics in state
    state["eval_metrics"] = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "sample_predictions": {
            "indices": sample_indices,
            "true_labels": true_labels,
            "pred_probs": pred_probs,
            "pred_class": pred_class
        }
    }

def display_performance_metrics(state):
    """Display model performance metrics"""
    metrics = state["eval_metrics"]["metrics"]
    
    st.markdown("#### Performance Metrics")
    
    # Create a visually appealing metrics display
    num_classes = len(state["classes"])
    
    if num_classes == 2:  # Binary classification
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col1.metric("AUC-ROC", f"{metrics['auc']:.2%}")
        
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col2.metric("Recall", f"{metrics['recall']:.2%}")
        
        col3.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        
        # Plot ROC curve (simulated for demonstration)
        st.markdown("#### ROC Curve")
        
        # Generate a simulated ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.5)  # Simple power function to simulate ROC curve
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {metrics["auc"]:.2f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
    else:  # Multi-class classification
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
        col3.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
        
        st.markdown("#### Class-wise Metrics")
        
        # Create simulated class-wise metrics
        class_metrics = []
        for i, cls in enumerate(state["classes"]):
            class_metrics.append({
                "Class": cls,
                "Precision": np.random.uniform(0.8, 0.95),
                "Recall": np.random.uniform(0.8, 0.95),
                "F1-Score": np.random.uniform(0.8, 0.95)
            })
        
        # Display as a table
        class_df = pd.DataFrame(class_metrics)
        st.dataframe(class_df)
        
        # Plot class-wise F1 scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(class_df["Class"], class_df["F1-Score"], color='skyblue')
        ax.set_title("F1-Score by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("F1-Score")
        ax.axhline(y=class_df["F1-Score"].mean(), color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean F1: {class_df["F1-Score"].mean():.2f}')
        ax.legend()
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)

def display_confusion_matrix(state):
    """Display confusion matrix visualization"""
    cm = state["eval_metrics"]["confusion_matrix"]
    classes = state["classes"]
    
    st.markdown("#### Confusion Matrix")
    
    # Display the confusion matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Explanation of the confusion matrix
    st.markdown("""
    **Understanding the Confusion Matrix:**
    - Each row represents the instances of a true class
    - Each column represents the instances of a predicted class
    - The diagonal cells show correct predictions
    - Off-diagonal cells show incorrect predictions
    """)
    
    # Calculate and display normalized confusion matrix
    st.markdown("#### Normalized Confusion Matrix")
    
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_sample_predictions(state):
    """Display sample predictions with visualizations"""
    samples = state["eval_metrics"]["sample_predictions"]
    classes = state["classes"]
    
    st.markdown("#### Sample Predictions")
    st.markdown("Visualizing how the model performs on sample images from the test set:")
    
    # In a real implementation, we would load and display actual images
    # For this simulation, we'll create a table of predictions
    
    # Create a DataFrame for sample predictions
    sample_data = []
    
    for i in range(len(samples["indices"])):
        true_label = samples["true_labels"][i]
        pred_class = samples["pred_class"][i]
        
        # Get the confidence score for the predicted class
        confidence = samples["pred_probs"][i, pred_class]
        
        # Determine if prediction is correct
        is_correct = true_label == pred_class
        
        sample_data.append({
            "Sample": f"Sample {i+1}",
            "True Class": classes[true_label],
            "Predicted Class": classes[pred_class],
            "Confidence": confidence,
            "Correct": is_correct
        })
    
    # Convert to DataFrame
    samples_df = pd.DataFrame(sample_data)
    
    # Style the dataframe to highlight correct/incorrect predictions
    def highlight_correct(val):
        if val == True:
            return 'background-color: #CCFFCC'  # Light green
        elif val == False:
            return 'background-color: #FFCCCC'  # Light red
        return ''
    
    # Display styled dataframe
    st.dataframe(samples_df.style.applymap(highlight_correct, subset=['Correct']))
    
    # Display a bar chart of confidence scores
    st.markdown("#### Prediction Confidence Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar colors based on correct/incorrect
    colors = ['green' if c else 'red' for c in samples_df['Correct']]
    
    bars = ax.bar(samples_df['Sample'], samples_df['Confidence'], color=colors)
    ax.set_title("Prediction Confidence by Sample")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct Prediction'),
        Patch(facecolor='red', label='Incorrect Prediction')
    ]
    ax.legend(handles=legend_elements)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_evaluation_report(state):
    """Generate HTML report for model evaluation"""
    # This is a simple HTML template for demonstration
    # In a real implementation, this would be more comprehensive
    
    model_name = state.get("architecture", "CNN Model")
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format metrics for report
    metrics = state["eval_metrics"]["metrics"]
    metrics_html = ""
    
    for key, value in metrics.items():
        formatted_key = key.replace("_", " ").title()
        metrics_html += f"<tr><td>{formatted_key}</td><td>{value:.4f}</td></tr>"
    
    # Basic HTML report template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CNN Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .metric-value {{ font-weight: bold; }}
            .footer {{ margin-top: 40px; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CNN Model Evaluation Report</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <h2>Dataset Information</h2>
            <p>Number of Classes: {len(state['classes'])}</p>
            <p>Classes: {', '.join(state['classes'])}</p>
            
            <h2>Model Architecture</h2>
            <pre>{model_name} with custom configuration</pre>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {metrics_html}
            </table>
            
            <h2>Conclusion</h2>
            <p>This model achieved good performance across the evaluation metrics. The accuracy and F1 score indicate that the model is effective at classifying the images in the dataset.</p>
            
            <div class="footer">
                <p>Generated by ML-FORGE CNN Module</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

def model_deployment(state):
    """Handle model deployment and inference"""
    st.markdown("### 🚀 Model Deployment & Inference")
    
    deploy_tabs = st.tabs(["Save Model", "Load Model", "Run Inference"])
    
    with deploy_tabs[0]:
        model_saving(state)
    
    with deploy_tabs[1]:
        model_loading(state)
    
    with deploy_tabs[2]:
        run_inference(state)

def model_saving(state):
    """Handle model saving and export"""
    st.markdown("#### Save Trained Model")
    
    # Model format options
    save_format = st.radio(
        "Model Save Format",
        ["Keras H5 (.h5)", "TensorFlow SavedModel (.pb)", "ONNX Format (.onnx)"]
    )
    
    # Model name
    model_name = st.text_input(
        "Model Name",
        value=f"cnn_model_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Extra options
    include_weights = st.checkbox("Include Weights", value=True)
    include_optimizer = st.checkbox("Include Optimizer State", value=False)
    
    # Save model button
    if st.button("Save Model"):
        with st.spinner("Saving model..."):
            # Simulate saving the model
            time.sleep(2)
            
            # In a real implementation, this would save the actual model
            # For demonstration, we'll just show a success message
            
            # Show different messages based on format
            if save_format == "Keras H5 (.h5)":
                st.success(f"✅ Model successfully saved as {model_name}.h5")
            elif save_format == "TensorFlow SavedModel (.pb)":
                st.success(f"✅ Model successfully saved as {model_name}_savedmodel")
            else:  # ONNX
                st.success(f"✅ Model successfully saved as {model_name}.onnx")
            
            # Provide a mock download button
            st.download_button(
                label="Download Model",
                data=f"This would be the actual model file for {model_name}",
                file_name=f"{model_name}.h5" if save_format == "Keras H5 (.h5)" else f"{model_name}.onnx",
                mime="application/octet-stream"
            )
    
    # Export model architecture
    st.markdown("#### Export Model Architecture")
    
    if st.button("Export Architecture Summary"):
        # Generate a summary of the architecture
        if state["model"] is not None:
            summary_io = io.StringIO()
            state["model"].summary(print_fn=lambda x: summary_io.write(x + '\n'))
            summary_string = summary_io.getvalue()
            summary_io.close()
            
            # Provide download button for the summary
            st.download_button(
                label="Download Architecture Summary",
                data=summary_string,
                file_name=f"{model_name}_architecture.txt",
                mime="text/plain"
            )
    
    # Export training configuration
    st.markdown("#### Export Training Configuration")
    
    if st.button("Export Training Configuration"):
        # Convert training configuration to JSON
        config = {
            "model_architecture": state.get("architecture", "Custom CNN"),
            "image_size": state.get("image_size", (224, 224)),
            "classes": state.get("classes", []),
            "training_parameters": state.get("training_config", {})
        }
        
        config_json = str(config)  # In a real app, use json.dumps()
        
        # Provide download button for the configuration
        st.download_button(
            label="Download Training Configuration",
            data=config_json,
            file_name=f"{model_name}_config.json",
            mime="application/json"
        )

def model_loading(state):
    """Handle model loading functionality"""
    st.markdown("#### Load Pretrained Model")
    
    # Upload model file
    uploaded_model = st.file_uploader("Upload Trained Model", type=["h5", "pb", "onnx"])
    
    if uploaded_model is not None:
        # Model configuration after upload
        st.markdown("#### Model Configuration")
        
        # Class mapping
        st.markdown("**Class Mapping**")
        st.markdown("Enter class labels (comma-separated):")
        
        class_input = st.text_input(
            "Class Labels",
            value=",".join(state.get("classes", [])) if state.get("classes") else "class1,class2,class3"
        )
        
        classes = [c.strip() for c in class_input.split(",")]
        
        # Image size for the model
        st.markdown("**Input Image Size**")
        
        col1, col2 = st.columns(2)
        with col1:
            img_width = st.number_input("Width", min_value=16, max_value=512, value=224)
        with col2:
            img_height = st.number_input("Height", min_value=16, max_value=512, value=224)
        
        # Load model button
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                # Simulate loading the model
                time.sleep(2)
                
                # In a real implementation, this would load the actual model
                # For demonstration, we'll just update the state
                
                state["classes"] = classes
                state["image_size"] = (img_height, img_width)
                state["model_loaded"] = True
                
                st.success(f"✅ Model successfully loaded with {len(classes)} classes")
                
                # Show the expected input shape
                st.info(f"Expected input shape: ({img_height}, {img_width}, 3)")
                
                # Show class mapping
                st.markdown("**Class Mapping:**")
                for i, cls in enumerate(classes):
                    st.markdown(f"Class {i}: {cls}")

def run_inference(state):
    """Handle inference on new images"""
    st.markdown("#### Run Inference on New Images")
    
    # Check if model is trained or loaded
    if not state.get("model_trained", False) and not state.get("model_loaded", False):
        st.warning("Please train a model or load a pretrained model first")
        return
    
    # Upload images for inference
    st.markdown("Upload images for prediction:")
    uploaded_files = st.file_uploader(
        "Upload Images for Inference",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Inference options
        st.markdown("#### Inference Options")
        
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        topk = st.slider("Show Top-K Predictions", min_value=1, max_value=5, value=3)
        
        # Run inference button
        if st.button("Run Inference"):
            with st.spinner("Running inference..."):
                # In a real implementation, we would use the actual model
                # For demonstration, we'll display random predictions
                
                for i, img_file in enumerate(uploaded_files):
                    # Display the image and predictions side by side
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Display the image
                        img = Image.open(img_file)
                        st.image(img, caption=f"Image {i+1}: {img_file.name}", use_column_width=True)
                    
                    with col2:
                        # Generate simulated predictions
                        sim_preds = simulate_predictions(state["classes"], topk)
                        
                        # Display predictions
                        st.markdown(f"**Predictions for Image {i+1}:**")
                        
                        # Create a DataFrame for predictions
                        pred_df = pd.DataFrame(sim_preds)
                        
                        if show_confidence:
                            # Display predictions with confidence bar chart
                            fig, ax = plt.subplots(figsize=(8, 3))
                            bars = ax.barh(pred_df["Class"], pred_df["Confidence"], color="skyblue")
                            ax.set_xlim(0, 1)
                            ax.set_xlabel("Confidence")
                            ax.set_title(f"Top {topk} Predictions")
                            
                            # Add confidence values
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                       f'{width:.2f}', ha='left', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            # Just show the top prediction
                            st.success(f"Predicted: {sim_preds[0]['Class']} ({sim_preds[0]['Confidence']:.2f})")

def simulate_predictions(classes, topk):
    """Simulate model predictions for demonstration"""
    # Generate random probabilities
    probs = np.random.rand(len(classes))
    
    # Normalize to sum to 1
    probs = probs / probs.sum()
    
    # Sort to get top-k
    sorted_indices = np.argsort(probs)[::-1][:topk]
    
    predictions = []
    for i in sorted_indices:
        predictions.append({
            "Class": classes[i],
            "Confidence": probs[i]
        })
    
    return predictions
