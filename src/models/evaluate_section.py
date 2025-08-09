import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)
from sklearn.preprocessing import LabelEncoder

def evaluate_section():
    """Model Evaluation Section"""
    st.subheader("🔍 Model Evaluation")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first!")
        return
    
    if not hasattr(st.session_state, "processed_data"):
        st.error("Please preprocess your data first!")
        return
    
    # Handle both 4-value and 6-value tuple formats
    processed_data = st.session_state.processed_data
    if len(processed_data) == 6:
        _, _, X_train, X_test, y_train, y_test = processed_data
    else:
        X_train, X_test, y_train, y_test = processed_data
    
    model = st.session_state.trained_model
    model_type = st.session_state.model_type
    
    st.info(f"Currently evaluating: {model_type}")
    
    # Determine if it's a deep learning model
    is_dl_model = False
    try:
        is_dl_model = isinstance(model, tf.keras.Model)
    except:
        is_dl_model = False
    
    try:
        # Create tabs for different evaluation aspects
        eval_tabs = st.tabs(["Model Performance", "Feature Analysis", "Predictions"])
        
        with eval_tabs[0]:
            st.markdown("### Model Performance")
            
            # Get predictions
            if is_dl_model:
                # Handle data reshaping for CNN and RNN
                if "CNN" in model_type:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = model.predict(X_test_reshaped)
                elif "RNN" in model_type:
                    sequence_length = min(5, X_test.shape[1])
                    n_features = X_test.shape[1] // sequence_length
                    X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                    y_pred = model.predict(X_test_reshaped)
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Determine problem type
            if isinstance(y_test, pd.Series):
                y_test_values = y_test.values
            else:
                y_test_values = y_test
            
            # Handle reshape for numpy arrays only
            if isinstance(y_test_values, np.ndarray) and len(y_test_values.shape) > 1:
                unique_classes = np.unique(y_test_values.reshape(-1))
            else:
                unique_classes = np.unique(y_test_values)
            
            n_classes = len(unique_classes)
            
            # Classification metrics
            if n_classes > 1:  # Classification
                st.markdown("#### Classification Metrics")
                
                # Convert labels to integers if they're not
                try:
                    # Check if we have a label encoder stored in session state
                    if hasattr(st.session_state, "label_encoder") and st.session_state.label_encoder is not None:
                        le = st.session_state.label_encoder
                        # Make sure all classes in y_test are in the encoder
                        if set(y_test.unique()) - set(le.classes_):
                            # There are new classes, need to refit
                            st.warning("New classes detected in test data. Refitting label encoder.")
                            le = LabelEncoder()
                            le.fit(pd.concat([y_train, y_test]))
                            st.session_state.label_encoder = le
                    else:
                        # No encoder found, create a new one
                        le = LabelEncoder()
                        le.fit(pd.concat([y_train, y_test]))
                        st.session_state.label_encoder = le
                        
                    y_test_encoded = le.transform(y_test)
                except Exception as e:
                    st.warning(f"Error with label encoding: {str(e)}. Creating new encoder.")
                    # Fallback to a new encoder
                    le = LabelEncoder()
                    le.fit(pd.concat([y_train, y_test]))
                    st.session_state.label_encoder = le
                    y_test_encoded = le.transform(y_test)
                
                # Convert predictions to class labels if needed
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Multi-class with probabilities
                    y_pred_class = np.argmax(y_pred, axis=1)
                else:
                    # Binary or already class labels
                    y_pred_class = np.round(y_pred).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_encoded, y_pred_class)
                precision = precision_score(y_test_encoded, y_pred_class, average='weighted', zero_division=0)
                recall = recall_score(y_test_encoded, y_pred_class, average='weighted', zero_division=0)
                f1 = f1_score(y_test_encoded, y_pred_class, average='weighted', zero_division=0)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")
                
                # Calculate loss
                try:
                    # Get predicted probabilities
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                        
                        # Check if binary or multiclass
                        if n_classes == 2:  # Binary classification
                            # Binary cross-entropy loss
                            bce_loss = log_loss(y_test_encoded, y_pred_proba[:, 1])
                            st.metric("Binary Cross-Entropy Loss", f"{bce_loss:.4f}")
                        else:  # Multi-class classification
                            # Categorical cross-entropy loss
                            cce_loss = log_loss(y_test_encoded, y_pred_proba)
                            st.metric("Categorical Cross-Entropy Loss", f"{cce_loss:.4f}")
                except Exception as e:
                    st.warning(f"Could not calculate loss: {str(e)}")
                
                # Display confusion matrix
                st.markdown("##### Confusion Matrix")
                cm = confusion_matrix(y_test_encoded, y_pred_class)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Blues')
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig_cm)
                
                # ROC Curve for binary classification
                if n_classes == 2 and hasattr(model, 'predict_proba'):
                    st.markdown("##### ROC Curve")
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
                        auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
                        
                        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                        ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
                        ax_roc.plot([0, 1], [0, 1], 'k--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('Receiver Operating Characteristic (ROC)')
                        ax_roc.legend(loc='lower right')
                        st.pyplot(fig_roc)
                    except Exception as e:
                        st.warning(f"Could not generate ROC curve: {str(e)}")
            
            # Regression metrics
            else:
                st.markdown("#### Regression Metrics")
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                    st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Actual vs Predicted Plot
                st.markdown("##### Actual vs Predicted Values")
                fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
                ax_reg.scatter(y_test, y_pred, alpha=0.5)
                ax_reg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                ax_reg.set_xlabel('Actual')
                ax_reg.set_ylabel('Predicted')
                ax_reg.set_title('Actual vs Predicted Values')
                st.pyplot(fig_reg)
                
                # Residual Plot
                st.markdown("##### Residual Plot")
                residuals = y_test - y_pred
                fig_res, ax_res = plt.subplots(figsize=(8, 6))
                ax_res.scatter(y_pred, residuals, alpha=0.5)
                ax_res.axhline(y=0, color='r', linestyle='--')
                ax_res.set_xlabel('Predicted')
                ax_res.set_ylabel('Residuals')
                ax_res.set_title('Residual Plot')
                st.pyplot(fig_res)
        
        with eval_tabs[1]:
            st.markdown("### Feature Analysis")
            
            # Display feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### Feature Importance")
                importances = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                sns.barplot(data=importances.head(15), x='Importance', y='Feature', ax=ax_imp)
                ax_imp.set_title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig_imp)
                
                # Display table
                st.dataframe(importances)
            
            elif hasattr(model, 'coef_'):
                # For linear models
                st.markdown("#### Feature Coefficients")
                
                if len(model.coef_.shape) == 1:
                    coefficients = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                else:
                    # For multi-class models
                    coefficients = pd.DataFrame(model.coef_, columns=X_train.columns)
                    coefficients['Class'] = [f'Class {i}' for i in range(model.coef_.shape[0])]
                    coefficients = coefficients.melt(id_vars=['Class'], var_name='Feature', value_name='Coefficient')
                    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
                
                # Plot coefficients
                fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
                sns.barplot(data=coefficients.head(15), x='Coefficient', y='Feature', ax=ax_coef)
                ax_coef.set_title('Feature Coefficients')
                plt.tight_layout()
                st.pyplot(fig_coef)
                
                # Display table
                st.dataframe(coefficients)
            
            else:
                st.info("Feature importance visualization is not available for this type of model.")
        
        with eval_tabs[2]:
            st.markdown("### Predictions")
            
            # Allow user to make predictions on new data
            st.markdown("#### Make Predictions")
            
            # Option 1: Use test data
            if st.checkbox("Use sample from test data"):
                n_samples = min(5, X_test.shape[0])
                sample_indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
                
                X_sample = X_test.iloc[sample_indices]
                y_sample = y_test.iloc[sample_indices]
                
                # Make predictions
                if is_dl_model:
                    if "CNN" in model_type:
                        X_sample_reshaped = X_sample.values.reshape(X_sample.shape[0], X_sample.shape[1], 1)
                        predictions = model.predict(X_sample_reshaped)
                    elif "RNN" in model_type:
                        sequence_length = min(5, X_sample.shape[1])
                        n_features = X_sample.shape[1] // sequence_length
                        X_sample_reshaped = X_sample.values.reshape(X_sample.shape[0], sequence_length, n_features)
                        predictions = model.predict(X_sample_reshaped)
                    else:
                        predictions = model.predict(X_sample)
                else:
                    predictions = model.predict(X_sample)
                
                # Display results
                results = pd.DataFrame({
                    'Actual': y_sample.values,
                    'Predicted': predictions.flatten() if isinstance(predictions, np.ndarray) else predictions
                })
                
                st.dataframe(results)
            
            # Option 2: Manual input
            if st.checkbox("Enter custom values"):
                st.markdown("Enter values for each feature:")
                
                # Create input fields for each feature
                input_data = {}
                for col in X_train.columns:
                    # Get min and max values for numerical features
                    if np.issubdtype(X_train[col].dtype, np.number):
                        min_val = float(X_train[col].min())
                        max_val = float(X_train[col].max())
                        mean_val = float(X_train[col].mean())
                        
                        input_data[col] = st.slider(
                            f"{col}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val
                        )
                    else:
                        # For categorical features
                        unique_values = X_train[col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}", unique_values)
                
                # Create a DataFrame from the input
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                if st.button("Predict"):
                    try:
                        if is_dl_model:
                            if "CNN" in model_type:
                                input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
                                prediction = model.predict(input_reshaped)
                            elif "RNN" in model_type:
                                sequence_length = min(5, input_df.shape[1])
                                n_features = input_df.shape[1] // sequence_length
                                input_reshaped = input_df.values.reshape(1, sequence_length, n_features)
                                prediction = model.predict(input_reshaped)
                            else:
                                prediction = model.predict(input_df)
                        else:
                            prediction = model.predict(input_df)
                        
                        # Display prediction
                        st.success(f"Prediction: {prediction[0]}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred during evaluation: {str(e)}")
        st.info("Please check your model and data compatibility.")
