import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    log_loss,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import roc_auc_score

# Check for optional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def save_and_download_model(model, le=None, model_name=""):
    """Helper function to save and create download button for models"""
    st.markdown("##### Model Download")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        if le is not None:
            # For classification models, save both model and label encoder
            joblib.dump((model, le), tmp.name)
        else:
            # For regression models, save only the model
            joblib.dump(model, tmp.name)
        with open(tmp.name, "rb") as f:
            model_bytes = f.read()
        st.download_button(
            label=f"Download {model_name}",
            data=model_bytes,
            file_name=f"{model_name.lower().replace(' ', '_')}.joblib",
            mime="application/octet-stream",
        )


def calculate_and_display_loss(model, X_test, y_test):
    """Calculate and display loss metrics for classification models"""
    st.markdown("##### Loss Metrics:")
    try:
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_test)
        
        # Check if binary or multiclass
        if len(np.unique(y_test)) == 2:  # Binary classification
            # Binary cross-entropy loss
            bce_loss = log_loss(y_test, y_pred_proba[:, 1])
            st.write(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")
        else:  # Multi-class classification
            # Categorical cross-entropy loss
            cce_loss = log_loss(y_test, y_pred_proba)
            st.write(f"Categorical Cross-Entropy Loss: {cce_loss:.4f}")
    except Exception as e:
        st.warning(f"Could not calculate loss: {str(e)}")


def ml_model_training_section(X_train, X_test, y_train, y_test):
    st.markdown("## ⚙️ ML Model Training Section")

    # If data is not properly loaded, show warning
    if X_train is None or y_train is None:
        st.warning("Please complete data preprocessing first!")
        return

    # Store the label encoder for classification tasks
    if len(np.unique(y_train)) > 1 and not np.issubdtype(y_train.dtype, np.number):
        le = LabelEncoder()
        le.fit(pd.concat([y_train, y_test]))
        st.session_state.label_encoder = le

    problem_type = st.radio(
        "What type of Machine Learning problem are you trying to solve?",
        ("Regression", "Classification"),
    )

    if problem_type == "Regression":
        st.markdown("### Regression Models")
        regression_models = [
            "Linear Regression",
            "Elastic Net Regression",
            "Decision Tree Regression",
            "Random Forest Regression",
            "Gradient Boosting Regression",
            "KNN Regression",
            "SVR Regression",
        ]
        selected_model = st.selectbox("Select Regression Model", regression_models)

        if selected_model == "Linear Regression":
            st.markdown("#### Linear Regression")
            calculate_intercept = st.selectbox(
                "Do you want to calculate intercept for this model (default = True)",
                [True, False],
                index=0,
            )
            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in Linear Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Linear Regression Model"):
                    model = LinearRegression(fit_intercept=calculate_intercept)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store the trained model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Linear Regression"

                    st.success("Linear Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                # Only show download button if model is trained
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Linear Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="Linear Regression"
                    )

        elif selected_model == "Elastic Net Regression":
            st.markdown("#### Elastic Net Regression")
            calculate_intercept = st.selectbox(
                "Do you want to calculate intercept for this model? (default = True)",
                [True, False],
                index=0,
            )
            l1_ratio = st.number_input(
                "Select l1_ratio (min_value = 0, max_value = 1, default = 0.5). For l1_ratio = 1, it is an L1 penalty and l1_ratio = 0, it is a L2 penalty",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
            max_iter = st.number_input(
                "Select the maximum number of iterations (min_value = 1, max_value = 10000, default = 5000)",
                min_value=1,
                max_value=10000,
                value=5000,
                step=100,
            )
            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in Elastic Net Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Elastic Net Regression Model"):
                    model = ElasticNet(
                        fit_intercept=calculate_intercept,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Elastic Net Regression"

                    st.success("Elastic Net Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Elastic Net Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model,
                        model_name="Elastic Net Regression",
                    )

        elif selected_model == "Decision Tree Regression":
            st.markdown("#### Decision Tree Regression")
            criterion_options = [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ]
            criterion = st.selectbox(
                "Select one of the criteria to measure quality of split from below (default = squared_error)",
                criterion_options,
                index=criterion_options.index("squared_error"),
            )
            max_depth = st.number_input(
                "Select maximum depth of the tree (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            min_samples_split = st.number_input(
                "Select the minimum number of samples required to split an internal node (min_value = 1, max_value = 200, default = 2)",
                min_value=1,
                max_value=200,
                value=2,
                step=1,
            )
            min_samples_leaf = st.number_input(
                "Select the minimum number of samples required to be at a leaf node (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            max_leaf_nodes = st.number_input(
                "Select the maximum number of leaf nodes (min_value = 2, max_value = 200, default = 2)",
                min_value=2,
                max_value=200,
                value=2,
                step=1,
            )

            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in Decision Tree Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Decision Tree Regression Model"):
                    model = DecisionTreeRegressor(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_leaf_nodes=max_leaf_nodes,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Decision Tree Regression"

                    st.success("Decision Tree Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Decision Tree Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model,
                        model_name="Decision Tree Regression",
                    )

        elif selected_model == "Random Forest Regression":
            st.markdown("#### Random Forest Regression")
            n_estimators = st.number_input(
                "Select the number of trees in the forest (min_value = 1, max_value = 500, default = 100)",
                min_value=1,
                max_value=500,
                value=100,
                step=10,
            )
            criterion_options = [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ]
            criterion = st.selectbox(
                "Select one of the criteria to measure quality of split from below (default = squared_error)",
                criterion_options,
                index=criterion_options.index("squared_error"),
            )
            max_depth = st.number_input(
                "Select maximum depth of the tree (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            min_samples_split = st.number_input(
                "Select the minimum number of samples required to split an internal node (min_value = 1, max_value = 200, default = 2)",
                min_value=1,
                max_value=200,
                value=2,
                step=1,
            )
            min_samples_leaf = st.number_input(
                "Select the minimum number of samples required to be at a leaf node (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            max_leaf_nodes = st.number_input(
                "Select the maximum number of leaf nodes (min_value = 2, max_value = 200, default = 2)",
                min_value=2,
                max_value=200,
                value=2,
                step=1,
            )
            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in Random Forest Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Random Forest Regression Model"):
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_leaf_nodes=max_leaf_nodes,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Random Forest Regression"

                    st.success("Random Forest Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Random Forest Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model,
                        model_name="Random Forest Regression",
                    )

        elif selected_model == "Gradient Boosting Regression":
            st.markdown("#### Gradient Boosting Regression")

            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in Gradient Boosting Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Gradient Boosting Regression Model"):
                    model = GradientBoostingRegressor(
                        random_state=42
                    )  # Using default parameters for brevity as per UI
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Gradient Boosting Regression"

                    st.success("Gradient Boosting Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Gradient Boosting Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model,
                        model_name="Gradient Boosting Regression",
                    )

        elif selected_model == "KNN Regression":
            st.markdown("#### KNN Regression")
            n_neighbors = st.number_input(
                "Select the number of neighbors (min_value = 5, max_value = 200, default = 5)",
                min_value=5,
                max_value=200,
                value=5,
                step=1,
            )
            weights_options = ["uniform", "distance"]
            weights = st.selectbox(
                "Select the weight function to be used in parameter (default = uniform).",
                weights_options,
                index=weights_options.index("uniform"),
            )
            algorithm_options = ["auto", "ball_tree", "kd_tree", "brute"]
            algorithm = st.selectbox(
                "Select the algorithm to be used to compute the nearest neighbors (default = auto).",
                algorithm_options,
                index=algorithm_options.index("auto"),
            )

            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in KNN Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train KNN Regression Model"):
                    model = KNeighborsRegressor(
                        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "KNN Regression"

                    st.success("KNN Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "KNN Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="KNN Regression"
                    )

        elif selected_model == "SVR Regression":
            st.markdown("#### SVR Regression")
            kernel_options = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
            kernel = st.selectbox(
                "Select the kernel type to be used in the algorithm (default = rbf).",
                kernel_options,
                index=kernel_options.index("rbf"),
            )
            degree = st.number_input(
                "Select the degree of the polynomial kernel function (min_value = 1, max_value = 100, default = 3)",
                min_value=1,
                max_value=100,
                value=3,
                step=1,
            )
            max_iter = st.number_input(
                "Select the maximum number of iterations (min_value = -1, max_value = 2000, default = -1)",
                min_value=-1,
                max_value=2000,
                value=-1,
                step=100,
            )

            loss_functions = st.multiselect(
                "Select the loss functions to calculate the loss in SVR Regression",
                ["MSE", "MAE", "RMSE"],
                default=["MSE"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train SVR Regression Model"):
                    model = SVR(kernel=kernel, degree=degree, max_iter=max_iter)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "SVR Regression"

                    st.success("SVR Regression Model Trained!")

                    # Calculate and Display Loss Functions
                    st.markdown("##### Performance Metrics:")
                    for loss_fn in loss_functions:
                        if loss_fn == "MSE":
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                        elif loss_fn == "MAE":
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                        elif loss_fn == "RMSE":
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared (R2): {r2:.4f}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "SVR Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="SVR Regression"
                    )

    elif problem_type == "Classification":
        st.markdown("### Classification Models")
        classification_models = [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Gradient Boosting Classifier",
        ]

        if XGBOOST_AVAILABLE:
            classification_models.append("XGBoost Classifier")
        if LIGHTGBM_AVAILABLE:
            classification_models.append("LightGBM Classifier")

        selected_model = st.selectbox(
            "Select Classification Model", classification_models
        )

        if selected_model == "Logistic Regression":
            st.markdown("#### Logistic Regression")
            solver_options = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            solver = st.selectbox(
                "Select the algorithm to use in the optimization problem (default = lbfgs)",
                solver_options,
                index=solver_options.index("lbfgs"),
            )
            max_iter = st.number_input(
                "Select the maximum number of iterations (min_value = 1, max_value = 2000, default = 100)",
                min_value=1,
                max_value=2000,
                value=100,
                step=10,
            )
            multi_class_options = ["auto", "ovr", "multinomial"]
            multi_class = st.selectbox(
                "Select the multi-class strategy (default = auto)",
                multi_class_options,
                index=multi_class_options.index("auto"),
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Logistic Regression Model"):
                    model = LogisticRegression(
                        solver=solver,
                        max_iter=max_iter,
                        multi_class=multi_class,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Logistic Regression"

                    st.success("Logistic Regression Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Logistic Regression"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="Logistic Regression"
                    )

        elif selected_model == "Decision Tree Classifier":
            st.markdown("#### Decision Tree Classifier")
            criterion_options = ["gini", "entropy", "log_loss"]
            criterion = st.selectbox(
                "Select the function to measure the quality of a split (default = gini)",
                criterion_options,
                index=criterion_options.index("gini"),
            )
            max_depth = st.number_input(
                "Select maximum depth of the tree (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            min_samples_split = st.number_input(
                "Select the minimum number of samples required to split an internal node (min_value = 2, max_value = 200, default = 2)",
                min_value=2,
                max_value=200,
                value=2,
                step=1,
            )
            min_samples_leaf = st.number_input(
                "Select the minimum number of samples required to be at a leaf node (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Decision Tree Classifier Model"):
                    model = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Decision Tree Classifier"

                    st.success("Decision Tree Classifier Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Decision Tree Classifier"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="Decision Tree Classifier"
                    )

        elif selected_model == "Random Forest Classifier":
            st.markdown("#### Random Forest Classifier")
            n_estimators = st.number_input(
                "Select the number of trees in the forest (min_value = 1, max_value = 500, default = 100)",
                min_value=1,
                max_value=500,
                value=100,
                step=10,
            )
            criterion_options = ["gini", "entropy", "log_loss"]
            criterion = st.selectbox(
                "Select the function to measure the quality of a split (default = gini)",
                criterion_options,
                index=criterion_options.index("gini"),
            )
            max_depth = st.number_input(
                "Select maximum depth of the tree (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )
            min_samples_split = st.number_input(
                "Select the minimum number of samples required to split an internal node (min_value = 2, max_value = 200, default = 2)",
                min_value=2,
                max_value=200,
                value=2,
                step=1,
            )
            min_samples_leaf = st.number_input(
                "Select the minimum number of samples required to be at a leaf node (min_value = 1, max_value = 200, default = 1)",
                min_value=1,
                max_value=200,
                value=1,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Random Forest Classifier Model"):
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Random Forest Classifier"

                    st.success("Random Forest Classifier Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Random Forest Classifier"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="Random Forest Classifier"
                    )

        elif selected_model == "Support Vector Machine (SVM)":
            st.markdown("#### Support Vector Machine (SVM)")
            kernel_options = ["linear", "poly", "rbf", "sigmoid"]
            kernel = st.selectbox(
                "Select the kernel type to be used in the algorithm (default = rbf)",
                kernel_options,
                index=kernel_options.index("rbf"),
            )
            C = st.number_input(
                "Select the regularization parameter C (min_value = 0.1, max_value = 10.0, default = 1.0)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            degree = st.number_input(
                "Select the degree of the polynomial kernel function (min_value = 1, max_value = 10, default = 3)",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )
            gamma_options = ["scale", "auto"]
            gamma = st.selectbox(
                "Select the kernel coefficient (default = scale)",
                gamma_options,
                index=gamma_options.index("scale"),
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train SVM Model"):
                    model = SVC(
                        kernel=kernel,
                        C=C,
                        degree=degree,
                        gamma=gamma,
                        probability=True,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "SVM"

                    st.success("SVM Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "SVM"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="SVM"
                    )

        elif selected_model == "K-Nearest Neighbors (KNN)":
            st.markdown("#### K-Nearest Neighbors (KNN)")
            n_neighbors = st.number_input(
                "Select the number of neighbors (min_value = 1, max_value = 100, default = 5)",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
            )
            weights_options = ["uniform", "distance"]
            weights = st.selectbox(
                "Select the weight function to be used in prediction (default = uniform)",
                weights_options,
                index=weights_options.index("uniform"),
            )
            algorithm_options = ["auto", "ball_tree", "kd_tree", "brute"]
            algorithm = st.selectbox(
                "Select the algorithm to compute nearest neighbors (default = auto)",
                algorithm_options,
                index=algorithm_options.index("auto"),
            )
            p = st.number_input(
                "Select power parameter for the Minkowski metric (min_value = 1, max_value = 10, default = 2)",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train KNN Model"):
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm,
                        p=p,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "KNN"

                    st.success("KNN Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "KNN"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="KNN"
                    )

        elif selected_model == "Gradient Boosting Classifier":
            st.markdown("#### Gradient Boosting Classifier")
            n_estimators = st.number_input(
                "Select the number of boosting stages (min_value = 10, max_value = 500, default = 100)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
            )
            learning_rate = st.number_input(
                "Select the learning rate (min_value = 0.01, max_value = 1.0, default = 0.1)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )
            max_depth = st.number_input(
                "Select maximum depth of the individual regression estimators (min_value = 1, max_value = 20, default = 3)",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Gradient Boosting Classifier Model"):
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "Gradient Boosting Classifier"

                    st.success("Gradient Boosting Classifier Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "Gradient Boosting Classifier"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="Gradient Boosting Classifier"
                    )

        elif selected_model == "XGBoost Classifier" and XGBOOST_AVAILABLE:
            st.markdown("#### XGBoost Classifier")
            n_estimators = st.number_input(
                "Select the number of boosting rounds (min_value = 10, max_value = 500, default = 100)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
            )
            learning_rate = st.number_input(
                "Select the learning rate (min_value = 0.01, max_value = 1.0, default = 0.1)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )
            max_depth = st.number_input(
                "Select maximum depth of a tree (min_value = 1, max_value = 20, default = 6)",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train XGBoost Classifier Model"):
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "XGBoost Classifier"

                    st.success("XGBoost Classifier Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "XGBoost Classifier"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="XGBoost Classifier"
                    )

        elif selected_model == "LightGBM Classifier" and LIGHTGBM_AVAILABLE:
            st.markdown("#### LightGBM Classifier")
            n_estimators = st.number_input(
                "Select the number of boosting iterations (min_value = 10, max_value = 500, default = 100)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
            )
            learning_rate = st.number_input(
                "Select the learning rate (min_value = 0.01, max_value = 1.0, default = 0.1)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )
            max_depth = st.number_input(
                "Select maximum depth of a tree (min_value = -1, max_value = 20, default = -1)",
                min_value=-1,
                max_value=20,
                value=-1,
                step=1,
            )
            num_leaves = st.number_input(
                "Select number of leaves (min_value = 2, max_value = 256, default = 31)",
                min_value=2,
                max_value=256,
                value=31,
                step=1,
            )

            metrics = st.multiselect(
                "Select the metrics to evaluate the model",
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Confusion Matrix",
                    "ROC Curve",
                    "Loss",
                ],
                default=["Accuracy"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train LightGBM Classifier Model"):
                    model = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        num_leaves=num_leaves,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = "LightGBM Classifier"

                    st.success("LightGBM Classifier Model Trained!")

                    # Calculate loss
                    if "Loss" in metrics:
                        calculate_and_display_loss(model, X_test, y_test)

                    # Calculate and Display Metrics
                    st.markdown("##### Performance Metrics:")
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                    if "Precision" in metrics:
                        precision = precision_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Precision: {precision:.4f}")
                    if "Recall" in metrics:
                        recall = recall_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"Recall: {recall:.4f}")
                    if "F1 Score" in metrics:
                        f1 = f1_score(
                            y_test, y_pred, average="weighted", zero_division=0
                        )
                        st.write(f"F1 Score: {f1:.4f}")
                    if "Confusion Matrix" in metrics:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues", ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                    if "ROC Curve" in metrics:
                        try:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                auc_score = roc_auc_score(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.plot(
                                    fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})"
                                )
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic (ROC)")
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                            else:
                                st.info(
                                    "ROC Curve is only available for binary classification."
                                )
                        except Exception as e:
                            st.error(f"Error generating ROC curve: {e}")

            with col2:
                if (
                    st.session_state.trained_model is not None
                    and st.session_state.model_type == "LightGBM Classifier"
                ):
                    save_and_download_model(
                        st.session_state.trained_model, model_name="LightGBM Classifier"
                    )
