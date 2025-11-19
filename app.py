# Imports 

import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
from streamlit_option_menu import option_menu

# Helper functions
from scripts.model_methods import load_data, vis_heatmap, train_and_evaluate_model, predict_cpu_load

# Session State Initialization

def on_button_click():
    """Callback function to toggle button state"""
    st.session_state.button_clicked = True

# Configuration Variables

dataset_path = "./data/cpu_load_data.csv"
st.set_page_config(page_title="SOC Analysis Tool", layout="wide", initial_sidebar_state="collapsed")

# Load data once at startup
data = load_data(dataset_path)


# Sidebar Menu

with st.sidebar:
    selected = option_menu(
        menu_title="SOC Tools",
        options=["Home", "Data Analysis", "Model Training", "Prediction"],
        default_index=0
    )

# Page: Home

if selected == "Home":
    st.title("SOC Analyst Helper")
    st.subheader("Features")
    st.write("""
    - **Data Analysis**: Explore the dataset with preview and correlation visualization
    - **Model Training**: Configure and train KNN models with cross-validation
    - **Prediction**: Predict CPU load for new network traffic observations
    """)

    st.subheader("Project Context")
    st.write("""
    This work is part of an academic project. The objective is to provide users with the ability to:
    - Train a KNN regression model
    - Test different values of k (number of neighbors)
    - Evaluate model performance using cross-validation
    - Make predictions on new observations
    """)

    st.info("Use the sidebar menu to navigate between different sections.")

# Page: Data Analysis

elif selected == "Data Analysis":
    st.title("Data Analysis")
    st.write("Explore the CPU load dataset and analyze feature correlations.")

    # Initialize button state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    # Section: Data Preview
    st.subheader("Data Preview")
    st.write("Click the button below to preview the first rows of the dataset.")
    st.button("Preview Data", on_click=on_button_click)

    if st.session_state.button_clicked:
        st.dataframe(data.head(10))
        st.write(f"**Dataset shape:** {data.shape[0]} rows × {data.shape[1]} columns")

    # Section: Correlation Heatmap
    st.subheader("Correlation Matrix")
    st.write("Visualize the correlations between features to understand relationships in the data.")

    if st.button("Visualize Heatmap"):
        with st.spinner("Generating heatmap..."):
            st.pyplot(vis_heatmap(data))
        st.info("💡 Strong correlations (close to 1 or -1) indicate features that are highly related.")

# Page: Model Training

elif selected == "Model Training":
    st.title("Model Configuration and Training")
    st.write("Configure the KNN model parameters and evaluate its performance using cross-validation.")

    # Model Configuration
    st.subheader("Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # K-value selection
        k_value = st.slider(
            "Number of Neighbors (k)",
            min_value=1,
            max_value=19,
            value=5,
            step=2,
            help="Choose an odd number to avoid ties in predictions"
        )
        st.write(f"**Selected k value:** {k_value}")

    with col2:
        # Cross-validation folds
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of folds for cross-validation"
        )
        st.write(f"**CV Folds:** {cv_folds}")

    # Feature Selection Section
    st.subheader("Feature Selection")

    # Get available features (exclude target 'cpu_load')
    features_available = [col for col in data.columns if col != 'cpu_load']

    num_available = len(features_available)
    max_n = min(8, num_available) if num_available >= 2 else 2
    default_n = min(4, max_n)

    n_features = st.slider(
        "Number of features to use",
        min_value=1,
        max_value=max_n,
        value=default_n,
        step=1,
        help="Select how many features to use for training"
    )

    selected_features = st.multiselect(
        f"Select {n_features} feature(s) from the available features:",
        options=features_available,
        default=features_available[:n_features],
        help="Choose which network indicators to use for predicting CPU load"
    )

    # Validation and Training
    if len(selected_features) != n_features:
        st.warning(f"Please select exactly {n_features} feature(s). Currently selected: {len(selected_features)}.")
    else:
        st.success(f"{n_features} feature(s) selected: {', '.join(selected_features)}")

        # Train model button
        if st.button("Train Model and Evaluate", type="primary"):
            with st.spinner("Training model and performing cross-validation..."):
                # Train and evaluate
                results = train_and_evaluate_model(data, selected_features, k_value, cv_folds)

                # Store results in session state
                st.session_state['model'] = results['model']
                st.session_state['selected_features'] = selected_features
                st.session_state['k_value'] = k_value

                # Display results
                st.subheader("Model Performance")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Mean R² Score",
                        value=f"{results['mean_r2']:.4f}",
                        help="Average R² score across all folds (higher is better)"
                    )

                with col2:
                    st.metric(
                        label="Standard Deviation",
                        value=f"{results['std_r2']:.4f}",
                        help="Variability of R² scores across folds (lower is better)"
                    )

                with col3:
                    st.metric(
                        label="Number of Folds",
                        value=f"{cv_folds}"
                    )

                # Display individual fold scores
                st.subheader("Cross-Validation Scores by Fold")
                scores_df = pd.DataFrame({
                    'Fold': [f"Fold {i+1}" for i in range(len(results['scores']))],
                    'R² Score': results['scores']
                })
                st.dataframe(scores_df, use_container_width=True)

                # Interpretation
                st.info(f"""
                **Model Interpretation:**
                - The model achieved an average R² score of **{results['mean_r2']:.4f}**
                - R² score ranges from -∞ to 1.0, where 1.0 is perfect prediction
                - A score above 0.7 is generally considered good
                - Standard deviation of {results['std_r2']:.4f} indicates {'low' if results['std_r2'] < 0.1 else 'moderate' if results['std_r2'] < 0.2 else 'high'} variability across folds
                """)

                st.success("Model trained successfully! You can now use it for predictions in the Prediction tab.")

# Page: Prediction

elif selected == "Prediction":
    st.title("CPU Load Prediction")
    st.write("Predict CPU load for a new network traffic observation.")

    # Check if model is trained
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section before making predictions.")
        st.info("Go to **Model Training** to configure and train your model.")
    else:
        st.success(f"Model loaded with k={st.session_state.get('k_value', 'N/A')} and features: {', '.join(st.session_state['selected_features'])}")

        st.subheader("Enter Network Traffic Indicators")
        st.write("Provide values for the selected features to predict the corresponding CPU load.")

        # Create input fields based on selected features
        feature_values = {}

        # Define feature descriptions and ranges
        feature_info = {
            'tcp_connections': {
                'label': 'TCP Connections',
                'help': 'Number of active TCP connections',
                'min': 0.0,
                'max': 100.0,
                'default': 50.0
            },
            'data_volume_MB': {
                'label': 'Data Volume (MB)',
                'help': 'Volume of data transferred in megabytes',
                'min': 0.0,
                'max': 1000.0,
                'default': 100.0
            },
            'packet_loss_rate': {
                'label': 'Packet Loss Rate (%)',
                'help': 'Percentage of packets lost during transmission',
                'min': 0.0,
                'max': 100.0,
                'default': 5.0
            },
            'ids_alerts': {
                'label': 'IDS Alerts',
                'help': 'Number of Intrusion Detection System alerts',
                'min': 0.0,
                'max': 100.0,
                'default': 10.0
            }
        }

        # Create input fields in columns
        cols = st.columns(2)

        for idx, feature in enumerate(st.session_state['selected_features']):
            with cols[idx % 2]:
                info = feature_info.get(feature, {
                    'label': feature,
                    'help': f'Value for {feature}',
                    'min': 0.0,
                    'max': 100.0,
                    'default': 50.0
                })

                feature_values[feature] = st.number_input(
                    label=info['label'],
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=1.0,
                    help=info['help']
                )

        # Prediction button
        if st.button("Predict CPU Load", type="primary"):
            with st.spinner("Making prediction..."):
                # Make prediction
                prediction = predict_cpu_load(
                    st.session_state['model'],
                    feature_values,
                    st.session_state['selected_features']
                )

                # Display prediction
                st.subheader("Prediction Result")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(
                        label="Predicted CPU Load",
                        value=f"{prediction:.2f}",
                        help="Predicted CPU load value (standardized)"
                    )

                with col2:
                    # Visual indicator
                    st.write("**CPU Load Level:**")
                    if prediction < -1:
                        st.success("Very Low")
                    elif prediction < 0:
                        st.info("Low")
                    elif prediction < 1:
                        st.warning("Moderate")
                    else:
                        st.error("High")

                # Display input values
                st.subheader("Input Values Used")
                input_df = pd.DataFrame([feature_values])
                st.dataframe(input_df, use_container_width=True)

                st.info("""
                **Note:** The values are standardized (normalized).
                - Negative values indicate below average
                - Positive values indicate above average
                - The scale is relative to the training data
                """)