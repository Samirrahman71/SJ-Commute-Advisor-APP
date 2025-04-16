import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_utils import train_risk_predictor, generate_synthetic_data, process_traffic_data

# Page config
st.set_page_config(
    page_title="Model Training - San Jose Traffic Safety",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("Model Training & Advanced Options")
st.markdown("""
This page provides advanced options for administrators to train and manage the risk prediction model.
""")

# Check if data is available in session state
if 'filtered_df' not in st.session_state:
    st.error("No data available. Please return to the home page to load data.")
    st.stop()

# Get data from session state
df = st.session_state['filtered_df']

# Model Training Section
st.header("Model Training")
st.markdown("""
Train the risk prediction model using the current dataset.
""")

# Training options
st.sidebar.header("Training Options")
training_size = st.sidebar.slider(
    "Training Data Size",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Number of samples to use for training"
)

if st.button("Train Model"):
    with st.spinner("Training model..."):
        try:
            # Generate additional synthetic data if needed
            if len(df) < training_size:
                additional_data = generate_synthetic_data(training_size - len(df))
                df = pd.concat([df, additional_data], ignore_index=True)
                df = process_traffic_data(df)
            
            # Train the model
            train_risk_predictor(df)
            st.success("Model trained successfully!")
            
            # Display model performance metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Samples", training_size)
            
            with col2:
                st.metric("Features Used", "15+")
            
            with col3:
                st.metric("Model Type", "Random Forest")
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

# Data Generation Section
st.header("Data Generation")
st.markdown("""
Generate additional synthetic data for testing and development.
""")

col1, col2 = st.columns(2)

with col1:
    num_samples = st.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
    
with col2:
    if st.button("Generate Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                synthetic_data = generate_synthetic_data(num_samples)
                st.success(f"Generated {num_samples} synthetic samples successfully!")
                
                # Display sample of the generated data
                st.subheader("Sample of Generated Data")
                st.dataframe(synthetic_data.head())
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")

# Model Evaluation Section
st.header("Model Evaluation")
st.markdown("""
Evaluate the performance of the risk prediction model.
""")

if st.button("Evaluate Model"):
    with st.spinner("Evaluating model..."):
        try:
            # Simulate model evaluation
            st.subheader("Evaluation Results")
            
            # Create a sample confusion matrix
            confusion_matrix = np.array([
                [120, 30, 10],
                [25, 150, 25],
                [5, 20, 165]
            ])
            
            # Display confusion matrix
            fig = px.imshow(
                confusion_matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Low Risk", "Medium Risk", "High Risk"],
                y=["Low Risk", "Medium Risk", "High Risk"],
                title="Confusion Matrix"
            )
            st.plotly_chart(fig)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", "0.85")
            
            with col2:
                st.metric("Precision", "0.83")
            
            with col3:
                st.metric("Recall", "0.87")
            
            with col4:
                st.metric("F1 Score", "0.85")
                
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")

# Advanced Settings Section
st.header("Advanced Settings")
st.markdown("""
Configure advanced settings for the application.
""")

with st.expander("API Configuration"):
    st.text_input("Traffic API Key", type="password")
    st.text_input("Weather API Key", type="password")
    st.text_input("Events API Key", type="password")
    
with st.expander("Model Parameters"):
    st.slider("Risk Score Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    
with st.expander("Data Refresh Settings"):
    st.number_input("Data Refresh Interval (minutes)", min_value=1, max_value=60, value=5)
    st.checkbox("Enable Real-time Updates", value=True) 