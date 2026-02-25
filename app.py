import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Cell Tumor Diagnostic Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Setup (Load Pipeline) ---
@st.cache_resource
def load_setup():
    # Load the pipeline (contains both the StandardScaler and LogisticRegression)
    with open('model.pkl', 'rb') as file:
        pipeline = joblib.load(file)
    
    # Extract the exact feature names the pipeline expects from its internal memory
    feature_names = pipeline.feature_names_in_.tolist()
    
    return pipeline, feature_names

pipeline, feature_names = load_setup()

# --- Sidebar Navigation ---
st.sidebar.title("🔬 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Section:", ["Project Overview", "Clinical Data & Evaluation", "Live Diagnostic Engine"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Engineer:** Shivansh Sahu\n\n"
    "**Domain: Cancer biology & ML\n\n"
    "**Goal:** Bridging computational data with clinical diagnostics."
)

# --- PAGE 1: Project Overview ---
if page == "Project Overview":
    st.title("Cell Tumor Classification: ML Diagnostic Assistant")
    st.markdown("### Accelerating Triage in Clinical Pathology")
    
    st.write("""
    When a patient undergoes a Fine Needle Aspirate (FNA) biopsy for a suspected breast tumor, 
    pathologists must visually assess the extracted cell nuclei to determine if the mass is Benign or Malignant. 
    
    This machine learning project acts as a digital assistant. By analyzing the geometric and physical 
    measurements of cell nuclei (like radius, texture, and smoothness), this Logistic Regression algorithm 
    provides a highly accurate, objective diagnosis in milliseconds.
    """)
    
    st.info("**Technical Highlights:**\n"
            "* **Architecture:** Scikit-Learn Pipeline combining Preprocessing and Modeling.\n"
            "* **Feature Engineering:** Eliminated multicollinearity by dropping redundant geometric features (Area & Perimeter).\n"
            "* **Optimization:** Applied Z-Score Standardization to align microscopic physical measurements on a uniform mathematical scale.")

# --- PAGE 2: Clinical Data & Evaluation ---
elif page == "Clinical Data & Evaluation":
    st.title("📊 Data Analysis & Model Evaluation")
    st.write("A deep dive into the biological data distribution and the model's clinical viability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Class Distribution")
        st.write("Evaluating the dataset for class imbalances between Benign and Malignant samples.")
        if os.path.exists("images/01_class_distribution.png"):
            st.image("images/01_class_distribution.png", use_container_width=True)
        else:
            st.warning("Image '01_class_distribution.png' not found in images/ folder.")
            
        st.subheader("3. Clinical Confusion Matrix")
        st.write("Visualizing True/False Positives and Negatives on the hidden test set.")
        if os.path.exists("images/03_confusion_matrix.png"):
            st.image("images/03_confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Image '03_confusion_matrix.png' not found.")

    with col2:
        st.subheader("2. Multicollinearity Mapping")
        st.write("Identifying overlapping geometric features (Radius, Area, Perimeter) prior to feature reduction.")
        if os.path.exists("images/02_correlation_heatmap.png"):
            st.image("images/02_correlation_heatmap.png", use_container_width=True)
        else:
            st.warning("Image '02_correlation_heatmap.png' not found.")
            
        st.subheader("4. ROC-AUC Curve")
        st.write("Measuring the algorithm's ability to maximize Recall while minimizing false alarms.")
        if os.path.exists("images/04_roc_curve.png"):
            st.image("images/04_roc_curve.png", use_container_width=True)
        else:
            st.warning("Image '04_roc_curve.png' not found.")

# --- PAGE 3: Live Diagnostic Engine ---
elif page == "Live Diagnostic Engine":
    st.title("🩺 Live Diagnostic Predictor")
    st.write("Adjust the physical measurements of the cell nucleus below to see the model's real-time prediction.")
    
    # Create input fields grouped by category for better UI
    st.markdown("### Input Cellular Metrics")
    
    input_data = []
    
    mean_cols = [col for col in feature_names if 'mean' in col]
    se_cols = [col for col in feature_names if 'se' in col]
    worst_cols = [col for col in feature_names if 'worst' in col]
    
    col1, col2, col3 = st.columns(3)
    
    input_dict = {}
    
    with col1:
        st.markdown("**Mean Measurements**")
        for col in mean_cols:
            input_dict[col] = st.number_input(f"{col.replace('_mean', '').capitalize()}", value=15.0)
            
    with col2:
        st.markdown("**Standard Error (Variance)**")
        for col in se_cols:
            input_dict[col] = st.number_input(f"{col.replace('_se', '').capitalize()}", value=0.05)
            
    with col3:
        st.markdown("**Worst (Largest) Measurements**")
        for col in worst_cols:
            input_dict[col] = st.number_input(f"{col.replace('_worst', '').capitalize()}", value=20.0)

    # Reorder the dictionary to match the exact order the pipeline expects
    for col in feature_names:
        input_data.append(input_dict[col])
        
    st.markdown("---")
    
    if st.button("Run Diagnostic Scan", type="primary"):
        # 1. Convert input to a 2D numpy array with column names so the pipeline doesn't throw a warning
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # 2. Make the prediction (The pipeline handles the scaling automatically!)
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0]
        
        # 3. Display Results
        st.subheader("Diagnostic Results:")
        
        if prediction[0] == 1:
            st.error(f"🚨 **MALIGNANT** detected.")
            st.write(f"The algorithm calculates a **{probability[1]*100:.2f}%** probability that these cells are cancerous. Immediate clinical review is recommended.")
        else:
            st.success(f"✅ **BENIGN** detected.")
            st.write(f"The algorithm calculates a **{probability[0]*100:.2f}%** probability that these cells are healthy.")