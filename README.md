🔬 Oncology ML: Cell Tumor Classification & Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Deployment](https://img.shields.io/badge/Deployed%20on-Streamlit-red.svg)

> An interactive web application designed to predict and classify cell tumors using predictive machine learning algorithms.

<br>

## 📖 Project Overview
Early and accurate detection is critical in oncology. This project bridges the gap between raw medical data and actionable insights by utilizing a machine learning pipeline to classify cell tumors. 

Trained on a comprehensive genomic and clinical dataset from Kaggle, the predictive model analyzes multidimensional feature sets to determine tumor classifications. The underlying predictive engine is deployed through a clean, accessible Streamlit web interface, allowing users to input feature data and receive real-time diagnostic probabilities.

## ✨ Core Features
* **Interactive UI:** A modern, user-friendly dashboard built with Streamlit for seamless data input.
* **Predictive Inference:** Real-time tumor classification powered by trained classical machine learning models.
* **Confidence Metrics:** Outputs not just the binary classification, but the algorithm's statistical confidence probability.
* **Data-Driven Insights:** Capable of handling complex, non-linear biological data points efficiently.

## 🛠️ Tech Stack & Architecture

| Phase | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Data Sourcing** | Kaggle | High-quality dataset acquisition |
| **Data Wrangling** | Pandas, NumPy | Cleaning, scaling, and feature engineering |
| **Model Training** | Scikit-Learn | Training, cross-validation, and hyperparameter tuning |
| **Web Deployment** | Streamlit, Joblib | Model serialization and front-end interface hosting |

## 🧠 The Machine Learning Workflow

1. **Exploratory Data Analysis (EDA):** Visualizing feature distributions and correlations to identify the strongest predictors of tumor malignancy.
2. **Preprocessing:** Standardizing numerical inputs to ensure the mathematical models interpret the biological features equally without bias.
3. **Model Selection:** Training multiple algorithms to find the optimal balance of high accuracy and low false-negative rates (crucial for medical diagnostics).
4. **Serialization:** Saving the pipeline as a `.pkl` file for instant inference in the browser.

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/shivansh01234/Cancer-prediction-model.git](https://github.com/shivansh01234/Cancer-prediction-model.git)
