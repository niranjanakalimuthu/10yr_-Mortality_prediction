# 10-Year Mortality Prediction App

This Streamlit app predicts the 10-year mortality risk for breast cancer patients using a trained machine learning pipeline. It is designed for clinician-friendly use and includes SHAP-based interpretability to support transparent decision-making.

## Features

- Predicts 10-year mortality using clinical and pathological features
- Displays binary prediction and probability score (if available)
- SHAP plot for feature impact interpretation
- Handles missing features and class imbalance gracefully
- Deployable on Streamlit Cloud

##  Clinical Inputs

- Age at Diagnosis
- Tumor Size
- Lymph Nodes Examined Positive
- Pam50 + Claudin-low Subtype
- ER/PR/HER2 Status
- Chemotherapy, Hormone Therapy, Radio Therapy
- Type of Breast Surgery
- Neoplasm Histologic Grade
- Inferred Menopausal State

##  Model

- Algorithm: Random Forest Classifier
- Pipeline: Includes preprocessing and classifier steps
- Interpretability: SHAP TreeExplainer
- Trained on: Breast cancer cohort with survival outcomes

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
