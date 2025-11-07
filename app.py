import streamlit as st
import pandas as pd
import cloudpickle
import shap
import matplotlib.pyplot as plt

# Load pipeline using cloudpickle
try:
    with open("mortality_model.pkl", "rb") as f:
        model = cloudpickle.load(f)

    # Confirm it's a pipeline
    if not hasattr(model, "named_steps"):
        st.error("❌ Loaded object is not a scikit-learn pipeline.")
        st.stop()

    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

except Exception as e:
    st.error(f"❌ Failed to load model pipeline: {e}")
    st.stop()

st.title("🩺 10-Year Mortality Prediction")

# Inputs
age = st.number_input("Age at Diagnosis", 20, 90)
tumor_size = st.number_input("Tumor Size (mm)", 0, 100)
lymph_nodes = st.number_input("Lymph nodes examined positive", 0, 20)
subtype = st.selectbox("Pam50 + Claudin-low subtype", ["Luminal A", "Luminal B", "Her2", "Basal", "Normal"])
er_status = st.selectbox("ER Status", ["Positive", "Negative"])
pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
chemo = st.selectbox("Chemotherapy", ["Yes", "No"])
hormone = st.selectbox("Hormone Therapy", ["Yes", "No"])
radio = st.selectbox("Radio Therapy", ["Yes", "No"])
surgery = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"])
grade = st.selectbox("Neoplasm Histologic Grade", ["Grade 1", "Grade 2", "Grade 3"])

# Optional feature
if "Inferred Menopausal State" in preprocessor.feature_names_in_:
    menopausal_state = st.selectbox("Inferred Menopausal State", ["Pre", "Post"])
else:
    menopausal_state = None

# Assemble input
input_data = pd.DataFrame([{
    "Age at Diagnosis": age,
    "Tumor Size": tumor_size,
    "Lymph nodes examined positive": lymph_nodes,
    "Pam50 + Claudin-low subtype": subtype,
    "ER Status": er_status,
    "PR Status": pr_status,
    "HER2 Status": her2_status,
    "Chemotherapy": chemo,
    "Hormone Therapy": hormone,
    "Radio Therapy": radio,
    "Type of Breast Surgery": surgery,
    "Neoplasm Histologic Grade": grade
}])

if menopausal_state is not None:
    input_data["Inferred Menopausal State"] = menopausal_state

# Prediction and SHAP
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]

        if hasattr(classifier, "predict_proba") and len(classifier.classes_) == 2:
            proba = model.predict_proba(input_data)[0][1]
            st.success(f"Predicted 10-Year Mortality: {'Yes' if prediction == 1 else 'No'} ({proba:.2f} probability)")
        else:
            st.success(f"Predicted 10-Year Mortality: {'Yes' if prediction == 1 else 'No'}")
            st.warning("Probability score unavailable — model may have been trained on a single class.")

        st.subheader("🔍 Feature Impact (SHAP)")
        X_transformed = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        class_index = 1 if len(shap_values) > 1 else 0

        shap_exp = shap.Explanation(
            values=shap_values[class_index][0],
            base_values=explainer.expected_value[class_index],
            data=X_transformed[0],
            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
