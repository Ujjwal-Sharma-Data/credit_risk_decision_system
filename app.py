import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    model = joblib.load("/Users/ujjwalsharma/Desktop/credit_risk_decision_system/model/lgb_model.pkl")
    preprocessor = joblib.load("/Users/ujjwalsharma/Desktop/credit_risk_decision_system/model/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("📊 Credit Risk Decision Dashboard")

st.markdown("""
This dashboard uses a **machine learning credit risk model** to:
- Predict Probability of Default (PD)
- Estimate Expected Loss
- Simulate approval thresholds
- Explain decisions using SHAP
""")

uploaded_file = st.file_uploader(
    "Upload customer data (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

data = pd.read_csv(uploaded_file)

X = preprocessor.transform(data)
data["PD"] = model.predict_proba(X)[:, 1]

LGD = 0.45  # 45% loss given default

data["EXPECTED_LOSS"] = data["PD"] * data["AMT_CREDIT"] * LGD

threshold = st.slider(
    "Select Risk Threshold (PD)",
    min_value=0.05,
    max_value=0.90,
    value=0.30,
    step=0.01
)

data["DECISION"] = np.where(
    data["PD"] < threshold,
    "APPROVE",
    "REJECT"
)

approved = data[data["DECISION"] == "APPROVE"]
rejected = data[data["DECISION"] == "REJECT"]

baseline_loss = data["EXPECTED_LOSS"].sum()
model_loss = approved["EXPECTED_LOSS"].sum()
money_saved = baseline_loss - model_loss

st.subheader("📌 Portfolio Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Applications", len(data))
col2.metric("Approval Rate", f"{len(approved)/len(data):.2%}")
col3.metric("Expected Loss (₹)", f"{model_loss:,.0f}")
col4.metric("Money Saved (₹)", f"{money_saved:,.0f}")

@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(model)

explainer = get_shap_explainer()

st.subheader("🔍 Individual Customer Explanation")

idx = st.number_input(
    "Select customer index",
    min_value=0,
    max_value=len(data)-1,
    value=0
)

shap_values = explainer.shap_values(X[idx:idx+1])

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
    base_value = explainer.expected_value[1]
else:
    shap_vals = shap_values
    base_value = explainer.expected_value

fig, ax = plt.subplots(figsize=(8, 5))

shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0],
        base_values=base_value,
        data=X[idx],
        feature_names=preprocessor.get_feature_names_out()
    ),
    show=False
)

st.pyplot(fig)

