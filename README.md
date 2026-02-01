### Live Demo :https://creditriskdecisionsystem-5rkjp5cdxdfdppnehywrgk.streamlit.app/
# Credit Risk Decision System 🚀

An **end-to-end, production-style credit risk modeling and decision support system** built using **LightGBM**, deployed as an interactive **Streamlit dashboard** with **business KPIs** and **explainable AI (SHAP)**.

This project goes beyond model training and focuses on **real-world decision making**, **expected loss reduction**, and **explainability**, closely mimicking how credit risk systems operate in industry.

---

## 🔍 Problem Statement

Financial institutions must decide **which loan applicants to approve** while minimizing **credit losses**.

The objective of this project is to:
- Predict **Probability of Default (PD)** for loan applicants
- Support **risk-based approval decisions**
- Estimate **Expected Loss (EL)**
- Quantify **money saved** using a model-driven strategy
- Explain decisions transparently using **SHAP**

---

## 📊 Data Sources

The model integrates multiple relational datasets:

- **Application data** – demographics, income, loan details
- **Bureau data** – external credit history
- **Previous applications** – past loan decisions and rejections
- **Installment payments** – actual repayment behavior

These datasets are aggregated at the **customer level**, replicating real-world feature engineering practices.

---

## 🧠 Modeling Approach

- **Model**: LightGBM (Gradient Boosting Decision Trees)
- **Metric**: ROC–AUC  
- **Final Validation AUC**: **~0.775**
- **Class imbalance handling**: `scale_pos_weight`
- **Feature engineering**:
  - Bureau behavior ratios
  - Approval/refusal history
  - Installment delay and underpayment patterns
- **Explainability**: SHAP (global + local explanations)

---

## 💼 Business Logic & KPIs

### 🔹 Expected Loss (EL)

Expected Loss is calculated using the standard credit risk formula:
**Expected Loss = PD × Exposure (AMT_CREDIT) × LGD**


- **LGD (Loss Given Default)** is assumed to be **45%** (industry-standard conservative assumption).

---

### 🔹 Decision Strategy

Applicants are:
- **Approved** if `PD < threshold`
- **Rejected** otherwise

The dashboard allows **real-time threshold tuning** to simulate policy changes.

---

### 🔹 Key Business Metrics Shown

- Approval Rate
- Portfolio Expected Loss
- Baseline Loss (approve-all strategy)
- **Money Saved using the model**
- Individual customer risk explanation

---

## 📈 Dashboard Features (Streamlit)

### 1️⃣ Portfolio Overview
- Total applications
- Approval rate
- Expected loss
- Money saved vs baseline

### 2️⃣ Risk Threshold Simulator
- Adjustable PD cutoff
- Live update of approvals & losses
- Enables policy trade-off analysis

### 3️⃣ Individual Customer View
- Predicted PD
- Approve / Reject decision
- **SHAP waterfall plot explaining the decision**

### 4️⃣ Explainability (SHAP)
- Global feature importance
- Local (per-customer) explanations
- Feature interaction insights

---

## 🛠 Tech Stack

- **Python** 3.11
- **scikit-learn** 1.6.1
- **LightGBM**
- **SHAP**
- **Streamlit**
- pandas, numpy, matplotlib, joblib

---

## 🚀 How to Run Locally

```bash
# create virtual environment (recommended)
python3.11 -m venv myenv
source myenv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

---

## 🌐 Deployment

- The app is deployed on Streamlit Community Cloud.
- Note: For demonstration purposes, the app accepts small sample CSV files (≤1000 rows), reflecting how real credit systems score individual applicants or small batches.

---

## 📌 Key Insights

- External credit scores (EXT_SOURCE_*) are the strongest predictors

- Behavioral features (installment delays, past rejections) add major incremental value

- Extreme late payments are more predictive than average delays

- The model reduced expected loss by ~80% on a sample portfolio

- LightGBM + SHAP provides both performance and explainability

---

## 👤 Author

Ujjwal Sharma

This project was built to demonstrate:

- End-to-end applied data science

- Business-centric ML thinking

- Deployment & explainability skills

- Real-world credit risk modeling

---

## 📎 License

This project is for educational and portfolio purposes only.

