# 📞 O2R Retailer Order Prediction & Call Centre Optimization

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172434?style=for-the-badge&logo=xgboost&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

This repository contains the source code for the **O2R Retailer Order Prediction System**, developed as part of a Data Science Internship (2026). The project leverages Machine Learning to optimize outbound call centre operations by predicting which retailers are highly probable to place an order on any given day.

## 🎯 Business Problem & Solution
**The Problem:** Outbound sales representatives traditionally call every retailer in the database. This approach leads to massive operational costs, wasted agent time, and call fatigue for retailers who are not ready to order.

**The Solution:** We trained an **XGBoost Classifier** on historical order timelines, retailer profiles, and temporal features to output a daily "Probability to Order." This model is served via a responsive **Streamlit Dashboard** that provides call centre managers with a ranked "Call Priority List," drastically reducing call volume while maximizing captured orders.

---

## ✨ Key Features

- 📊 **Business ROI Tracking:** Automatically calculates daily cost savings based on call reduction metrics.
- 📋 **Call Priority List:** A ranked, exportable CSV table of retailers to call, complete with a visual probability progress bar.
- 📅 **Next Order Scheduling:** Predicts the exact future date a retailer is likely to place their next order based on their historical gap averages.
- 🔍 **Retailer Deep Dive:** Search for specific retailers to view their historical order timeline.
- 🧠 **AI Explainability (SHAP):** Features interactive SHAP Waterfall plots that explain exactly *why* the model assigned a specific probability to a retailer in plain English.

---

## 🏗️ Technical Architecture

### 1. Data Pipeline & Modeling (`notebooks/`)
- **EDA & Feature Engineering:** Extracted temporal features (day of week, month start/end), historical ordering rhythm (average gap between orders), and recent velocity (orders in last 7/14/30 days).
- **Model Training:** Built and tuned an XGBoost classifier. Handled class imbalance using probability thresholds rather than strict binary classification to allow for business-driven operating points.

### 2. Frontend Interface (`app.py`)
- Built entirely in **Streamlit** with custom HTML/CSS injections for a premium, dark-mode-native aesthetic.
- Interactive visualizations powered by **Plotly Express** & **Plotly Graph Objects**.
- Live feature contribution analysis powered by **SHAP**.

---

## 🚀 Running the App Locally

> **Note regarding Data Privacy:** This working copy currently includes the data, processed parquet files, trained models, and generated output CSVs required to run the dashboard locally. If you publish the repository externally, review those folders carefully and remove or anonymize any NDA-restricted data first.

If those files are missing in another environment, place the corresponding data artifacts in their expected directories before launching the app.

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/O2R-Order-Prediction.git
cd O2R-Order-Prediction
```

**2. Install Dependencies**
```bash
pip install streamlit pandas numpy xgboost plotly shap matplotlib
```

**3. Run the Dashboard**
```bash
streamlit run app.py
```

---

## 🖼️ Dashboard Preview Highlights
* **Threshold Analysis:** Dynamically adjust the minimum probability threshold to see the exact tradeoff between "Orders Missed" and "Call Reduction."
* **Contextual Metric Cards:** Scorecards automatically color-code red or green depending on whether a retailer is overdue compared to their historical ordering rhythm.

---

