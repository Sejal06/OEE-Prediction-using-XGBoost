# OEE-Prediction-using-XGBoost
# 🏭 AI OEE Prediction Dashboard

## 🚀 Overview
This project predicts Overall Equipment Effectiveness (OEE) using machine-level operational data and provides actionable insights to improve manufacturing performance.

It transforms traditional reactive analysis into predictive decision intelligence.

---

## 🎯 Problem Statement
In manufacturing environments, OEE is typically analyzed after losses occur.

This leads to:
- Delayed decision-making
- Unplanned downtime
- Reduced productivity

---

## ✅ Solution
Built an AI-powered system that:
- Predicts OEE based on real-time machine parameters
- Identifies key factors affecting performance
- Enables proactive optimization

---

## ⚙️ Features
- Real-time OEE prediction using XGBoost
- Interactive dashboard using Streamlit
- Feature importance analysis
- Scenario-based simulation via input sliders

---

## 📊 Input Parameters
- Machine Speed (RPM)
- Downtime (minutes)
- Defect Rate (%)
- Operator Efficiency (%)
- Temperature (°C)

---

## 📈 Output
- Predicted OEE (%)
- Key drivers impacting OEE
- AI-based insights and recommendations

---

## 🧠 Model
- Algorithm: XGBoost Regressor
- Approach: Supervised learning on synthetic industrial dataset

---

## 💡 Key Insights
- Downtime has the highest impact on OEE
- Defect rate significantly reduces quality component
- Operator efficiency plays a stabilizing role

---

## 🛠️ Tech Stack
- Python
- XGBoost
- Pandas / NumPy
- Streamlit
- Matplotlib

---

## 🚀 How to Run

```bash
pip install streamlit xgboost pandas numpy matplotlib
streamlit run app.py
