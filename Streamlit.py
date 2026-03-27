import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI OEE Predictor", layout="wide")

st.title("🏭 AI OEE Prediction Dashboard")
st.markdown("Predict machine efficiency and identify performance drivers")

# -----------------------------
# CREATE DATA (TRAIN MODEL ONCE)
# -----------------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "machine_speed": np.random.normal(1000, 100, n),
        "downtime": np.random.normal(60, 20, n),
        "defect_rate": np.random.normal(5, 2, n),
        "operator_efficiency": np.random.normal(85, 5, n),
        "temperature": np.random.normal(75, 10, n)
    })

    availability = 1 - (data["downtime"] / 480)
    performance = data["machine_speed"] / 1200
    quality = 1 - (data["defect_rate"] / 100)

    data["OEE"] = availability * performance * quality * data["operator_efficiency"]/100

    X = data.drop("OEE", axis=1)
    y = data["OEE"]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    return model, X.columns

model, features = train_model()

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("⚙️ Input Machine Parameters")

machine_speed = st.sidebar.slider("Machine Speed (RPM)", 800, 1200, 1000)
downtime = st.sidebar.slider("Downtime (minutes)", 0, 120, 60)
defect_rate = st.sidebar.slider("Defect Rate (%)", 0.0, 10.0, 5.0)
operator_eff = st.sidebar.slider("Operator Efficiency (%)", 70, 100, 85)
temperature = st.sidebar.slider("Temperature (°C)", 50, 100, 75)

input_data = pd.DataFrame([[machine_speed, downtime, defect_rate, operator_eff, temperature]],
                          columns=features)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(input_data)[0]

# -----------------------------
# DISPLAY KPI
# -----------------------------
st.subheader("📊 Predicted OEE")

st.metric("OEE (%)", round(prediction*100,2))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📈 Key Drivers of OEE")

importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)

# -----------------------------
# INSIGHT
# -----------------------------
st.subheader("🧠 AI Insight")

if prediction < 0.7:
    st.error("OEE is low. Reduce downtime and defects.")
elif prediction < 0.85:
    st.warning("Moderate OEE. Optimization possible.")
else:
    st.success("High OEE. System performing efficiently.")
    