import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

np.random.seed(42)

# -----------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -----------------------------

n = 1000

data = pd.DataFrame({
    "machine_speed": np.random.normal(1000, 100, n),
    "downtime": np.random.normal(60, 20, n),
    "defect_rate": np.random.normal(5, 2, n),
    "operator_efficiency": np.random.normal(85, 5, n),
    "temperature": np.random.normal(75, 10, n)
})

# -----------------------------
# STEP 2: CREATE OEE (TARGET)
# -----------------------------

availability = 1 - (data["downtime"] / 480)
performance = data["machine_speed"] / 1200
quality = 1 - (data["defect_rate"] / 100)

data["OEE"] = availability * performance * quality * data["operator_efficiency"]/100

# -----------------------------
# STEP 3: TRAIN TEST SPLIT
# -----------------------------

X = data.drop("OEE", axis=1)
y = data["OEE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# STEP 4: MODEL (XGBOOST)
# -----------------------------

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: PREDICTION
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# STEP 6: EVALUATION
# -----------------------------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

# -----------------------------
# STEP 7: FEATURE IMPORTANCE
# -----------------------------

importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.title("Feature Importance for OEE")
plt.show()