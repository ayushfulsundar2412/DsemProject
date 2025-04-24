import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset (replace with actual temporal_data.csv)
df = pd.read_csv("temporal_data.csv")

years = range(2000, 2007)
features = ["MAX_SYSTOLIC", "MAX_DIASTOLIC", "MAX_BMI", "MAX_VALUE_Chol"]
columns = [f"{f}_{y}" for y in years for f in features]

X = df[columns]
y = df["Class"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# SHAP explanation (CPU-friendly)
explainer = shap.Explainer(model.predict, X_scaled)
shap_values = explainer(X_scaled)

# Visualize SHAP summary
shap.summary_plot(shap_values, X_scaled, feature_names=columns)
