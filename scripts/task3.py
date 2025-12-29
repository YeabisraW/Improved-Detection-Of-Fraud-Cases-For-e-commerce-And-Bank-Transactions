# ---------------------------------------------
# Task 3: Model Explainability with SHAP (Fully Updated)
# ---------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------------------------
# Config
# -------------------------------
DATASET_NAME = "Credit Card Dataset"   # "E-commerce Dataset" or "Credit Card Dataset"
MODEL_PATH = "models/best_model_credit_card.joblib"  # Update if e-commerce
X_TEST_PATH = "data/processed/X_test_credit.csv"
Y_TEST_PATH = "data/processed/y_test_credit.csv"
SAMPLE_SIZE = 1000   # Limit SHAP computation for speed

# -------------------------------
# Load model & data
# -------------------------------
print("Loading model and data...")
model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

# Sample data for SHAP (speed)
if len(X_test) > SAMPLE_SIZE:
    X_sample = X_test.sample(SAMPLE_SIZE, random_state=42)
else:
    X_sample = X_test.copy()
y_sample = y_test.loc[X_sample.index]

# -------------------------------
# Built-in Feature Importance
# -------------------------------
print("Plotting built-in feature importance...")
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
elif hasattr(model, "coef_"):
    importances = np.abs(model.coef_[0])
else:
    raise ValueError("Model does not support feature importance.")

# Align feature importance with columns
if len(importances) != X_test.shape[1]:
    print("Warning: feature importance length mismatch, using sample columns.")
    feature_importance = pd.Series(importances, index=X_sample.columns[:len(importances)])
else:
    feature_importance = pd.Series(importances, index=X_test.columns)

top10 = feature_importance.sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
top10.sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importance (Built-in)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -------------------------------
# SHAP Explainer (Tree-based)
# -------------------------------
print("Running SHAP analysis (optimized)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification: take positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# -------------------------------
# SHAP Summary Plot
# -------------------------------
shap.summary_plot(shap_values, X_sample, show=True)

# -------------------------------
# Identify TP, FP, FN samples from sample
# -------------------------------
y_pred_sample = model.predict(X_sample)

tp_idx = np.where((y_sample == 1) & (y_pred_sample == 1))[0]
fp_idx = np.where((y_sample == 0) & (y_pred_sample == 1))[0]
fn_idx = np.where((y_sample == 1) & (y_pred_sample == 0))[0]

# -------------------------------
# SHAP Force Plot Helper
# -------------------------------
def plot_force(idx, label):
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray) and base_value.size > 1:
        base_value = base_value[1]  # positive class

    shap.plots.force(
        shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=X_sample.iloc[idx]
        )
    )
    plt.title(label)
    plt.show()

# -------------------------------
# SHAP Force Plots
# -------------------------------
if len(tp_idx) > 0:
    plot_force(tp_idx[0], "True Positive (Correct Fraud Detection)")
if len(fp_idx) > 0:
    plot_force(fp_idx[0], "False Positive (Legitimate Flagged)")
if len(fn_idx) > 0:
    plot_force(fn_idx[0], "False Negative (Missed Fraud)")

# -------------------------------
# Top 5 SHAP Feature Drivers
# -------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=X_sample.columns)
top5_shap = shap_importance.sort_values(ascending=False).head(5)

print("\nTop 5 SHAP Feature Drivers:")
print(top5_shap)

print("\n✅ Task 3 SHAP analysis completed successfully.")
