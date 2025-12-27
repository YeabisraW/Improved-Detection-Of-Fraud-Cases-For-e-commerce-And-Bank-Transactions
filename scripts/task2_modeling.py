# ---------------------------------------------
# Task 2: Model Building, Training, and Logging
# ---------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")
plt.switch_backend('agg')  # Non-interactive backend

# -------------------------------
# Helper Functions
# -------------------------------

def pr_auc_score(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def cross_val_f1(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
    return f1_scores

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = y_pred
    f1 = f1_score(y_test, y_pred)
    pr_auc = pr_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return f1, pr_auc, cm, report

def plot_shap_feature_importance(model, X, dataset_name):
    print("\nGenerating SHAP feature importance...")
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X, feature_perturbation="correlation_dependent")
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance: {dataset_name}")
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/shap_feature_importance_{dataset_name.replace(' ', '_').lower()}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP plot for {dataset_name}")

def train_and_select_best(X_train, X_test, y_train, y_test, dataset_name="dataset", log_metrics=True):
    print(f"\n========== {dataset_name} ==========\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    best_model = None
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        f1, pr_auc, cm, report = evaluate_model(model, X_test, y_test)
        cv_f1_scores = cross_val_f1(model, X_train, y_train, cv=5)

        print(f"\n--- {dataset_name} Model Evaluation ---")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)
        print(f"Cross-Validation F1 Scores: {cv_f1_scores}")
        print(f"Mean CV F1: {np.mean(cv_f1_scores):.4f}, Std: {np.std(cv_f1_scores):.4f}")

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "F1_Test": f1,
            "PR-AUC": pr_auc,
            "CV_F1_Mean": np.mean(cv_f1_scores),
            "CV_F1_Std": np.std(cv_f1_scores)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Save best model
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{dataset_name.replace(' ', '_').lower()}.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best Model: {best_model_name}")
    print(f"Saved best model to {model_path}")

    # SHAP plot
    plot_shap_feature_importance(best_model, X_train, dataset_name)

    # Save metrics to CSV
    if log_metrics:
        metrics_dir = "../metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"model_metrics_{dataset_name.replace(' ', '_').lower()}.csv")
        pd.DataFrame(results).to_csv(metrics_path, index=False)
        print(f"Saved model metrics to {metrics_path}")

    return best_model_name, best_model

# -------------------------------
# Load preprocessed datasets
# -------------------------------
X_train_f = pd.read_csv("data/processed/X_train_fraud.csv")
X_test_f = pd.read_csv("data/processed/X_test_fraud.csv")
y_train_f = pd.read_csv("data/processed/y_train_fraud.csv").squeeze()
y_test_f = pd.read_csv("data/processed/y_test_fraud.csv").squeeze()

X_train_c = pd.read_csv("data/processed/X_train_credit.csv")
X_test_c = pd.read_csv("data/processed/X_test_credit.csv")
y_train_c = pd.read_csv("data/processed/y_train_credit.csv").squeeze()
y_test_c = pd.read_csv("data/processed/y_test_credit.csv").squeeze()

# -------------------------------
# Train and evaluate models
# -------------------------------
train_and_select_best(X_train_f, X_test_f, y_train_f, y_test_f, dataset_name="E-commerce Dataset")
train_and_select_best(X_train_c, X_test_c, y_train_c, y_test_c, dataset_name="Credit Card Dataset")
