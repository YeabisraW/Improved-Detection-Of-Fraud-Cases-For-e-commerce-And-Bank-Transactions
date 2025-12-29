# ---------------------------------------------
# Task 2: Model Building, Training, and Logging (Updated with Export)
# ---------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.filterwarnings("ignore")
plt.switch_backend('agg')

# -------------------------------
# Helper Functions
# -------------------------------

def pr_auc_score(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    f1 = f1_score(y_test, y_pred)
    pr_auc = pr_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return f1, pr_auc, cm, report

def train_and_select_best(X_raw, y_raw, dataset_name="dataset"):
    print(f"\n========== {dataset_name} ==========\n")

    # 1. EXPLICIT STRATIFIED SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # Save split data for Task 3 to use consistently
    os.makedirs("data/processed", exist_ok=True)
    X_test.to_csv(f"data/processed/X_test_{dataset_name.lower().replace(' ', '_')}.csv", index=False)
    y_test.to_csv(f"data/processed/y_test_{dataset_name.lower().replace(' ', '_')}.csv", index=False)

    # 2. BASELINE: LOGISTIC REGRESSION
    print("\nTraining Baseline: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    # 3. HYPERPARAMETER TUNING
    print("\nTuning Ensemble Model: Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # 4. MODEL COMPARISON
    # Including XGBoost as a baseline comparison
    xgb_model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    
    models = {
        "Logistic Regression": lr_model,
        "Tuned Random Forest": best_rf,
        "XGBoost Baseline": xgb_model
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        f1, pr_auc, cm, report = evaluate_model(model, X_test, y_test)
        results.append({
            "Model": name,
            "F1_Score": f1,
            "PR_AUC": pr_auc
        })

    # Save Comparison Table
    res_df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    res_df.to_csv(f"reports/model_comparison_{dataset_name.replace(' ', '_').lower()}.csv", index=False)
    
    # 5. SELECT BEST MODEL (Highest F1)
    best_model_name = res_df.loc[res_df['F1_Score'].idxmax()]['Model']
    best_model = models[best_model_name]
    print(f"\nWinning Model: {best_model_name}")

    # 6. EXPORT MODEL FOR TASK 3 (The missing part)
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/best_model_{dataset_name.replace(' ', '_').lower()}.joblib"
    joblib.dump(best_model, model_filename)
    
    print(f"SUCCESS: Model saved to {model_filename}")
    return best_model

# -------------------------------
# Execution Logic
# -------------------------------

if __name__ == "__main__":
    try:
        # Load your data
        X_f = pd.concat([pd.read_csv("data/processed/X_train_fraud.csv"), pd.read_csv("data/processed/X_test_fraud.csv")])
        y_f = pd.concat([pd.read_csv("data/processed/y_train_fraud.csv"), pd.read_csv("data/processed/y_test_fraud.csv")]).squeeze()

        # Run training and save
        train_and_select_best(X_f, y_f, dataset_name="Fraud")
        
    except Exception as e:
        print(f"Error: {e}")