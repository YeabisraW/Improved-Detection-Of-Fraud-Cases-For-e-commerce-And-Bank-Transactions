"""
Task 2 Pipeline:
- Model Training
- SHAP Explainability & Error Handling
- Strategic Business Recommendations & Narrative Contrast
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sys

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
try:
    X_train_e = np.load(REPORT_DIR / "X_train_ecom.npy", allow_pickle=True)
    y_train_e = np.load(REPORT_DIR / "y_train_ecom.npy", allow_pickle=True)
    X_train_c = np.load(REPORT_DIR / "X_train_cc.npy", allow_pickle=True)
    y_train_c = np.load(REPORT_DIR / "y_train_cc.npy", allow_pickle=True)

    X_train_e = np.asarray(X_train_e, dtype=float)
    y_train_e = np.asarray(y_train_e, dtype=int)
    X_train_c = np.asarray(X_train_c, dtype=float)
    y_train_c = np.asarray(y_train_c, dtype=int)
    print("✅ Data loaded successfully.")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# -----------------------------
# Train Models
# -----------------------------
print("Training models...")
ecom_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cc_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

ecom_model.fit(X_train_e, y_train_e)
cc_model.fit(X_train_c, y_train_c)
print("✅ Models trained successfully.")

# -----------------------------
# Load feature names
# -----------------------------
ecom_feature_names = pd.read_csv(REPORT_DIR / "ecom_feature_names.csv")["feature"].tolist()
cc_feature_names = pd.read_csv(REPORT_DIR / "cc_feature_names.csv")["feature"].tolist()

# -----------------------------
# SHAP Explainability Function
# -----------------------------
def compute_and_save_shap(model, X, feature_names, filename):
    print(f"--- Computing SHAP for {filename} ---")
    n_samples = min(500, X.shape[0]) 
    idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # FIX: Handle dimension mismatch (select Fraud class index 1)
    if isinstance(shap_values, list):
        final_shap = shap_values[1]
    elif len(shap_values.shape) == 3:
        final_shap = shap_values[:, :, 1]
    else:
        final_shap = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(final_shap, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename)
    plt.close()
    
    return final_shap, X_sample

# Execute SHAP
shap_e, X_e_shap = compute_and_save_shap(ecom_model, X_train_e, ecom_feature_names, "shap_summary_ecommerce.png")
shap_c, X_c_shap = compute_and_save_shap(cc_model, X_train_c, cc_feature_names, "shap_summary_creditcard.png")

# -----------------------------
# Strategic Business Recommendations & Narrative
# -----------------------------
def generate_strategic_report(model_name, model, shap_values, feature_names, top_n=5):
    # 1. SHAP Importance vs Built-in (Gini)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_shap_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    built_in_importances = model.feature_importances_
    top_gini_idx = np.argmax(built_in_importances)

    recommendations = []
    
    # Actionable Business Rules linking SHAP features to Fraud Controls
    for i in top_shap_idx:
        feat = feature_names[i]
        if any(k in feat.lower() for k in ["country", "ip", "location"]):
            recommendations.append(f"- [GEO-CONTROL]: {feat} is a primary driver. ACTION: Trigger 'Step-up Authentication' (OTP) for transactions from high-risk locations identified in SHAP clusters.")
        elif "amount" in feat.lower():
            recommendations.append(f"- [MONITORING]: {feat} significantly shifts risk. ACTION: Set dynamic velocity thresholds that flag transactions for manual review if they exceed 3x the user's weekly average.")
        elif any(k in feat.lower() for k in ["hour", "time", "day"]):
            recommendations.append(f"- [UX CHANGE]: {feat} indicates anomalous behavior. ACTION: If high-risk timing is detected, enforce biometric verification (face/thumbprint) before finalizing checkout.")
        else:
            recommendations.append(f"- [RULE REFINEMENT]: {feat} impact detected. ACTION: Integrate {feat} into the real-time scoring engine with a higher weight to improve precision.")

    # Narrative Contrast
    narrative = (
        f"\nNARRATIVE CONTRAST:\n"
        f"The model's built-in Gini Importance identifies '{feature_names[top_gini_idx]}' as the top feature for node splitting. "
        f"While Gini captures global relevance, it lacks directional insight. In contrast, SHAP analysis identifies '{feature_names[top_shap_idx[0]]}' "
        f"as the most critical driver of actual fraud probability. SHAP allows us to see how feature values directly push a prediction toward "
        f"fraud, enabling the targeted controls (like Biometrics and OTP) listed above, whereas built-in importance only highlights raw predictive power."
    )

    # Save Report
    out_file = REPORT_DIR / f"{model_name}_business_recommendations.txt"
    with open(out_file, "w") as f:
        f.write(f"BUSINESS STRATEGY REPORT: {model_name.upper()}\n")
        f.write("="*40 + "\n")
        f.write("\n".join(recommendations))
        f.write("\n" + narrative)
    
    print(f"✅ Strategic report saved for {model_name}")

# Run reporting
generate_strategic_report("ecommerce_model", ecom_model, shap_e, ecom_feature_names)
generate_strategic_report("creditcard_model", cc_model, shap_c, cc_feature_names)

print("\n🚀 Pipeline completed! Check the 'reports' folder for summary plots and strategy text files.")