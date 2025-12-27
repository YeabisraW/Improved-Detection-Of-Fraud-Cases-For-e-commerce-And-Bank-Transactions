# =========================================
# TASK 2 – MODEL BUILDING & TRAINING
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

# E-commerce fraud data
fraud = pd.read_csv("data/raw/Fraud_Data.csv")

# Credit card fraud data
credit = pd.read_csv("data/raw/creditcard.csv")

print("Datasets loaded successfully")

# -----------------------------------------
# 2. BASIC PREPROCESSING
# -----------------------------------------

# -------- E-commerce --------
fraud['signup_time'] = pd.to_datetime(fraud['signup_time'])
fraud['purchase_time'] = pd.to_datetime(fraud['purchase_time'])

fraud['hour'] = fraud['purchase_time'].dt.hour
fraud['dayofweek'] = fraud['purchase_time'].dt.dayofweek
fraud['time_since_signup'] = (
    fraud['purchase_time'] - fraud['signup_time']
).dt.total_seconds() / 3600

fraud = fraud.drop(columns=['signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id'])

fraud = pd.get_dummies(fraud, drop_first=True)

X_fraud = fraud.drop('class', axis=1)
y_fraud = fraud['class']

# -------- Credit card --------
X_credit = credit.drop('Class', axis=1)
y_credit = credit['Class']

# -----------------------------------------
# 3. TRAIN–TEST SPLIT (STRATIFIED)
# -----------------------------------------

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_credit, y_credit, test_size=0.2, stratify=y_credit, random_state=42
)

# -----------------------------------------
# 4. BASELINE MODEL – LOGISTIC REGRESSION
# -----------------------------------------

log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000))
])

log_pipeline.fit(Xf_train, yf_train)
yf_pred = log_pipeline.predict(Xf_test)
yf_proba = log_pipeline.predict_proba(Xf_test)[:, 1]

print("\n--- Logistic Regression (E-commerce) ---")
print("F1-score:", f1_score(yf_test, yf_pred))
print("AUC-PR:", average_precision_score(yf_test, yf_proba))
print(confusion_matrix(yf_test, yf_pred))

# -----------------------------------------
# 5. ENSEMBLE MODEL – RANDOM FOREST
# -----------------------------------------

rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(Xf_train, yf_train)
rf_pred = rf_pipeline.predict(Xf_test)
rf_proba = rf_pipeline.predict_proba(Xf_test)[:, 1]

print("\n--- Random Forest (E-commerce) ---")
print("F1-score:", f1_score(yf_test, rf_pred))
print("AUC-PR:", average_precision_score(yf_test, rf_proba))
print(confusion_matrix(yf_test, rf_pred))

# -----------------------------------------
# 6. STRATIFIED CROSS-VALIDATION
# -----------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_f1 = cross_val_score(
    rf_pipeline, X_fraud, y_fraud,
    scoring='f1', cv=skf
)

cv_aucpr = cross_val_score(
    rf_pipeline, X_fraud, y_fraud,
    scoring='average_precision', cv=skf
)

print("\n--- Cross Validation (Random Forest) ---")
print(f"F1-score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"AUC-PR:   {cv_aucpr.mean():.4f} ± {cv_aucpr.std():.4f}")

# -----------------------------------------
# 7. MODEL COMPARISON SUMMARY
# -----------------------------------------

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'F1-score': [
        f1_score(yf_test, yf_pred),
        f1_score(yf_test, rf_pred)
    ],
    'AUC-PR': [
        average_precision_score(yf_test, yf_proba),
        average_precision_score(yf_test, rf_proba)
    ]
})

print("\nModel Comparison:")
print(results)
