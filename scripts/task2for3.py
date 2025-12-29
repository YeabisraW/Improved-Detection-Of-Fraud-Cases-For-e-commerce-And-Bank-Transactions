# ---------------------------------------------
# Task 2 for Task 3: Model Training & Dataset Prep
# ---------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# -------------------------------
# Config
# -------------------------------
datasets = {
    "E-commerce": "data/raw/Fraud_Data.csv",
    "Credit Card": "data/raw/creditcard.csv"
}

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------

def detect_target_column(df):
    if 'Class' in df.columns:
        return 'Class'
    elif 'class' in df.columns:
        return 'class'
    else:
        raise ValueError("Target column not found in dataset")

def preprocess_ecommerce(df):
    # Convert timestamps to numeric features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Encode categorical features
    df = pd.get_dummies(df, columns=['source','browser','sex'], drop_first=True)

    # Drop columns not needed
    df.drop(['user_id','signup_time','purchase_time','device_id','ip_address'], axis=1, inplace=True, errors='ignore')

    return df

def preprocess_creditcard(df):
    # Drop columns not needed
    df = df.copy()
    if 'Time' in df.columns:
        df.drop(['Time'], axis=1, inplace=True)
    return df

def align_columns_to_model(df, model):
    """Aligns DataFrame columns to model's training features"""
    if hasattr(model, "feature_names_in_"):
        model_cols = model.feature_names_in_
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[model_cols]
    return df

def train_best_model(X_train, y_train):
    """Trains a simple ensemble (Random Forest + XGBoost) and returns the best by F1"""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    best_model = None
    best_f1 = 0

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_train)
        f1 = f1_score(y_train, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    return best_model

# -------------------------------
# Main Processing Loop
# -------------------------------

for name, path in datasets.items():
    print(f"\nProcessing {name} dataset...")
    if not os.path.exists(path):
        print(f"⚠ Failed to load {name}: {path} does not exist.")
        continue

    df = pd.read_csv(path)
    try:
        target = detect_target_column(df)
    except ValueError as e:
        print(f"⚠ Target column not found in {name} dataset.")
        continue

    if name == "E-commerce":
        df = preprocess_ecommerce(df)
    else:
        df = preprocess_creditcard(df)

    # Split X, y
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model if not exists
    model_path = os.path.join(models_dir, f"best_model_{name.lower().replace(' ','_')}.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model for {name}")
    else:
        print(f"Training models for {name}...")
        model = train_best_model(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Saved model for {name}: {model_path}")

    # Align test set for SHAP analysis
    X_test_aligned = align_columns_to_model(X_test, model)

    # Save processed data for Task 3
    X_train.to_csv(os.path.join(processed_dir, f"X_train_{name.lower().replace(' ','_')}.csv"), index=False)
    X_test_aligned.to_csv(os.path.join(processed_dir, f"X_test_{name.lower().replace(' ','_')}.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, f"y_train_{name.lower().replace(' ','_')}.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, f"y_test_{name.lower().replace(' ','_')}.csv"), index=False)

    print(f"✅ Processed dataset saved for {name}: shape {X_test_aligned.shape}")

print("\nAll available datasets processed successfully.")
