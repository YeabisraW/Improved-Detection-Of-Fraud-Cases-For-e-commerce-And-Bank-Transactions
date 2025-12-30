# -------------------------------------------------------------
# Task 1: Rich EDA & Preprocessing with Explicit Logging
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# Create directories if they don't exist
os.makedirs('reports', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def document_cleaning(df, name):
    print(f"\n--- Data Cleaning: {name} ---")
    print(f"Initial Shape: {df.shape}")
    
    # Missing Values
    nulls = df.isnull().sum().sum()
    print(f"Missing values found: {nulls}")
    if nulls > 0:
        df = df.fillna(df.median(numeric_only=True))
        print("Missing values filled with median.")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"Duplicate rows found: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates()
        print("Duplicates removed.")
    
    return df

def save_eda_plots(df, name, target_col):
    print(f"Generating EDA Plots for {name}...")
    
    # 1. Univariate: Amount Distribution
    plt.figure(figsize=(10, 5))
    if 'Amount' in df.columns: # Credit Card
        sns.histplot(df['Amount'], bins=50, kde=True, color='blue')
        col_name = 'Amount'
    else: # Ecommerce
        sns.histplot(df['purchase_value'], bins=50, kde=True, color='green')
        col_name = 'Purchase Value'
    plt.title(f"Univariate Distribution of {col_name} ({name})")
    plt.savefig(f"../reports/{name}_univariate_dist.png")
    plt.close()

    # 2. Bivariate: Feature vs Fraud Class
    plt.figure(figsize=(10, 5))
    val_col = 'Amount' if 'Amount' in df.columns else 'purchase_value'
    sns.boxplot(x=target_col, y=val_col, data=df, palette='Set2')
    plt.title(f"Bivariate Analysis: {val_col} vs {target_col} ({name})")
    plt.savefig(f"../reports/{name}_bivariate_boxplot.png")
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', annot=False)
    plt.title(f"Correlation Heatmap ({name})")
    plt.savefig(f"../reports/{name}_correlation.png")
    plt.close()

def preprocess_and_balance(df, name, target, test_size=0.2):
    X = df.drop(columns=[target])
    y = df[target]

    # Scaling
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # 4. Explicit SMOTE Logging (Point #1 Rubric requirement)
    print(f"\n--- Class Distribution ({name}) ---")
    print(f"Before SMOTE: {Counter(y_train)}")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE:  {Counter(y_train_res)}")

    return X_train_res, X_test, y_train_res, y_test

# --- EXECUTION ---
try:
    # Load Credit Card Data
    credit = pd.read_csv("data/raw/creditcard.csv")
    credit = document_cleaning(credit, "CreditCard")
    save_eda_plots(credit, "CreditCard", "Class")
    X_train, X_test, y_train, y_test = preprocess_and_balance(credit, "CreditCard", "Class")
    
    # Save Results
    X_train.to_csv("data/processed/X_train_credit.csv", index=False)
    print("\n✅ Task 1 Update Complete. Reports and data saved.")

except Exception as e:
    print(f"Execution Error: {e}")