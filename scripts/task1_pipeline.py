"""
Task 1 Pipeline: Data Cleaning, Geolocation Integration,
Feature Engineering, Encoding/Scaling, and SMOTE

Covers:
1) E-commerce Fraud Dataset
2) Credit Card Fraud Dataset

Outputs:
- Fraud by Country analysis (CSV + Top 10 plot)
- Cleaned & resampled datasets ready for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data/raw"
REPORT_DIR = BASE_DIR / "reports"

ECOM_FILE = DATA_DIR / "Fraud_Data.csv"
CC_FILE = DATA_DIR / "creditcard.csv"
IP_COUNTRY_FILE = DATA_DIR / "IpAddress_to_Country.csv"

REPORT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
ecom_df = pd.read_csv(ECOM_FILE)
cc_df = pd.read_csv(CC_FILE)
ip_country_df = pd.read_csv(IP_COUNTRY_FILE)

print("Datasets loaded successfully")

# -----------------------------
# Geolocation Merge
# -----------------------------
ip_country_df.rename(
    columns={
        "lower_bound_ip_address": "ip_lower",
        "upper_bound_ip_address": "ip_upper"
    },
    inplace=True
)

ecom_df["ip_address"] = ecom_df["ip_address"].astype(int)

def map_ip_to_country(ip):
    match = ip_country_df[
        (ip_country_df["ip_lower"] <= ip) &
        (ip_country_df["ip_upper"] >= ip)
    ]
    if len(match) == 0:
        return "Unknown"
    return match.iloc[0]["country"]

ecom_df["country"] = ecom_df["ip_address"].apply(map_ip_to_country)

# -----------------------------
# Fraud-by-Country Analysis
# -----------------------------
fraud_country = (
    ecom_df
    .groupby("country")["class"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"class": "fraud_rate"})
)

fraud_country.to_csv(REPORT_DIR / "fraud_rate_by_country.csv", index=False)
print("Fraud-by-country report saved")

# Top 10 fraud countries plot
top10 = fraud_country.sort_values("fraud_rate", ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.bar(top10["country"], top10["fraud_rate"], color="tomato")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Fraud Rate")
plt.title("Top 10 Countries by E-commerce Fraud Rate")
plt.tight_layout()
plt.savefig(REPORT_DIR / "top10_fraud_countries.png")
plt.close()
print("Top 10 fraud countries plot saved")

# -----------------------------
# Feature Engineering
# -----------------------------
ecom_df["signup_time"] = pd.to_datetime(ecom_df["signup_time"])
ecom_df["purchase_time"] = pd.to_datetime(ecom_df["purchase_time"])

ecom_df["account_age_hours"] = (
    (ecom_df["purchase_time"] - ecom_df["signup_time"]).dt.total_seconds() / 3600
)
ecom_df["purchase_hour"] = ecom_df["purchase_time"].dt.hour

# Drop leakage / unused columns
ecom_df.drop(
    columns=["signup_time", "purchase_time", "ip_address", "user_id", "device_id"],
    inplace=True,
    errors="ignore"
)

# -----------------------------
# Split Features / Target
# -----------------------------
X_ecom = ecom_df.drop("class", axis=1)
y_ecom = ecom_df["class"]

X_cc = cc_df.drop("Class", axis=1)
y_cc = cc_df["Class"]

# -----------------------------
# Preprocessing Pipelines
# -----------------------------
cat_features = X_ecom.select_dtypes(include="object").columns.tolist()
num_features = X_ecom.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_ecom, y_ecom, test_size=0.2, stratify=y_ecom, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=42
)

# -----------------------------
# Apply Preprocessing
# -----------------------------
X_train_e = preprocessor.fit_transform(X_train_e)
X_test_e = preprocessor.transform(X_test_e)

scaler_cc = StandardScaler()
X_train_c = scaler_cc.fit_transform(X_train_c)
X_test_c = scaler_cc.transform(X_test_c)

# -----------------------------
# Apply SMOTE
# -----------------------------
smote = SMOTE(random_state=42)

X_train_e_sm, y_train_e_sm = smote.fit_resample(X_train_e, y_train_e)
X_train_c_sm, y_train_c_sm = smote.fit_resample(X_train_c, y_train_c)

print("SMOTE applied successfully")

# -----------------------------
# Save Outputs
# -----------------------------
np.save(REPORT_DIR / "X_train_ecom.npy", X_train_e_sm)
np.save(REPORT_DIR / "y_train_ecom.npy", y_train_e_sm)

np.save(REPORT_DIR / "X_train_cc.npy", X_train_c_sm)
np.save(REPORT_DIR / "y_train_cc.npy", y_train_c_sm)

print("Task 1 pipeline completed successfully")
