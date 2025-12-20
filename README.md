# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Overview
This project detects fraudulent transactions in e-commerce and bank credit card datasets using machine learning, feature engineering, and SMOTE to handle class imbalance. Includes geolocation analysis and explainability using SHAP.

## Project Structure

fraud-detection/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # Jupyter notebooks
├── src/                    # Python scripts
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── models/                 # Saved model artifacts
├── .github/                # GitHub workflows
├── .vscode/                # VS Code settings
├── requirements.txt        # Python dependencies
└── README.md               # Project overview

## Datasets

- E-commerce: Fraud_Data.csv, IpAddress_to_Country.csv
- Credit Card: creditcard.csv

## Setup

\`\`\`bash
git clone https://github.com/YeabisraW/Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions.git
cd Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
\`\`\`

## Usage

### Notebooks
- eda-fraud-data.ipynb — EDA for e-commerce data
- eda-creditcard.ipynb — EDA for credit card data
- feature-engineering.ipynb — Preprocessing and feature creation
- modeling.ipynb — Train and evaluate models
- shap-explainability.ipynb — Interpret model predictions

### Scripts
- src/ and scripts/ contain reusable functions for preprocessing, training, and evaluation.

## Key Features
- Handles imbalanced datasets using SMOTE
- Geolocation-based fraud detection
- Time-based and frequency-based features
- Numeric scaling and categorical encoding
- Explainable AI (SHAP) for model interpretability

## License
For educational and research purposes.
EOL
