# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## 📜 Business Context
Fraudulent activities in e-commerce and banking result in billions of dollars in losses annually. This project develops a robust machine learning pipeline to identify suspicious transactions. The primary challenge is the extreme class imbalance (fraud is rare). Our solution focuses on maximizing **Recall** to catch fraud while maintaining high **F1/PR-AUC** to ensure legitimate customers are not negatively impacted by false positives.

## 📂 Project Structure
- data/: Contains raw/ datasets and processed/ files. All data is split using Stratified Sampling to maintain class ratios.
- models/: Serialized .joblib files of the best-performing models (e.g., Tuned Random Forest, XGBoost).
- reports/: Model comparison tables, evaluation metrics (F1, PR-AUC), and SHAP importance plots.
- scripts/: Production-ready Python modules for data cleaning, engineering, and model training.
- notebooks/: Jupyter notebooks for exploratory data analysis (EDA) and experimentation.
- tests/: Unit tests for validating data integrity and model consistency.
- requirements.txt: Comprehensive list of dependencies with specific versions for reproducibility.

## 🛠️ Setup & Installation
1. Clone the Repository:
   git clone https://github.com/YeabisraW/Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions.git
   cd Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions

2. Environment Setup:
   python -m venv venv
   # Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

## 🚀 Usage
### 1. Preprocessing & Feature Engineering
Execute the scripts to clean data and map IP addresses to countries. Key Features include time-based velocity, frequency-based features, and geolocation analysis.

### 2. Modeling & Evaluation
Run the modeling scripts (e.g., scripts/task2_modeling.py) to execute the full pipeline. This includes Stratified Splitting to maintain fraud ratios, Baseline Modeling using Logistic Regression for interpretability, Hyperparameter Tuning via GridSearchCV, and Explainability using SHAP summary plots to visualize global feature importance.

## 📊 Model Strategy & Selection
While ensemble models (Random Forest/XGBoost) provide superior predictive power (higher F1 and PR-AUC), we maintain the Logistic Regression baseline for scenarios where Business Explainability and low-latency decision-making are prioritized over marginal performance gains. Logistic Regression coefficients provide a clear, linear relationship between features and the probability of fraud.

## 🧪 Key Technical Features
- Class Imbalance: Handled via Stratified Splitting and SMOTE during preprocessing to ensure the model learns fraud patterns effectively.
- Geolocation: Mapping transaction IPs to specific countries for location-based risk scoring.
- XAI (Explainable AI): Using SHAP (SHapley Additive exPlanations) to provide transparency for individual model predictions.

## 📜 License
For educational and research purposes.