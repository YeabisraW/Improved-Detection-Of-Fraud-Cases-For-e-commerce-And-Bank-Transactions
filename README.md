# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## 📜 Business Context
Fraudulent activities in e-commerce and banking result in billions of dollars in losses annually. This project develops a robust machine learning pipeline to identify suspicious transactions. The primary challenge is the extreme class imbalance (fraud is rare). Our solution focuses on maximizing Recall to catch fraud while maintaining high F1/PR-AUC to ensure legitimate customers are not negatively impacted by false positives.

## 📂 Project Structure
- data/: Contains raw/ datasets and processed/ files. All data is split using Stratified Sampling to maintain class ratios.
- models/: Serialized .joblib files of the best-performing models (e.g., Tuned Random Forest, XGBoost).
- reports/: Model comparison tables, evaluation metrics (Mean ± Std), and SHAP importance plots.
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
Run the modeling scripts to execute the full pipeline:
- Stratified Splitting: Implemented using train_test_split(stratify=y) to ensure the fraud ratio remains consistent across sets.
- Hyperparameter Tuning: Uses GridSearchCV to optimize Ensemble models for the best balance of precision and recall.
- Cross-Validation Reporting: Metrics are aggregated across 5 folds to ensure statistical stability.

## 📊 Model Performance & Selection
### Cross-Validation Results (Mean ± Std)
| Model | Mean F1-Score | Std Dev (±) | Mean PR-AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.824 | 0.021 | 0.801 |
| Random Forest | 0.912 | 0.012 | 0.895 |
| XGBoost (Selected) | 0.935 | 0.011 | 0.928 |

### Selection Rationale
After comparing the models, XGBoost was selected as the primary production model:
- Statistical Stability: It achieved the highest Mean PR-AUC and F1-score with the lowest standard deviation (±0.011), proving it is the most reliable model.
- Performance vs. Interpretability: The non-linear complexity of fraud patterns required the gradient boosting approach of XGBoost to minimize False Negatives.
- Explainability: We utilize SHAP values to interpret decisions, providing both global and local transaction-level explanations.

## 🧪 Key Technical Features
- Class Imbalance: Handled via Stratified Splitting and targeted metric optimization during GridSearch.
- Geolocation: Mapping transaction IPs to specific countries for location-based risk scoring.
- XAI (Explainable AI): Using SHAP to provide transparency for individual model predictions.

## 📜 License
For educational and research purposes.    