# Improved Detection of Fraud Cases for E-commerce and Bank Transactions
## 📜 Business Context
Fraudulent activities in e-commerce and banking result in billions of dollars in losses annually. This project develops a robust end-to-end machine learning pipeline to identify suspicious transactions in both e-commerce and bank transaction datasets. The primary challenge addressed is extreme class imbalance, where fraudulent cases represent a very small fraction of all transactions. The solution prioritizes high Recall to minimize missed fraud cases, while also maintaining strong F1-score and PR-AUC to reduce unnecessary false positives that may inconvenience legitimate customers.

This repository follows a task-based structure aligned with typical data science project workflows, including data preparation, modeling, evaluation, and explainability (Task 3).

## 📂 Project Structure
- data/: Contains raw/ and processed/ datasets. All splits are performed using stratified sampling to preserve the original fraud-to-non-fraud ratio.
- models/: Directory reserved for trained models. Large serialized model files (.joblib) are intentionally excluded from Git tracking and ignored via .gitignore due to GitHub size limits.
- reports/: Stores model comparison tables, evaluation metrics (Mean ± Std), and explainability outputs such as SHAP plots and CSV summaries.
- scripts/: Modular, production-ready Python scripts for preprocessing, feature engineering, modeling, evaluation, and explainability (Tasks 1–3).
- notebooks/: Jupyter notebooks used for exploratory data analysis (EDA), prototyping, and validation experiments.
- tests/: Unit tests to validate preprocessing logic, feature consistency, and model behavior.
- requirements.txt: Complete list of Python dependencies with fixed versions to ensure reproducibility.

## 🛠️ Setup & Installation
1. Clone the repository:
   git clone https://github.com/YeabisraW/Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions.git
   cd Improved-Detection-Of-Fraud-Cases-For-e-commerce-And-Bank-Transactions

2. Create and activate a virtual environment:
   python -m venv venv
   Windows: venv\Scripts\activate
   Linux/Mac: source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## 🚀 Usage
### Task 1: Data Preprocessing & Feature Engineering
Run preprocessing scripts to clean the datasets, handle missing values, encode categorical variables, and engineer fraud-relevant features. Key engineered features include transaction velocity, frequency-based behavioral features, and IP-based geolocation attributes.

### Task 2: Modeling & Evaluation
Execute modeling scripts to train and evaluate multiple machine learning models:
- Stratified Splitting: Ensures consistent class distribution across training and test sets using stratified sampling.
- Hyperparameter Tuning: GridSearchCV is applied to ensemble models to optimize Recall, F1-score, and PR-AUC.
- Cross-Validation Reporting: Performance metrics are aggregated across 5-fold cross-validation to ensure robustness and statistical reliability.

### Task 3: Model Explainability (XAI)
Explainability is implemented using SHAP to interpret the selected model:
- Global Explainability: Feature importance analysis to identify the most influential predictors of fraud.
- Local Explainability: Instance-level explanations for False Positives and False Negatives to understand model decisions in critical cases.
- Visualization: Force plots and summary plots are generated to support transparent decision-making and stakeholder trust.

## 📊 Model Performance & Selection
### Cross-Validation Results (Mean ± Std)
| Model | Mean F1-Score | Std Dev (±) | Mean PR-AUC |
|------|---------------|-------------|-------------|
| Logistic Regression | 0.824 | 0.021 | 0.801 |
| Random Forest | 0.912 | 0.012 | 0.895 |
| XGBoost (Selected) | 0.935 | 0.011 | 0.928 |

### Selection Rationale
XGBoost was selected as the final production model based on the following:
- Statistical Stability: Achieved the highest Mean PR-AUC and F1-score with the lowest variance, indicating consistent performance across folds.
- Fraud Sensitivity: Gradient boosting effectively captures non-linear fraud patterns, significantly reducing False Negatives.
- Explainability Support: Seamless integration with SHAP enables both global and local interpretability, satisfying transparency requirements.

## 🧪 Key Technical Features
- Class Imbalance Handling: Managed through stratified sampling and metric-driven optimization.
- Geolocation Analysis: IP-to-country mapping enhances risk profiling based on transaction origin.
- Explainable AI (XAI): SHAP-based explanations provide insight into model behavior for auditing and trust.
- Reproducibility: Fixed random seeds, versioned dependencies, and structured scripts ensure consistent results.

## ⚠️ Note on Model Files
Trained model files (.joblib) are excluded from version control due to GitHub file size constraints. Users can regenerate models locally by running the Task 2 modeling scripts.

## 📜 License
This project is intended for educational and research purposes only.
