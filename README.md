# Loan Prediction - End-to-End ML Project

> Best model: Random Forest | AUC 88.7% | F1 90.0% | Accuracy 85.4%

## Quick Start

pip install -r requirements.txt
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
uvicorn app.server:app --port 8000

## Project Structure

loan_prediction/
  data/raw/              Original train.csv (614 rows)
  data/processed/        Cleaned, scaled, SMOTE-balanced splits
  notebooks/eda.ipynb    Exploratory Data Analysis
  src/
    data/make_dataset.py       Ingest, impute, split
    features/build_features.py Feature engineering + SMOTE + PCA
    models/train_model.py      Train 6 models, GridSearchCV, threshold tuning
    models/predict_model.py    Inference with saved artifacts
  pipelines/training_pipeline.py  ZenML pipeline + MLflow tracking
  app/
    index.html   Production UI
    server.py    FastAPI backend
  models/        Saved .pkl artifacts
  reports/       Plots, metrics, confusion matrices

## Dataset
- Source: Kaggle Loan Prediction Dataset
- Rows: 614 applicants | Features: 12 raw -> 24 engineered
- Target: Loan_Status (Y=Approved / N=Rejected)
- Imbalance: 69% Approved, fixed with SMOTE

## Engineered Features
- TotalIncome = Applicant + CoApplicant income
- EMI = LoanAmount / Loan_Amount_Term
- BalanceIncome = TotalIncome - EMI x 1000
- LoanToIncome = LoanAmount / TotalIncome
- log_* = log1p of all skewed numerics

## Corrections Applied (False Positive Reduction)
- SMOTE: 69/31 -> 50/50 balance, AUC +13%
- class_weight={0:2, 1:1} on SVM
- Precision-Recall optimal threshold = 0.38
- StandardScaler + PCA retaining 95% variance

## Model Comparison (5-Fold CV)
  Random Forest      AUC 89.0% (BEST)
  Gradient Boosting  AUC 86.2%
  SVM                AUC 85.3%
  AdaBoost           AUC 83.3%
  KNN                AUC 82.8%
  Logistic Regression AUC 77.7%

## Final Test Results (threshold=0.38)
  Accuracy:  85.4%
  F1 Score:  90.0%
  ROC AUC:   85.2%
  Precision (Rejected): 86%
  Recall (Approved): 95%

## ZenML Pipeline
  pip install typing_extensions  # Python 3.8 fix
  zenml init
  python pipelines/training_pipeline.py
  mlflow ui  -> http://localhost:5000

## Docker
  docker build -t loan-prediction .
  docker run -p 8000:8000 loan-prediction
