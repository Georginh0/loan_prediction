"""
src/features/build_features.py
───────────────────────────────
Feature Engineering:
  1. Log-transform skewed numeric columns
  2. Create interaction features (TotalIncome, EMI, BalanceIncome)
  3. StandardScaler
  4. PCA for dimensionality reduction
  NEW — Option 3: SMOTE to fix class imbalance (69/31 -> 50/50)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # Option 3 -- SMOTE import

PROCESSED = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_splits():
    X_train = pd.read_csv(f"{PROCESSED}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
    X["EMI"] = X["LoanAmount"] / X["Loan_Amount_Term"].replace(0, np.nan).fillna(1)
    X["BalanceIncome"] = X["TotalIncome"] - (X["EMI"] * 1000)
    X["LoanToIncome"] = X["LoanAmount"] / (X["TotalIncome"] + 1)
    skewed = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "TotalIncome",
        "BalanceIncome",
        "EMI",
    ]
    for col in skewed:
        if col in X.columns:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
    print(f"[features] Engineered features. New shape: {X.shape}")
    return X


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("[scale] StandardScaler fitted & saved")
    return X_train_sc, X_test_sc, scaler


# ============================================================
# Option 3 -- SMOTE
# WHERE: called AFTER scale_features(), BEFORE model training
# WHY:   Fixes 69/31 imbalance that caused 18 False Positives
# RULE:  Apply to TRAINING data only -- never touch test set
# ============================================================
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print(f"[SMOTE] Before : {y_train.value_counts().to_dict()}")
    print(f"[SMOTE] After  : {y_res.value_counts().to_dict()}")
    return X_res, y_res


def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, variance: float = 0.95):
    pca = PCA(n_components=variance, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    n_components = pca.n_components_
    print(
        f"[PCA] Components selected: {n_components} (explains {variance * 100}% variance)"
    )
    col_names = [f"PC{i + 1}" for i in range(n_components)]
    X_train_pca = pd.DataFrame(X_train_pca, columns=col_names)
    X_test_pca = pd.DataFrame(X_test_pca, columns=col_names)
    joblib.dump(pca, f"{MODEL_DIR}/pca.pkl")
    return X_train_pca, X_test_pca, pca


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_splits()

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
    X_train_sc.to_csv(f"{PROCESSED}/X_train_scaled.csv", index=False)
    X_test_sc.to_csv(f"{PROCESSED}/X_test_scaled.csv", index=False)

    # Option 3: SMOTE on scaled train only -- X_test stays untouched
    X_train_smote, y_train_smote = apply_smote(X_train_sc, y_train)
    X_train_smote.to_csv(f"{PROCESSED}/X_train_smote.csv", index=False)
    y_train_smote.to_csv(f"{PROCESSED}/y_train_smote.csv", index=False)

    # PCA applied on SMOTE-balanced training data
    X_train_pca, X_test_pca, _ = apply_pca(X_train_smote, X_test_sc, variance=0.95)
    X_train_pca.to_csv(f"{PROCESSED}/X_train_pca.csv", index=False)
    X_test_pca.to_csv(f"{PROCESSED}/X_test_pca.csv", index=False)

    print("\n[build_features] All feature engineering complete")
