"""

────────────────────────
Load raw data, impute missing values, cap outliers,
encode categoricals, and produce train/test splits.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
RAW_PATH = "data/raw/train.csv"
PROCESSED = "data/processed"

os.makedirs(PROCESSED, exist_ok=True)


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] Shape: {df.shape}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    print("[impute] Missing values treated ✅")
    return df


def cap_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    print(f"[outliers] Capped: {cols} ✅")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Loan_ID"], inplace=True)
    df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)
    cat_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("[encode] Categorical encoding done ✅")
    return df


def split_and_save(df: pd.DataFrame):
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train.to_csv(f"{PROCESSED}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED}/y_test.csv", index=False)
    print(
        f"[split] Train {X_train.shape} | Test {X_test.shape} — saved to {PROCESSED} ✅"
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_raw()
    df = impute_missing(df)
    df = cap_outliers(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
    df.to_csv(f"{PROCESSED}/cleaned.csv", index=False)
    df = encode(df)
    split_and_save(df)
