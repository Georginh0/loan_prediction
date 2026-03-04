"""
pipelines/training_pipeline.py
────────────────────────────────
ZenML end-to-end training pipeline.

Install: pip install zenml mlflow
Init:    zenml init
Run:     python pipelines/training_pipeline.py
"""

from zenml import pipeline, step
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Tuple

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow


# ─── Steps ───────────────────────────────────────────────────────────────────


@step
def ingest_data(path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Load raw CSV."""
    df = pd.read_csv(path)
    print(f"[ingest] Loaded {df.shape}")
    return df


@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute + cap outliers."""
    df = df.copy()
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    print("[preprocess] Done ✅")
    return df


@step
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Encode + derive features."""
    df = df.copy()
    df.drop(columns=["Loan_ID"], inplace=True)
    df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, 1)
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"] * 1000
    df["LoanToIncome"] = df["LoanAmount"] / (df["TotalIncome"] + 1)
    for col in [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "TotalIncome",
        "BalanceIncome",
        "EMI",
    ]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    cat_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[features] Shape after engineering: {df.shape} ✅")
    return df


@step
def split_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


@step
def scale_and_reduce(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_final"],
    Annotated[pd.DataFrame, "X_test_final"],
]:
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train_sc),
        columns=[f"PC{i}" for i in range(pca.n_components_)],
    )
    X_test_pca = pd.DataFrame(
        pca.transform(X_test_sc), columns=[f"PC{i}" for i in range(pca.n_components_)]
    )
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    print(f"[scale/pca] PCA components: {pca.n_components_} ✅")
    return X_train_pca, X_test_pca


@step
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> str:
    """Train multiple models, pick best, log to MLflow."""
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_name, best_auc, best_model = None, 0.0, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_experiment("loan-prediction")

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            cv_auc = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc"
            ).mean()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            mlflow.log_params({"model": name})
            mlflow.log_metrics(
                {"cv_auc": cv_auc, "test_acc": acc, "test_f1": f1, "test_auc": auc}
            )
            mlflow.sklearn.log_model(model, artifact_path=name)

            print(f"  {name}: CV_AUC={cv_auc:.4f} | Test AUC={auc:.4f}")
            if auc > best_auc:
                best_auc, best_name, best_model = auc, name, model

    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\n[train] Best: {best_name}  AUC={best_auc:.4f} ✅")
    return best_name


# ─── Pipeline ─────────────────────────────────────────────────────────────────


@pipeline
def loan_prediction_pipeline():
    df = ingest_data()
    df = preprocess_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_f, X_test_f = scale_and_reduce(X_train, X_test)
    best_model = train_and_evaluate(X_train_f, X_test_f, y_train, y_test)


if __name__ == "__main__":
    loan_prediction_pipeline()
