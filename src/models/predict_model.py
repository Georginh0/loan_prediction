"""
src/models/predict_model.py

"""

import pandas as pd
import numpy as np
import joblib
import json


def load_artifacts(model_dir: str = "models"):
    model = joblib.load(f"{model_dir}/best_model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    with open(f"{model_dir}/model_meta.json") as f:
        meta = json.load(f)

    # ============================================================
    # Option 1 -- load the optimal threshold saved during training
    # WHERE: load_artifacts(), consumed by predict() below
    # WHY:   Keeps inference consistent with training-time threshold
    #        Prevents the 18 False Positives problem at serve time
    # ============================================================
    try:
        threshold = joblib.load(f"{model_dir}/threshold.pkl")
        print(f"[artifacts] Loaded optimal threshold: {threshold:.3f}")
    except FileNotFoundError:
        threshold = meta.get("optimal_threshold", 0.5)
        print(f"[artifacts] Falling back to meta threshold: {threshold:.3f}")

    return model, scaler, meta, threshold


def preprocess_input(raw: dict, scaler, feature_names: list) -> pd.DataFrame:
    """
    raw: dict of raw applicant data (same structure as training CSV).
    Returns scaled DataFrame ready for prediction.
    """
    df = pd.DataFrame([raw])

    # Derived features -- must mirror build_features.py exactly
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"].replace(0, np.nan).fillna(1))
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
        if col in df.columns:
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

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return pd.DataFrame(scaler.transform(df), columns=feature_names)


def predict(raw: dict, threshold: float = None) -> dict:
    """
    Predict loan approval for a single applicant.

    Args:
        raw:       dict of raw applicant features
        threshold: optional manual override (default: use saved optimal threshold)

    Returns:
        dict with prediction label, probability, and threshold used
    """
    model, scaler, meta, optimal_threshold = load_artifacts()

    # Option 1: use saved optimal threshold unless manually overridden
    decision_threshold = threshold if threshold is not None else optimal_threshold

    X = preprocess_input(raw, scaler, meta["features"])
    prob = model.predict_proba(X)[0][1]

    # Option 1 -- apply optimal threshold (not hardcoded 0.5)
    label = "Approved" if prob >= decision_threshold else "Rejected"

    return {
        "prediction": label,
        "approval_probability": round(float(prob), 4),
        "threshold_used": round(decision_threshold, 3),
    }


if __name__ == "__main__":
    sample = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }
    result = predict(sample)
    print(f"\nPrediction  : {result['prediction']}")
    print(f"Probability : {result['approval_probability']}")
    print(f"Threshold   : {result['threshold_used']}")
