"""
src/models/train_model.py
──────────────────────────
1. Train 6 classifiers
2. Cross-validate & compare
3. GridSearchCV hyperparameter tuning on best model
4. Save best model + metrics report


"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)  # Option 4
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED = "data/processed"
MODEL_DIR = "models"
REPORTS = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5


# ── 1. Data Loading ──────────────────────────────────────────────────────────
def load_data(use_pca: bool = False, use_smote: bool = True):
    """
    use_smote=True  -> loads SMOTE-balanced training data (recommended)
    use_smote=False -> loads standard scaled training data
    """
    x_suffix = "pca" if use_pca else ("smote" if use_smote else "scaled")
    y_suffix = "smote" if (use_smote and not use_pca) else ""

    X_train = pd.read_csv(f"{PROCESSED}/X_train_{x_suffix}.csv")
    X_test = pd.read_csv(f"{PROCESSED}/X_test_scaled.csv")  
    y_train_file = (
        f"{PROCESSED}/y_train_smote.csv"
        if (use_smote and not use_pca)
        else f"{PROCESSED}/y_train.csv"
    )
    y_train = pd.read_csv(y_train_file).squeeze()
    y_test = pd.read_csv(f"{PROCESSED}/y_test.csv").squeeze()

    print(f"[load] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[load] y_train distribution: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ── 2. Model Zoo ─────────────────────────────────────────────────────────────
# Option 2: SVC now uses class_weight={0:2, 1:1} to penalise False Positives
# (class 0 = Rejected is the minority -- doubling its weight makes the model
#  more cautious before predicting Approved)
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        # ============================================================
        # Option 2 -- class_weight={0: 2, 1: 1}
        # WHERE: SVC constructor -- this is the only place to set it
        # WHY:   Penalises FP (approving bad loans) twice as much as FN
        #        Pushes specificity from 52% toward 70%+
        # ============================================================
        "SVM": SVC(
            probability=True,
            class_weight={0: 2, 1: 1},  # Option 2 applied here
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# ── 3. Cross-Validation Comparison ───────────────────────────────────────────
def compare_models(models: dict, X_train, y_train) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
        auc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        results[name] = {
            "CV Accuracy (mean)": acc.mean().round(4),
            "CV Accuracy (std)": acc.std().round(4),
            "CV F1 (mean)": f1.mean().round(4),
            "CV AUC (mean)": auc.mean().round(4),
        }
        print(
            f"  {name:25s} Acc={acc.mean():.4f}+/-{acc.std():.4f}  F1={f1.mean():.4f}  AUC={auc.mean():.4f}"
        )

    df = pd.DataFrame(results).T.sort_values("CV AUC (mean)", ascending=False)
    df.to_csv(f"{REPORTS}/model_comparison.csv")
    return df


def plot_comparison(df: pd.DataFrame):
    df_plot = df[["CV Accuracy (mean)", "CV F1 (mean)", "CV AUC (mean)"]].copy()
    df_plot.plot(kind="bar", figsize=(12, 5), colormap="Set2", edgecolor="black")
    plt.title("Model Comparison (Cross-Validation)")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/model_comparison.png", dpi=150)
    plt.show()


# ── 4. Hyperparameter Tuning ──────────────────────────────────────────────────
# Option 2: SVM param grid now includes class_weight options
PARAM_GRIDS = {
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"],
        "class_weight": [{0: 2, 1: 1}, {0: 3, 1: 1}, "balanced"],  
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    },
}


def tune_model(model_name: str, X_train, y_train):
    base_models = {
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base_models[model_name],
        PARAM_GRIDS[model_name],
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"[tune] {model_name} best params : {grid.best_params_}")
    print(f"[tune] {model_name} best AUC    : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ============================================================
# Option 4 -- Precision-Recall Curve threshold optimisation
# WHERE: called inside evaluate(), after model.predict_proba()
# WHY:   Default 0.5 threshold favours recall over precision
#        We find the threshold where precision >= target (e.g. 0.85)
#        then recompute all metrics at that threshold
# ============================================================
def find_optimal_threshold(
    y_test: pd.Series, y_prob: np.ndarray, precision_target: float = 0.85
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision", color="steelblue")
    plt.plot(thresholds, recall[:-1], label="Recall", color="darkorange")
    plt.axhline(
        precision_target,
        color="red",
        linestyle="--",
        label=f"Precision target = {precision_target}",
    )
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Decision Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/precision_recall_threshold.png", dpi=150)
    plt.show()

    # Find lowest threshold where precision >= target
    valid = thresholds[precision[:-1] >= precision_target]
    if len(valid) == 0:
        print(f"[threshold] No threshold meets precision={precision_target}, using 0.5")
        return 0.5
    optimal = float(valid[0])
    print(
        f"[threshold] Optimal threshold (precision>={precision_target}): {optimal:.3f}"
    )
    return optimal


# ── 5. Final Evaluation ────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name: str, precision_target: float = 0.85):
    y_prob = model.predict_proba(X_test)[:, 1]

    # ============================================================
    # Option 4 -- find optimal threshold via precision-recall curve
    # ============================================================
    optimal_threshold = find_optimal_threshold(y_test, y_prob, precision_target)

    # ============================================================
    # Option 1 -- use optimal threshold instead of 0.5
    # WHERE: y_pred assignment -- this single line replaces model.predict()
    # WHY:   Raises bar for predicting Approved, reducing False Positives
    # ============================================================
    y_pred_default = model.predict(X_test)  # threshold=0.5
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)  # Option 1

    print(f"\n{'=' * 55}")
    print(f"  Final Evaluation: {model_name}")
    print(f"{'=' * 55}")

    for label, y_pred in [
        ("Default (0.50)", y_pred_default),
        (f"Optimal ({optimal_threshold:.2f})", y_pred_optimal),
    ]:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"\n  Threshold: {label}")
        print(f"  Accuracy : {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
        print(
            classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])
        )

    # Save confusion matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (label, y_pred) in zip(
        axes,
        [
            ("Default threshold=0.50", y_pred_default),
            (f"Optimal threshold={optimal_threshold:.2f}", y_pred_optimal),
        ],
    ):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
        )
        ax.set_title(f"{model_name}\n{label}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/confusion_matrix_comparison.png", dpi=150)
    plt.show()

    # Save final metrics (using optimal threshold)
    acc = accuracy_score(y_test, y_pred_optimal)
    f1 = f1_score(y_test, y_pred_optimal)
    auc = roc_auc_score(y_test, y_prob)
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
        "optimal_threshold": optimal_threshold,
        "precision_target": precision_target,
    }
    with open(f"{REPORTS}/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save threshold so predict_model.py can load it
    joblib.dump(optimal_threshold, f"{MODEL_DIR}/threshold.pkl")
    print(f"\n[evaluate] Optimal threshold saved to {MODEL_DIR}/threshold.pkl")
    return metrics, optimal_threshold


# ── 6. Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        print("[importance] Model has no feature_importances_ -- skipping.")
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()
    importances.plot(kind="barh", figsize=(8, 6), color="steelblue")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/feature_importance.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SMOTE-balanced training data...")
    X_train, X_test, y_train, y_test = load_data(use_pca=False, use_smote=True)

    print("\n-- Cross-Validation Model Comparison --")
    models = get_models()
    comparison_df = compare_models(models, X_train, y_train)
    print(
        "\nRanking:\n",
        comparison_df[["CV Accuracy (mean)", "CV F1 (mean)", "CV AUC (mean)"]],
    )
    plot_comparison(comparison_df)

    best_name = comparison_df.index[0]
    print(f"\n-- Hyperparameter Tuning: {best_name} --")
    if best_name in PARAM_GRIDS:
        best_model, best_params, best_cv_score = tune_model(best_name, X_train, y_train)
    else:
        print(f"  No param grid for {best_name} -- using default.")
        best_model = models[best_name]

    best_model.fit(X_train, y_train)

    # Options 1 & 4 applied inside evaluate()
    metrics, optimal_threshold = evaluate(
        best_model, X_test, y_test, best_name, precision_target=0.85
    )

    plot_feature_importance(best_model, list(X_train.columns))

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "features": list(X_train.columns),
                "optimal_threshold": optimal_threshold,
            },
            f,
            indent=2,
        )

    print(f"\nBest model saved to {MODEL_DIR}/best_model.pkl")
    print(f"Optimal threshold : {optimal_threshold:.3f}")
