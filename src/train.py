"""Train and evaluate a reproducible telecom churn prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
REPORTS_DIR = ROOT / "reports"


def load_data(path: Path) -> pd.DataFrame:
    """Load raw data and fix source-data types."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.drop_duplicates().copy()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def score_model(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def save_confusion_matrix(y_test: pd.Series, predictions, output: Path) -> None:
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    plt.title("Churn prediction confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    df = load_data(DATA_PATH)

    x = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    results, fitted = [], {}
    for name, classifier in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor(x)), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        results.append(score_model(name, pipeline, x_test, y_test))
        fitted[name] = pipeline

    best = max(results, key=lambda item: item["roc_auc"])
    best_pipeline = fitted[best["model"]]
    joblib.dump(best_pipeline, MODEL_PATH)
    best_predictions = best_pipeline.predict(x_test)
    save_confusion_matrix(y_test, best_predictions, REPORTS_DIR / "confusion_matrix.png")

    metrics = {"data_rows": len(df), "test_rows": len(x_test), "best_model": best["model"], "models": results}
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best model: {best['model']} | ROC-AUC: {best['roc_auc']}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
