"""Create churn-risk scores from a saved training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score telecom customers for churn risk.")
    parser.add_argument("--input", type=Path, required=True, help="CSV file with customer records")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "churn_pipeline.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    identifiers = df.get("customerID", pd.Series(range(len(df)), name="customerID"))
    features = df.drop(columns=[column for column in ["customerID", "Churn"] if column in df.columns])
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    model = joblib.load(args.model)
    probabilities = model.predict_proba(features)[:, 1]
    scores = pd.DataFrame({
        "customerID": identifiers,
        "churn_probability": probabilities.round(4),
        "risk_band": [risk_band(value) for value in probabilities],
    })
    scores["retention_priority"] = scores["risk_band"].map(
        {"High": "Contact immediately", "Medium": "Targeted offer", "Low": "Monitor"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"Saved {len(scores)} customer scores to {args.output}")


if __name__ == "__main__":
    main()
