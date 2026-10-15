"""Score new customer CSVs with the complete saved preprocessing pipeline."""

import argparse
from pathlib import Path
import pickle
import sys

# Support both `python src/predict.py` and `python -m src.predict`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def risk_band(probability):
    """Illustrative operational bands; thresholds are not calibrated promises."""
    if probability >= 0.70:
        return 'High'
    if probability >= 0.40:
        return 'Medium'
    return 'Low'


def predict_customers(frame, model):
    """Preserve identifiers and apply the training pipeline unchanged."""
    probabilities = model.predict_proba(frame)[:, 1]
    identifiers = frame.get(
        'customerID', pd.Series(range(len(frame)), index=frame.index)
    )
    scores = pd.DataFrame({
        'customerID': identifiers.to_numpy(),
        'churn_probability': probabilities.round(4),
        'risk_band': [risk_band(value) for value in probabilities],
    })
    scores['retention_priority'] = scores['risk_band'].map({
        'High': 'Contact immediately', 'Medium': 'Targeted offer', 'Low': 'Monitor',
    })
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'models/churn_pipeline.joblib')
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError('Input and output paths must be different.')
        if args.model.resolve() == args.output.resolve():
            raise ValueError('Output must not overwrite the model.')
        # joblib uses pickle: load only artifacts from a trusted source.
        model = joblib.load(args.model)
        if not callable(getattr(model, 'predict_proba', None)):
            raise ValueError('Model artifact must provide predict_proba; train it first.')
        frame = pd.read_csv(args.input, dtype={'customerID': 'string'})
        scores = predict_customers(frame, model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output, index=False)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        parser.exit(2, f'Prediction failed: {exc}\n')
    print(f'Saved {len(scores)} customer scores to {args.output}')


if __name__ == '__main__':
    main()
