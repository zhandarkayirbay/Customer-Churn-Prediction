"""Train candidates, select by training-only CV ROC-AUC, evaluate a holdout."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import CustomerCleaner, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
SEED = 42


def load_data(path):
    """Reject ambiguous targets/IDs and remove exact duplicate source records."""
    frame = pd.read_csv(path).drop_duplicates().copy()
    CustomerCleaner().fit(frame)
    if 'Churn' not in frame or not frame['Churn'].isin(['No', 'Yes']).all():
        raise ValueError('Training requires a Churn column containing only No/Yes.')
    if frame['Churn'].nunique() != 2:
        raise ValueError('Training requires both churn classes.')
    if 'customerID' in frame and (
        frame['customerID'].isna().any() or frame['customerID'].duplicated().any()
    ):
        raise ValueError('customerID must be unique and nonmissing in training data.')
    return frame


def build_pipeline(classifier):
    return Pipeline([
        ('cleaner', CustomerCleaner()),
        ('preprocessor', build_preprocessor()),
        ('classifier', classifier),
    ])


def score_model(name, pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        'model': name,
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions, zero_division=0)),
        'recall': float(recall_score(y_test, predictions, zero_division=0)),
        'f1': float(f1_score(y_test, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probabilities)),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'classification_report': classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }


def train(data_path=DATA_PATH, model_path=None, reports_dir=None, jobs=2):
    model_path = Path(model_path or ROOT / 'models/churn_pipeline.joblib')
    reports_dir = Path(reports_dir or ROOT / 'reports')
    figures = reports_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    x = frame.drop(columns=['customerID', 'Churn'], errors='ignore')
    y = frame['Churn'].map({'No': 0, 'Yes': 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    candidates = {
        'logistic_regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=SEED
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight='balanced',
            random_state=SEED, n_jobs=jobs,
        ),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fitted, cv_scores = {}, {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        # Sequential folds avoid nested thread oversubscription in the forest.
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv,
                                 scoring='roc_auc', n_jobs=1, error_score='raise')
        cv_scores[name] = scores
        fitted[name] = pipeline.fit(x_train, y_train)
    # Select before examining test metrics. Never tune against the holdout.
    best_name = max(cv_scores, key=lambda name: cv_scores[name].mean())
    results = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pipeline in fitted.items():
        result = score_model(name, pipeline, x_test, y_test)
        result['cv_roc_auc_mean'] = float(cv_scores[name].mean())
        result['cv_roc_auc_std'] = float(cv_scores[name].std())
        result['cv_roc_auc_folds'] = cv_scores[name].tolist()
        results.append(result)
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test, ax=ax, name=name)
        cm_fig, cm_ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(
            pipeline, x_test, y_test, ax=cm_ax,
            display_labels=['Stayed', 'Churned'], colorbar=False,
        )
        cm_ax.set_title(name.replace('_', ' ').title())
        cm_fig.tight_layout()
        cm_fig.savefig(figures / f'confusion_matrix_{name}.png', dpi=160)
        if name == best_name:
            cm_fig.savefig(reports_dir / 'confusion_matrix.png', dpi=160)
        plt.close(cm_fig)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
    ax.set_title('Holdout ROC curves (20% stratified split)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'roc_curves.png', dpi=160)
    plt.close(fig)
    joblib.dump(fitted[best_name], model_path)
    # Validate serialization immediately, using raw input and the same cleaner.
    restored = joblib.load(model_path)
    np.testing.assert_allclose(
        restored.predict_proba(x_test), fitted[best_name].predict_proba(x_test)
    )
    metrics = {
        'data_rows': len(frame), 'train_rows': len(x_train), 'test_rows': len(x_test),
        'random_seed': SEED, 'test_size': 0.2, 'classification_threshold': 0.5,
        'selection_criterion': 'Highest mean ROC-AUC in 3-fold stratified CV on training only',
        'best_model': best_name,
        'majority_baseline_accuracy': float((y_test == 0).mean()),
        'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                     'joblib': joblib.__version__},
        'models': results,
    }
    (reports_dir / 'model_metrics.json').write_text(
        json.dumps(metrics, indent=2), encoding='utf-8'
    )
    print(f'Selected by training-only CV: {best_name}')
    for result in results:
        print(f"{result['model']}: CV ROC-AUC={result['cv_roc_auc_mean']:.4f}, "
              f"test ROC-AUC={result['roc_auc']:.4f}, F1={result['f1']:.4f}")
    print(f'Saved and reloaded model: {model_path}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/churn_pipeline.joblib')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    try:
        train(args.data, args.model, args.reports, args.jobs)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'Training failed: {exc}\n')


if __name__ == '__main__':
    main()
