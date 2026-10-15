"""Shared schema validation and preprocessing fitted only on training data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class CustomerCleaner(TransformerMixin, BaseEstimator):
    """Validate schema and normalize values without learning from the data.

    Missing values and blank charges are imputed by the fitted pipeline.
    Nonblank malformed numeric values are rejected rather than silently scored.
    Extra columns (including customerID and Churn) never become model features.
    """

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Input must be a pandas DataFrame with named columns.')
        missing = sorted(set(FEATURES) - set(X.columns))
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(missing))
        if X.empty:
            raise ValueError('Input contains no customer rows.')
        cleaned = X[FEATURES].copy()
        for column in NUMERIC_FEATURES:
            values = cleaned[column].replace(r'^\s*$', np.nan, regex=True)
            numeric = pd.to_numeric(values, errors='coerce')
            if (values.notna() & numeric.isna()).any():
                raise ValueError(f'{column} contains invalid numeric values.')
            if ((numeric < 0) | np.isinf(numeric)).any():
                raise ValueError(f'{column} must contain finite nonnegative values.')
            cleaned[column] = numeric.astype(float)
        if not cleaned['SeniorCitizen'].dropna().isin([0, 1]).all():
            raise ValueError('SeniorCitizen must be 0 or 1.')
        for column in CATEGORICAL_FEATURES:
            values = cleaned[column].astype('string').str.strip()
            cleaned[column] = values.replace('', pd.NA).astype(object)
            cleaned[column] = cleaned[column].where(values.notna() & values.ne(''), np.nan)
        return cleaned


def build_preprocessor():
    """Impute, scale and encode using statistics learned during pipeline.fit."""
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('numeric', numeric, NUMERIC_FEATURES),
        ('categorical', categorical, CATEGORICAL_FEATURES),
    ])
