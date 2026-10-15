"""Regression checks for leakage, raw-input inference and CLI failures."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.predict import predict_customers, risk_band
from src.preprocessing import CustomerCleaner, NUMERIC_FEATURES
from src.train import DATA_PATH, ROOT, build_pipeline, load_data


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data(DATA_PATH).iloc[:200].copy()
        cls.model = build_pipeline(LogisticRegression(max_iter=1000))
        cls.model.fit(cls.frame, cls.frame['Churn'].map({'No': 0, 'Yes': 1}))

    def test_pipeline_excludes_identifiers_and_target(self):
        changed = self.frame.iloc[:3].copy()
        before = self.model.predict_proba(changed)
        changed['customerID'] = 'unrelated'
        changed['Churn'] = 'not a target'
        np.testing.assert_allclose(before, self.model.predict_proba(changed))

    def test_imputer_does_not_learn_from_inference(self):
        imputer = self.model['preprocessor'].named_transformers_['numeric']['imputer']
        before = imputer.statistics_.copy()
        cleaned = CustomerCleaner().transform(self.frame)
        np.testing.assert_allclose(before, cleaned[NUMERIC_FEATURES].median())
        unseen = self.frame.iloc[:2].copy()
        unseen['TotalCharges'] = [None, 1e9]
        self.model.predict_proba(unseen)
        np.testing.assert_array_equal(before, imputer.statistics_)

    def test_missing_values_and_unknown_categories(self):
        unseen = self.frame.iloc[:3].copy()
        unseen['TotalCharges'] = [' ', None, '123.4']
        unseen['PaymentMethod'] = ['new method', None, ' ']
        scores = predict_customers(unseen, self.model)
        self.assertEqual(len(scores), 3)
        self.assertTrue(scores.churn_probability.between(0, 1).all())
        self.assertFalse(scores.isna().any().any())

    def test_validation_rejects_bad_schema_and_values(self):
        cases = [self.frame.drop(columns='Contract'), self.frame.iloc[:0]]
        for value in ['broken', -1, np.inf]:
            bad = self.frame.copy()
            bad['TotalCharges'] = value
            cases.append(bad)
        for frame in cases:
            with self.subTest(columns=frame.columns.tolist(), rows=len(frame)):
                with self.assertRaises(ValueError):
                    self.model.predict_proba(frame)

    def test_risk_boundaries(self):
        self.assertEqual([risk_band(p) for p in [0, .3999, .4, .6999, .7, 1]],
                         ['Low', 'Low', 'Medium', 'Medium', 'High', 'High'])

    def test_no_identifier_or_target_is_required_for_inference(self):
        scores = predict_customers(self.frame.drop(columns=['customerID', 'Churn']), self.model)
        self.assertEqual(scores.customerID.tolist(), list(range(len(self.frame))))

    def test_invalid_training_target_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            for column, value in [('Churn', 'Maybe'), ('customerID', 'same')]:
                frame = self.frame.copy()
                frame[column] = value
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_serialized_cli_predicts_unlabelled_data(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path, input_path, output_path = [directory / n for n in
                                                  ['model.joblib', 'input.csv', 'out.csv']]
            joblib.dump(self.model, model_path)
            frame = self.frame.iloc[:4].drop(columns='Churn')
            frame.to_csv(input_path, index=False)
            command = [sys.executable, str(ROOT / 'src/predict.py'), '--model',
                       str(model_path), '--input', str(input_path), '--output', str(output_path)]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            scores = pd.read_csv(output_path)
            self.assertEqual(scores.customerID.tolist(), frame.customerID.tolist())
            np.testing.assert_allclose(scores.churn_probability,
                                       predict_customers(frame, self.model).churn_probability)
            # Invalid input must fail without replacing an existing result file.
            input_path.write_text('wrong\nvalue\n')
            previous = output_path.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Missing required columns', result.stderr)
            self.assertEqual(previous, output_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
