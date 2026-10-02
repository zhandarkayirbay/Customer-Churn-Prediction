# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave the service. The project turns a raw customer table into an explainable churn-risk score that can support retention campaigns.

## Business goal

Customer acquisition costs more than retention. This project identifies customers with high churn risk so a telecom team can prioritise outreach, tailored offers, and service improvements.

## Dataset

The project uses IBM's public **Telco Customer Churn** sample: 7,043 customers, 21 columns, and a binary `Churn` target. Each row contains demographic, subscription, service, and billing information.

Source: [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data).

## What is included

- Data-quality cleaning for `TotalCharges` and duplicated records
- Exploratory analysis notebook
- Reproducible train/test split and preprocessing pipeline
- Logistic Regression and Random Forest model comparison
- ROC-AUC, accuracy, precision, recall, F1, confusion matrix, and classification report
- Saved best model for repeatable predictions
- Command-line prediction utility

## Project structure

```text
.
├── data/                         # IBM source data
├── notebooks/01_eda.ipynb        # exploratory data analysis
├── models/                       # generated model artifact (after training)
├── reports/                      # generated metrics and charts
├── src/train.py                  # model training pipeline
├── src/predict.py                # batch prediction utility
├── requirements.txt
└── README.md
```

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py src/train.py
```

The training script saves `models/churn_pipeline.joblib`, `reports/model_metrics.json`, and a confusion-matrix chart.

## Make predictions

```powershell
py src/predict.py --input data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output reports/customer_risk_scores.csv
```

The result contains a churn probability, a risk band, and a recommended retention priority for every customer.

## Notes

This is an educational portfolio project based on a fictional telecom dataset. Predictions should support, not replace, human business decisions.
