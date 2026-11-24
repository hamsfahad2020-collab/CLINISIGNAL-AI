"""Train and evaluate a baseline model on the project's synthetic ICU dataset.

All metrics are software-demo metrics on artificial rule-generated labels only.
They are not estimates of clinical performance.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_icu.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
FEATURES = [
    "heart_rate_bpm", "map_mmhg", "resp_rate_min", "temperature_c",
    "spo2_pct", "wbc_10e9_l", "lactate_mmol_l", "creatinine_mg_dl",
    "platelets_10e9_l",
]
TARGET = "risk_event_within_6h"


def split_patients(df: pd.DataFrame):
    # Stratify patients by whether they have at least one positive row.
    patient_summary = df.groupby("patient_id")[TARGET].max().rename("has_positive")
    ids = patient_summary.index.to_numpy()
    strata = patient_summary.to_numpy()
    train_ids, temp_ids = train_test_split(ids, test_size=0.36, random_state=42, stratify=strata)
    temp_strata = patient_summary.loc[temp_ids].to_numpy()
    val_ids, test_ids = train_test_split(temp_ids, test_size=(5/9), random_state=42, stratify=temp_strata)
    return set(train_ids), set(val_ids), set(test_ids)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}. Run python src/generate_data.py")
    df = pd.read_csv(DATA)
    train_ids, val_ids, test_ids = split_patients(df)
    train = df[df.patient_id.isin(train_ids)]
    val = df[df.patient_id.isin(val_ids)]
    test = df[df.patient_id.isin(test_ids)]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # Choose a decision threshold using validation data only (maximize F1).
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    val_f1 = [f1_score(y_val, val_probs >= t, zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(val_f1))])

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    metrics = {
        "model": "Logistic Regression",
        "threshold": threshold,
        "patient_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "row_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_positive_labels": int(y_test.sum()),
        "test_negative_labels": int((y_test == 0).sum()),
        "sensitivity": float(recall_score(y_test, test_pred, zero_division=0)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)) if y_test.nunique() == 2 else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel().tolist(),
        "warning": "Synthetic rule-generated labels only. Not clinical evidence or real-world sepsis performance.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "threshold": threshold}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
