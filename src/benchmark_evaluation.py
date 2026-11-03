import sys
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Ensure project root is in sys.path regardless of how the script is executed
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_collection import fetch_page, extract_features
from src.preprocessing import calculate_quality_score

# 2. Benchmark Query Dimensions
LANGUAGES = ["Go", "Rust"]
STAR_RANGES = ["0..10", "11..100", "101..1000", ">1000"]
DATE_WINDOWS = [
    "pushed:>2026-08-01",              # Active: < 90 days
    "pushed:2025-11-01..2026-07-31",   # Recent: 3-12 months
    "pushed:2023-11-01..2025-10-31",   # Stale: 1-3 years
    "pushed:<2023-11-01"               # Dormant: > 3 years
]


def main():
    # 1. Load trained models
    rf = joblib.load(ROOT_DIR / "models" / "random_forest_v1.pkl")
    log_reg = joblib.load(ROOT_DIR / "models" / "logistic_regression_v1.pkl")
    scaler = joblib.load(ROOT_DIR / "models" / "scaler_v1.pkl")
    expected_features = list(scaler.feature_names_in_)

    # 2. Collect benchmark data (32 queries x 15 repos = ~480 repos)
    print("Collecting benchmark dataset (Go & Rust across 4 recency windows)...")
    records = []
    for lang in LANGUAGES:
        for stars in STAR_RANGES:
            for date_win in DATE_WINDOWS:
                query = f"language:{lang} stars:{stars} {date_win}"
                items = fetch_page(query=query, page=1, per_page=15)
                records.extend([extract_features(r) for r in items])
                time.sleep(0.3)

    df = pd.DataFrame(records).drop_duplicates(subset=["full_name"]).reset_index(drop=True)
    print(f"Fetched {len(df)} unique repositories.\n")

    # 3. Ground truth labeling & feature preparation
    df = calculate_quality_score(df, threshold=42.5)
    df["owner_type_org"] = (df["owner_type"] == "Organization").astype(int)
    for col in ["has_license", "has_wiki", "has_pages", "has_projects", "has_discussions"]:
        df[col] = df[col].astype(int)

    # Set unseen training languages to 0
    df["lang_JavaScript"] = 0
    df["lang_Python"] = 0
    df["lang_TypeScript"] = 0

    X = df[expected_features]
    y = df["is_high_quality"]
    X_scaled = scaler.transform(X)

    # 4. Predict
    preds = {
        "Random Forest": (rf.predict(X), rf.predict_proba(X)[:, 1]),
        "Logistic Regression": (log_reg.predict(X_scaled), log_reg.predict_proba(X_scaled)[:, 1])
    }

    # 5. Overall Metrics
    metrics = []
    for name, (y_pred, y_prob) in preds.items():
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1-Score": f1_score(y, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, y_prob)
        })

    print("=== OVERALL BENCHMARK PERFORMANCE ===")
    print(pd.DataFrame(metrics).round(4).to_string(index=False))
    print()

    # Confusion matrices
    for name, (y_pred, _) in preds.items():
        cm = confusion_matrix(y, y_pred)
        print(f"{name} Confusion Matrix:\n[[TN={cm[0,0]}  FP={cm[0,1]}]\n [FN={cm[1,0]}  TP={cm[1,1]}]]\n")

    # 6. Slice Analysis by Language
    slices = []
    for lang in LANGUAGES:
        mask = df["language"] == lang
        slices.append({
            "Language": lang,
            "Count": int(mask.sum()),
            "RF Accuracy": accuracy_score(y[mask], preds["Random Forest"][0][mask]),
            "LR Accuracy": accuracy_score(y[mask], preds["Logistic Regression"][0][mask])
        })

    print("=== PERFORMANCE BY LANGUAGE ===")
    print(pd.DataFrame(slices).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
