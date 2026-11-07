from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "hard_negative_usn_integrated"
    / "hard_negative_usn_integrated_all_v2.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "rf_hard_negative_loso"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RANDOM FOREST CONFIG
# ============================================================

RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("RANDOM FOREST — HARD NEGATIVE LOSO")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print()
print(f"Dataset shape: {df.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "scenario",
    "file_id",
    "label",
    "negative_type",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {col}"
        )

if df["label"].value_counts().get(1, 0) != 12:
    raise ValueError(
        "Jumlah positive harus 12."
    )

if df["label"].value_counts().get(0, 0) != 1200:
    raise ValueError(
        "Jumlah negative harus 1200."
    )

if df.duplicated(
    subset=["scenario", "file_id"]
).any():
    raise ValueError(
        "Terdapat duplicate (scenario, file_id)."
    )


# ============================================================
# FEATURE EXCLUSION
# ============================================================

EXCLUDE_COLUMNS = {
    # Target / metadata
    "label",
    "scenario",
    "file_id",
    "negative_type",

    # Identifier-like
    "EntryNumber",
    "SequenceNumber",

    # Selection-only variable.
    # TIDAK BOLEH menjadi feature RF.
    "hard_negative_distance",
}


feature_columns = [
    c
    for c in df.columns
    if c not in EXCLUDE_COLUMNS
]


# ------------------------------------------------------------
# Numeric only
# ------------------------------------------------------------

non_numeric = [
    c
    for c in feature_columns
    if not pd.api.types.is_numeric_dtype(
        df[c]
    )
]

if non_numeric:
    print()
    print("Non-numeric features yang dibuang:")
    for c in non_numeric:
        print(" -", c)

    feature_columns = [
        c
        for c in feature_columns
        if c not in non_numeric
    ]


print()
print(
    f"Candidate features: "
    f"{len(feature_columns)}"
)


# ============================================================
# PREPARE X / y
# ============================================================

X = df[feature_columns].copy()
y = df["label"].astype(int)
scenario = df["scenario"].astype(int)


# ============================================================
# STORAGE
# ============================================================

fold_results = []
prediction_rows = []
importance_rows = []


# ============================================================
# LOSO
# ============================================================

scenarios = sorted(
    scenario.unique()
)

print()
print("=" * 80)
print("LEAVE-ONE-SCENARIO-OUT")
print("=" * 80)


for test_scenario in scenarios:

    print()
    print("-" * 80)
    print(
        f"TEST SCENARIO: S{test_scenario:02d}"
    )
    print("-" * 80)

    train_mask = (
        scenario != test_scenario
    )

    test_mask = (
        scenario == test_scenario
    )

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    # --------------------------------------------------------
    # Fold information
    # --------------------------------------------------------

    print(
        f"Train: {len(X_train):,} "
        f"(positive={int(y_train.sum())}, "
        f"negative={int((y_train == 0).sum())})"
    )

    print(
        f"Test : {len(X_test):,} "
        f"(positive={int(y_test.sum())}, "
        f"negative={int((y_test == 0).sum())})"
    )

    # --------------------------------------------------------
    # Median imputation
    #
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )

    assert X_train_imp.shape[1] == len(feature_columns)
    assert X_test_imp.shape[1] == len(feature_columns)

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train_imp,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_imp
    )

    y_prob = model.predict_proba(
        X_test_imp
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )
    except ValueError:
        roc_auc = np.nan

    # PR-AUC
    try:
        pr_auc = average_precision_score(
            y_test,
            y_prob
        )
    except ValueError:
        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall   : {recall:.4f}"
    )
    print(
        f"F1       : {f1:.4f}"
    )
    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )
    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print()
    print(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    # --------------------------------------------------------
    # Fold result
    # --------------------------------------------------------

    fold_results.append({
        "scenario": test_scenario,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_positive": int(y_train.sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    test_indices = df.index[
        test_mask
    ]

    for idx, pred, prob in zip(
        test_indices,
        y_pred,
        y_prob
    ):

        prediction_rows.append({
            "scenario": test_scenario,
            "file_id": df.loc[idx, "file_id"],
            "y_true": int(
                df.loc[idx, "label"]
            ),
            "y_pred": int(pred),
            "y_probability": float(prob),
            "negative_type": df.loc[
                idx,
                "negative_type"
            ],
        })

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    for feature, importance in zip(
        feature_columns,
        model.feature_importances_
    ):

        importance_rows.append({
            "scenario": test_scenario,
            "feature": feature,
            "importance": float(
                importance
            ),
        })


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

predictions_df = pd.DataFrame(
    prediction_rows
)

importance_df = pd.DataFrame(
    importance_rows
)


# ============================================================
# SUMMARY
# ============================================================

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

summary = {
    "dataset": "HARD_NEGATIVE_USN_INTEGRATED",
    "n_rows": len(df),
    "n_features": len(feature_columns),
}

for metric in metric_columns:
    summary[
        f"mean_{metric}"
    ] = results_df[metric].mean()

summary_df = pd.DataFrame(
    [summary]
)


# ============================================================
# AGGREGATE CONFUSION MATRIX
# ============================================================

total_tn = results_df["tn"].sum()
total_fp = results_df["fp"].sum()
total_fn = results_df["fn"].sum()
total_tp = results_df["tp"].sum()

aggregate_metrics = {
    "TN": total_tn,
    "FP": total_fp,
    "FN": total_fn,
    "TP": total_tp,
}


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

global_importance = (
    importance_df
    .groupby("feature")["importance"]
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

global_importance.columns = [
    "feature",
    "mean_importance",
]


# ============================================================
# SAVE
# ============================================================

results_file = (
    OUTPUT_DIR
    / "hard_negative_loso_fold_results.csv"
)

predictions_file = (
    OUTPUT_DIR
    / "hard_negative_loso_predictions.csv"
)

importance_file = (
    OUTPUT_DIR
    / "hard_negative_loso_feature_importance.csv"
)

global_importance_file = (
    OUTPUT_DIR
    / "hard_negative_loso_global_importance.csv"
)

summary_file = (
    OUTPUT_DIR
    / "hard_negative_loso_summary.csv"
)


results_df.to_csv(
    results_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)

importance_df.to_csv(
    importance_file,
    index=False
)

global_importance.to_csv(
    global_importance_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL HARD-NEGATIVE LOSO RESULTS")
print("=" * 80)

print()
print(results_df.to_string(
    index=False
))

print()
print("-" * 80)
print("MEAN METRICS")
print("-" * 80)

for metric in metric_columns:
    print(
        f"{metric:10s}: "
        f"{results_df[metric].mean():.6f}"
    )

print()
print("-" * 80)
print("AGGREGATE CONFUSION MATRIX")
print("-" * 80)

print(
    f"TN={total_tn}, "
    f"FP={total_fp}, "
    f"FN={total_fn}, "
    f"TP={total_tp}"
)

print()
print("-" * 80)
print("TOP 20 FEATURES")
print("-" * 80)

print(
    global_importance
    .head(20)
    .to_string(index=False)
)

print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(results_file)
print(predictions_file)
print(importance_file)
print(global_importance_file)
print(summary_file)

print()
print("=" * 80)
print("DONE")
print("=" * 80)