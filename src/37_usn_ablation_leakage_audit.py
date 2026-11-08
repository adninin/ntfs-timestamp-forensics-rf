from pathlib import Path
import pandas as pd
import numpy as np

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

ROOT = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

DATASET = (
    ROOT
    / "output"
    / "ml_dataset_all"
    / "all_scenarios_ml_dataset.csv"
)

OUTDIR = ROOT / "output" / "rf_usn_ablation_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Metadata / identifier columns yang TIDAK boleh menjadi fitur
META_COLS = {
    "scenario",
    "file_id",
    "label",
    "negative_type",
    "source_file",
}

# Identifier-like columns yang juga harus dikeluarkan
IDENTIFIER_TERMS = [
    "entrynumber",
    "sequencenumber",
]

# Kandidat USN features
USN_PREFIX = "usn_"

# Presence flag
HAS_USN = "has_usn"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("SCRIPT 37 - USN ABLATION + LEAKAGE AUDIT")
print("=" * 80)

print(f"[INFO] Dataset: {DATASET}")

df = pd.read_csv(DATASET)

print(f"[INFO] Shape: {df.shape}")
print(f"[INFO] Labels:\n{df['label'].value_counts(dropna=False)}")

# ============================================================
# IDENTIFY FEATURE SETS
# ============================================================

candidate_cols = [
    c for c in df.columns
    if c not in META_COLS
]

# Buang identifier-like
identifier_cols = [
    c for c in candidate_cols
    if any(term in c.lower() for term in IDENTIFIER_TERMS)
]

candidate_cols = [
    c for c in candidate_cols
    if c not in identifier_cols
]

# hanya numeric
numeric_cols = [
    c for c in candidate_cols
    if pd.api.types.is_numeric_dtype(df[c])
]

usn_cols = [
    c for c in numeric_cols
    if c.lower().startswith(USN_PREFIX)
]

# has_usn diperlakukan khusus
if HAS_USN in numeric_cols:
    usn_context_cols = usn_cols + [HAS_USN]
else:
    usn_context_cols = usn_cols.copy()

usn_context_cols = sorted(set(usn_context_cols))

mft_cols = [
    c for c in numeric_cols
    if c not in usn_context_cols
]

# MFT+USN
mft_usn_cols = sorted(set(mft_cols + usn_context_cols))

# MFT+USN tanpa has_usn
mft_usn_no_presence_cols = [
    c for c in mft_usn_cols
    if c != HAS_USN
]

# USN only
usn_only_cols = usn_context_cols.copy()

# USN only tanpa has_usn
usn_only_no_presence_cols = [
    c for c in usn_only_cols
    if c != HAS_USN
]


print("\n" + "=" * 80)
print("FEATURE INVENTORY")
print("=" * 80)

print(f"MFT features              : {len(mft_cols)}")
print(f"USN context features      : {len(usn_context_cols)}")
print(f"MFT + USN                 : {len(mft_usn_cols)}")
print(f"MFT + USN - has_usn       : {len(mft_usn_no_presence_cols)}")
print(f"USN only                  : {len(usn_only_cols)}")
print(f"USN only - has_usn        : {len(usn_only_no_presence_cols)}")

print("\nUSN features:")
for c in usn_context_cols:
    print("  ", c)


# ============================================================
# SAVE FEATURE INVENTORY
# ============================================================

inventory_rows = []

for group_name, cols in [
    ("MFT_ONLY", mft_cols),
    ("USN_CONTEXT", usn_context_cols),
    ("MFT_USN", mft_usn_cols),
    ("MFT_USN_NO_HAS_USN", mft_usn_no_presence_cols),
    ("USN_ONLY", usn_only_cols),
    ("USN_ONLY_NO_HAS_USN", usn_only_no_presence_cols),
]:
    for c in cols:
        inventory_rows.append({
            "group": group_name,
            "feature": c,
        })

pd.DataFrame(inventory_rows).to_csv(
    OUTDIR / "feature_inventory_audit.csv",
    index=False
)


# ============================================================
# DISTRIBUTION AUDIT
# ============================================================

print("\n" + "=" * 80)
print("TARGET VS BACKGROUND DISTRIBUTION AUDIT")
print("=" * 80)

audit_cols = [
    c for c in [
        HAS_USN,
        "usn_event_count",
        "usn_basic_info_count",
        "usn_close_count",
        "usn_data_extend_count",
        "usn_file_create_count",
        "usn_sequence_span",
        "usn_time_span_seconds",
        "usn_basic_info_ratio",
        "usn_close_ratio",
        "usn_data_extend_ratio",
        "usn_file_create_ratio",
    ]
    if c in df.columns
]

audit_rows = []

for col in audit_cols:

    for label_value, label_name in [
        (0, "background"),
        (1, "target"),
    ]:

        s = pd.to_numeric(
            df.loc[df["label"] == label_value, col],
            errors="coerce"
        )

        audit_rows.append({
            "feature": col,
            "class": label_name,
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
        })

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    OUTDIR / "usn_distribution_audit.csv",
    index=False
)

print(audit_df.to_string(index=False))


# ============================================================
# HAS_USN CROSSTAB
# ============================================================

if HAS_USN in df.columns:

    print("\n" + "=" * 80)
    print("HAS_USN CROSSTAB")
    print("=" * 80)

    crosstab = pd.crosstab(
        df["label"],
        df[HAS_USN],
        margins=True
    )

    print(crosstab)

    crosstab.to_csv(
        OUTDIR / "has_usn_crosstab.csv"
    )

    # Distribusi proporsi per kelas
    crosstab_pct = pd.crosstab(
        df["label"],
        df[HAS_USN],
        normalize="index"
    )

    print("\nProporsi:")
    print(crosstab_pct)

    crosstab_pct.to_csv(
        OUTDIR / "has_usn_crosstab_percentage.csv"
    )


# ============================================================
# LOSO EVALUATION FUNCTION
# ============================================================

def evaluate_loso(data, feature_cols, experiment_name):

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)

    rows = []
    predictions = []
    importances = []

    scenarios = sorted(data["scenario"].unique())

    for scenario in scenarios:

        train = data[data["scenario"] != scenario].copy()
        test = data[data["scenario"] == scenario].copy()

        X_train = train[feature_cols].copy()
        X_test = test[feature_cols].copy()

        y_train = train["label"].astype(int)
        y_test = test["label"].astype(int)

        # ----------------------------------------------------
        # Median imputation FIT ONLY on training fold
        # ----------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train_imp, y_train)

        pred = model.predict(X_test_imp)
        prob = model.predict_proba(X_test_imp)[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_test, pred)

        precision = precision_score(
            y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        # ROC/PR can technically fail if only one class exists.
        # Each test fold in this experiment contains 1 positive,
        # so both classes should exist.
        try:
            roc_auc = roc_auc_score(y_test, prob)
        except Exception:
            roc_auc = np.nan

        try:
            pr_auc = average_precision_score(y_test, prob)
        except Exception:
            pr_auc = np.nan

        rows.append({
            "experiment": experiment_name,
            "scenario": scenario,
            "n_features": len(feature_cols),
            "n_train": len(train),
            "n_test": len(test),
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

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        for i, (_, row) in enumerate(test.iterrows()):

            predictions.append({
                "experiment": experiment_name,
                "scenario": scenario,
                "file_id": row["file_id"],
                "label": int(row["label"]),
                "prediction": int(pred[i]),
                "probability": float(prob[i]),
                "negative_type": row.get(
                    "negative_type",
                    ""
                ),
            })

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        for feature, importance in zip(
            feature_cols,
            model.feature_importances_
        ):

            importances.append({
                "experiment": experiment_name,
                "scenario": scenario,
                "feature": feature,
                "importance": float(importance),
            })

        print(
            f"{scenario:>3} | "
            f"Acc={accuracy:.4f} "
            f"Prec={precision:.4f} "
            f"Rec={recall:.4f} "
            f"F1={f1:.4f} "
            f"ROC={roc_auc:.4f} "
            f"PR={pr_auc:.4f} "
            f"| TN={tn} FP={fp} FN={fn} TP={tp}"
        )

    result_df = pd.DataFrame(rows)

    prediction_df = pd.DataFrame(predictions)

    importance_df = pd.DataFrame(importances)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": experiment_name,
        "n_features": len(feature_cols),
        "mean_accuracy": result_df["accuracy"].mean(),
        "mean_precision": result_df["precision"].mean(),
        "mean_recall": result_df["recall"].mean(),
        "mean_f1": result_df["f1"].mean(),
        "mean_roc_auc": result_df["roc_auc"].mean(),
        "mean_pr_auc": result_df["pr_auc"].mean(),
        "TN": result_df["tn"].sum(),
        "FP": result_df["fp"].sum(),
        "FN": result_df["fn"].sum(),
        "TP": result_df["tp"].sum(),
    }

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return (
        result_df,
        prediction_df,
        importance_df,
        summary,
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

experiments = [
    ("MFT_ONLY", mft_cols),

    ("MFT_USN", mft_usn_cols),

    ("MFT_USN_NO_HAS_USN", mft_usn_no_presence_cols),

    ("USN_ONLY", usn_only_cols),

    ("USN_ONLY_NO_HAS_USN", usn_only_no_presence_cols),
]

all_results = []
all_predictions = []
all_importances = []
all_summaries = []

for experiment_name, feature_cols in experiments:

    if len(feature_cols) == 0:
        print(
            f"[WARNING] Skip {experiment_name}: "
            "no features."
        )
        continue

    (
        result_df,
        prediction_df,
        importance_df,
        summary,
    ) = evaluate_loso(
        df,
        feature_cols,
        experiment_name
    )

    all_results.append(result_df)
    all_predictions.append(prediction_df)
    all_importances.append(importance_df)
    all_summaries.append(summary)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.concat(
    all_results,
    ignore_index=True
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

importances_df = pd.concat(
    all_importances,
    ignore_index=True
)

summary_df = pd.DataFrame(all_summaries)

results_df.to_csv(
    OUTDIR / "ablation_fold_results.csv",
    index=False
)

predictions_df.to_csv(
    OUTDIR / "ablation_predictions.csv",
    index=False
)

importances_df.to_csv(
    OUTDIR / "ablation_feature_importance.csv",
    index=False
)

summary_df.to_csv(
    OUTDIR / "ablation_summary.csv",
    index=False
)


# ============================================================
# TOP FEATURES PER EXPERIMENT
# ============================================================

top_features = (
    importances_df
    .groupby(
        ["experiment", "feature"],
        as_index=False
    )["importance"]
    .mean()
)

top_features = top_features.sort_values(
    ["experiment", "importance"],
    ascending=[True, False]
)

top_features["rank"] = (
    top_features
    .groupby("experiment")["importance"]
    .rank(
        method="first",
        ascending=False
    )
)

top_features = top_features[
    top_features["rank"] <= 20
]

top_features.to_csv(
    OUTDIR / "top20_features_by_experiment.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL ABLATION SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOutput directory:")
print(OUTDIR)

print("\nFiles:")
for p in sorted(OUTDIR.iterdir()):
    print(" ", p.name)

print("\nDONE.")