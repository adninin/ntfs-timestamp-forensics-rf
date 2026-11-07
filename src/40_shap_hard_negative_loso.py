from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


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
    / "shap_hard_negative_loso"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# RANDOM FOREST
# ============================================================

RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# EXCLUDED COLUMNS
# ============================================================

EXCLUDE_COLUMNS = {
    "label",
    "scenario",
    "file_id",
    "negative_type",
    "EntryNumber",
    "SequenceNumber",
    "hard_negative_distance",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SHAP — HARD NEGATIVE LOSO")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print()
print(f"Dataset shape: {df.shape}")


# ============================================================
# VALIDATION
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
# FEATURE SELECTION
# ============================================================

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
    print("Non-numeric features dibuang:")

    for c in non_numeric:
        print(" -", c)

    feature_columns = [
        c
        for c in feature_columns
        if c not in non_numeric
    ]


print()
print(
    f"SHAP candidate features: "
    f"{len(feature_columns)}"
)


# ============================================================
# DATA
# ============================================================

X = df[
    feature_columns
].copy()

y = df[
    "label"
].astype(int)

scenario = df[
    "scenario"
].astype(int)


# ============================================================
# STORAGE
# ============================================================

global_shap_rows = []

local_shap_rows = []

prediction_rows = []

fold_summary_rows = []


# ============================================================
# LOSO
# ============================================================

scenarios = sorted(
    scenario.unique()
)


print()
print("=" * 80)
print("SHAP LEAVE-ONE-SCENARIO-OUT")
print("=" * 80)


for test_scenario in scenarios:

    print()
    print("-" * 80)
    print(
        f"SCENARIO S{test_scenario:02d}"
    )
    print("-" * 80)

    train_mask = (
        scenario != test_scenario
    )

    test_mask = (
        scenario == test_scenario
    )

    X_train = X.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()


    # ========================================================
    # IMPUTATION
    # ========================================================

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

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train_imp,
        y_train
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test_imp
    )

    y_prob = model.predict_proba(
        X_test_imp
    )[:, 1]


    # ========================================================
    # SHAP EXPLAINER
    # ========================================================

    print("Creating TreeExplainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_result = explainer.shap_values(
        X_test_imp
    )


    # ========================================================
    # HANDLE SHAP OUTPUT FORMAT
    # ========================================================

    # SHAP versions differ:
    #
    # Binary classifier may return:
    #
    # 1. list with two arrays
    # 2. ndarray with shape:
    #    (samples, features, classes)
    #
    # We want class 1.

    if isinstance(
        shap_result,
        list
    ):

        if len(shap_result) == 2:
            shap_values = shap_result[1]
        else:
            shap_values = shap_result[0]

    else:

        shap_values = np.asarray(
            shap_result
        )

        if shap_values.ndim == 3:

            # samples x features x classes

            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values[:, :, 0]

        elif shap_values.ndim == 2:
            pass

        else:
            raise ValueError(
                "Unexpected SHAP shape: "
                f"{shap_values.shape}"
            )


    shap_values = np.asarray(
        shap_values
    )


    # ========================================================
    # VALIDATE SHAP SHAPE
    # ========================================================

    if shap_values.shape != X_test_imp.shape:

        raise ValueError(
            "SHAP shape mismatch.\n"
            f"SHAP: {shap_values.shape}\n"
            f"X_test: {X_test_imp.shape}"
        )


    # ========================================================
    # TEST INDICES
    # ========================================================

    test_indices = df.index[
        test_mask
    ]


    # ========================================================
    # LOCAL SHAP
    # ========================================================

    for row_position, original_index in enumerate(
        test_indices
    ):

        row_shap = shap_values[
            row_position
        ]

        row_meta = {
            "scenario": test_scenario,
            "file_id": df.loc[
                original_index,
                "file_id"
            ],
            "y_true": int(
                df.loc[
                    original_index,
                    "label"
                ]
            ),
            "y_pred": int(
                y_pred[row_position]
            ),
            "y_probability": float(
                y_prob[row_position]
            ),
            "negative_type": df.loc[
                original_index,
                "negative_type"
            ],
        }

        for feature, value in zip(
            feature_columns,
            row_shap
        ):

            local_shap_rows.append({
                **row_meta,
                "feature": feature,
                "shap_value": float(value),
                "abs_shap_value": float(
                    abs(value)
                ),
            })


    # ========================================================
    # GLOBAL SHAP PER FOLD
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    mean_shap = np.mean(
        shap_values,
        axis=0
    )


    for feature, mean_abs, mean_value in zip(
        feature_columns,
        mean_abs_shap,
        mean_shap
    ):

        global_shap_rows.append({
            "scenario": test_scenario,
            "feature": feature,
            "mean_abs_shap": float(
                mean_abs
            ),
            "mean_shap": float(
                mean_value
            ),
        })


    # ========================================================
    # PREDICTION STORAGE
    # ========================================================

    for row_position, original_index in enumerate(
        test_indices
    ):

        prediction_rows.append({
            "scenario": test_scenario,
            "file_id": df.loc[
                original_index,
                "file_id"
            ],
            "y_true": int(
                df.loc[
                    original_index,
                    "label"
                ]
            ),
            "y_pred": int(
                y_pred[row_position]
            ),
            "y_probability": float(
                y_prob[row_position]
            ),
            "negative_type": df.loc[
                original_index,
                "negative_type"
            ],
        })


    # ========================================================
    # FOLD SUMMARY
    # ========================================================

    positive_mask = (
        y_test.to_numpy() == 1
    )

    positive_positions = np.where(
        positive_mask
    )[0]

    if len(positive_positions) == 1:

        pos = positive_positions[0]

        target_probability = float(
            y_prob[pos]
        )

        target_prediction = int(
            y_pred[pos]
        )

    else:

        target_probability = np.nan
        target_prediction = np.nan


    fold_summary_rows.append({
        "scenario": test_scenario,
        "target_prediction": target_prediction,
        "target_probability": target_probability,
        "target_correct": (
            target_prediction == 1
            if not pd.isna(target_prediction)
            else False
        ),
        "n_test": len(X_test),
    })


    print(
        f"Test rows      : {len(X_test)}"
    )

    print(
        f"Target y_true  : 1"
    )

    print(
        f"Target y_pred  : {target_prediction}"
    )

    print(
        f"Target prob    : "
        f"{target_probability:.6f}"
    )


# ============================================================
# DATAFRAMES
# ============================================================

global_shap_df = pd.DataFrame(
    global_shap_rows
)

local_shap_df = pd.DataFrame(
    local_shap_rows
)

prediction_df = pd.DataFrame(
    prediction_rows
)

fold_summary_df = pd.DataFrame(
    fold_summary_rows
)


# ============================================================
# GLOBAL AGGREGATION
# ============================================================

global_summary = (
    global_shap_df
    .groupby("feature")
    .agg(
        mean_abs_shap=(
            "mean_abs_shap",
            "mean"
        ),
        mean_shap=(
            "mean_shap",
            "mean"
        ),
        folds=(
            "scenario",
            "nunique"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


# ============================================================
# TARGET-ONLY SHAP
# ============================================================

target_local = local_shap_df[
    local_shap_df["y_true"] == 1
].copy()


target_global = (
    target_local
    .groupby("feature")
    .agg(
        mean_abs_shap=(
            "abs_shap_value",
            "mean"
        ),
        mean_shap=(
            "shap_value",
            "mean"
        ),
        target_count=(
            "scenario",
            "count"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


# ============================================================
# MISCLASSIFIED TARGETS
# ============================================================

misclassified_targets = (
    target_local[
        target_local["y_pred"] == 0
    ]
    .copy()
)


misclassified_summary = (
    misclassified_targets
    .groupby("feature")
    .agg(
        mean_abs_shap=(
            "abs_shap_value",
            "mean"
        ),
        mean_shap=(
            "shap_value",
            "mean"
        ),
        count=(
            "scenario",
            "count"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


# ============================================================
# SAVE
# ============================================================

global_shap_file = (
    OUTPUT_DIR
    / "shap_global_per_fold.csv"
)

global_summary_file = (
    OUTPUT_DIR
    / "shap_global_summary.csv"
)

target_global_file = (
    OUTPUT_DIR
    / "shap_target_summary.csv"
)

local_shap_file = (
    OUTPUT_DIR
    / "shap_local_all_test_samples.csv"
)

prediction_file = (
    OUTPUT_DIR
    / "shap_loso_predictions.csv"
)

fold_summary_file = (
    OUTPUT_DIR
    / "shap_loso_fold_summary.csv"
)

misclassified_file = (
    OUTPUT_DIR
    / "shap_misclassified_target_summary.csv"
)


global_shap_df.to_csv(
    global_shap_file,
    index=False
)

global_summary.to_csv(
    global_summary_file,
    index=False
)

target_global.to_csv(
    target_global_file,
    index=False
)

local_shap_df.to_csv(
    local_shap_file,
    index=False
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

fold_summary_df.to_csv(
    fold_summary_file,
    index=False
)

misclassified_summary.to_csv(
    misclassified_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("SHAP FINAL SUMMARY")
print("=" * 80)

print()
print("Target predictions:")

print(
    fold_summary_df.to_string(
        index=False
    )
)


print()
print("-" * 80)
print("TOP 20 GLOBAL SHAP FEATURES")
print("-" * 80)

print(
    global_summary
    .head(20)
    .to_string(index=False)
)


print()
print("-" * 80)
print("TOP 20 TARGET-ONLY SHAP FEATURES")
print("-" * 80)

print(
    target_global
    .head(20)
    .to_string(index=False)
)


print()
print("-" * 80)
print("TOP FEATURES FOR MISCLASSIFIED TARGETS")
print("-" * 80)

if len(misclassified_summary) > 0:

    print(
        misclassified_summary
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "Tidak ada target yang salah diklasifikasikan."
    )


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(global_shap_file)
print(global_summary_file)
print(target_global_file)
print(local_shap_file)
print(prediction_file)
print(fold_summary_file)
print(misclassified_file)

print()
print("=" * 80)
print("DONE")
print("=" * 80)