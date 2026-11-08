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
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

DATA_DIR = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "rf_baseline_loso"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MFT_PATH = (
    DATA_DIR
    / "dataset_mft_only.csv"
)

MFT_USN_PATH = (
    DATA_DIR
    / "dataset_mft_usn.csv"
)

RANDOM_SEED = 42


# ============================================================
# NON-FEATURE COLUMNS
# ============================================================

META_COLUMNS = [
    "scenario",
    "file_id",
    "label",
    "negative_type",
]


# ============================================================
# RANDOM FOREST CONFIG
# ============================================================

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
    "max_features": "sqrt",
}


# ============================================================
# HELPER
# ============================================================

def load_dataset(path):

    print()
    print("=" * 80)
    print(f"LOADING: {path.name}")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        "Labels:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    return df


def get_feature_columns(df):

    feature_columns = [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]

    # Hanya numeric
    numeric_features = []

    for col in feature_columns:

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            numeric_features.append(col)

    return numeric_features


def evaluate_fold(
    y_true,
    y_pred,
    y_score,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    # ROC-AUC membutuhkan kedua kelas
    if len(
        np.unique(y_true)
    ) == 2:

        result["roc_auc"] = (
            roc_auc_score(
                y_true,
                y_score
            )
        )

        result["pr_auc"] = (
            average_precision_score(
                y_true,
                y_score
            )
        )

    else:

        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def run_loso(
    df,
    dataset_name,
):

    print()
    print("#" * 80)
    print(f"DATASET: {dataset_name}")
    print("#" * 80)

    feature_columns = get_feature_columns(
        df
    )

    print()
    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print()

    print(
        "Feature list:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    scenarios = sorted(
        df["scenario"]
        .unique()
    )

    print()
    print(
        f"Scenarios: {scenarios}"
    )

    fold_results = []
    prediction_results = []
    feature_importance_results = []

    # ========================================================
    # LEAVE-ONE-SCENARIO-OUT
    # ========================================================

    for test_scenario in scenarios:

        print()
        print("-" * 80)
        print(
            f"TEST SCENARIO: "
            f"{test_scenario}"
        )
        print("-" * 80)

        train_mask = (
            df["scenario"]
            != test_scenario
        )

        test_mask = (
            df["scenario"]
            == test_scenario
        )

        train_df = (
            df.loc[train_mask]
            .copy()
        )

        test_df = (
            df.loc[test_mask]
            .copy()
        )

        print(
            f"Train rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Test rows : "
            f"{len(test_df):,}"
        )

        print(
            "Train label:"
        )

        print(
            train_df["label"]
            .value_counts()
            .sort_index()
        )

        print(
            "Test label:"
        )

        print(
            test_df["label"]
            .value_counts()
            .sort_index()
        )

        # ----------------------------------------------------
        # X / y
        # ----------------------------------------------------

        X_train = (
            train_df[feature_columns]
            .copy()
        )

        y_train = (
            train_df["label"]
            .astype(int)
            .copy()
        )

        X_test = (
            test_df[feature_columns]
            .copy()
        )

        y_test = (
            test_df["label"]
            .astype(int)
            .copy()
        )

        # ----------------------------------------------------
        # IMPUTATION
        #
        # Fit imputer hanya pada training fold.
        # Ini penting untuk mencegah leakage.
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = (
            imputer.fit_transform(
                X_train
            )
        )

        X_test_imp = (
            imputer.transform(
                X_test
            )
        )

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        model = RandomForestClassifier(
            **RF_PARAMS
        )

        model.fit(
            X_train_imp,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test_imp
        )

        y_score = (
            model.predict_proba(
                X_test_imp
            )[:, 1]
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = evaluate_fold(
            y_test,
            y_pred,
            y_score
        )

        metrics["dataset"] = (
            dataset_name
        )

        metrics["test_scenario"] = (
            test_scenario
        )

        metrics["train_rows"] = (
            len(train_df)
        )

        metrics["test_rows"] = (
            len(test_df)
        )

        metrics["train_positive"] = int(
            (y_train == 1).sum()
        )

        metrics["test_positive"] = int(
            (y_test == 1).sum()
        )

        fold_results.append(
            metrics
        )

        # ----------------------------------------------------
        # Print fold result
        # ----------------------------------------------------

        print()
        print(
            "Fold result:"
        )

        print(
            f"  TN        : {metrics['tn']}"
        )

        print(
            f"  FP        : {metrics['fp']}"
        )

        print(
            f"  FN        : {metrics['fn']}"
        )

        print(
            f"  TP        : {metrics['tp']}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"  ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC    : "
            f"{metrics['pr_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Prediction detail
        # ----------------------------------------------------

        fold_predictions = test_df[
            [
                "scenario",
                "file_id",
                "label",
                "negative_type",
            ]
        ].copy()

        fold_predictions[
            "predicted_label"
        ] = y_pred

        fold_predictions[
            "manipulation_probability"
        ] = y_score

        fold_predictions[
            "correct"
        ] = (
            fold_predictions[
                "label"
            ]
            ==
            fold_predictions[
                "predicted_label"
            ]
        )

        prediction_results.append(
            fold_predictions
        )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
            "dataset": dataset_name,
            "test_scenario": test_scenario,
        })

        feature_importance_results.append(
            importance
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.concat(
        prediction_results,
        ignore_index=True
    )

    importance_df = pd.concat(
        feature_importance_results,
        ignore_index=True
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / f"{dataset_name}_fold_results.csv"
    )

    results_df.to_csv(
        fold_path,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_path = (
        OUTPUT_DIR
        / f"{dataset_name}_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    # ========================================================
    # SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_path = (
        OUTPUT_DIR
        / f"{dataset_name}_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # ========================================================
    # AGGREGATED FEATURE IMPORTANCE
    # ========================================================

    mean_importance = (
        importance_df
        .groupby(
            "feature"
        )["importance"]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
        .sort_values(
            "mean",
            ascending=False
        )
    )

    mean_importance_path = (
        OUTPUT_DIR
        / f"{dataset_name}_feature_importance_summary.csv"
    )

    mean_importance.to_csv(
        mean_importance_path,
        index=False
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary = {
        "dataset": dataset_name,
        "folds": len(results_df),
        "features": len(feature_columns),
        "total_test_records": len(
            predictions_df
        ),
        "positive_test_records": int(
            predictions_df["label"]
            .sum()
        ),
        "mean_accuracy": (
            results_df["accuracy"]
            .mean()
        ),
        "std_accuracy": (
            results_df["accuracy"]
            .std()
        ),
        "mean_precision": (
            results_df["precision"]
            .mean()
        ),
        "std_precision": (
            results_df["precision"]
            .std()
        ),
        "mean_recall": (
            results_df["recall"]
            .mean()
        ),
        "std_recall": (
            results_df["recall"]
            .std()
        ),
        "mean_f1": (
            results_df["f1"]
            .mean()
        ),
        "std_f1": (
            results_df["f1"]
            .std()
        ),
        "mean_roc_auc": (
            results_df["roc_auc"]
            .mean()
        ),
        "std_roc_auc": (
            results_df["roc_auc"]
            .std()
        ),
        "mean_pr_auc": (
            results_df["pr_auc"]
            .mean()
        ),
        "std_pr_auc": (
            results_df["pr_auc"]
            .std()
        ),
        "total_tp": int(
            results_df["tp"].sum()
        ),
        "total_fp": int(
            results_df["fp"].sum()
        ),
        "total_fn": int(
            results_df["fn"].sum()
        ),
        "total_tn": int(
            results_df["tn"].sum()
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        OUTPUT_DIR
        / f"{dataset_name}_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("#" * 80)
    print(
        f"SUMMARY - {dataset_name}"
    )
    print("#" * 80)

    print()
    print(
        f"Mean Accuracy : "
        f"{summary['mean_accuracy']:.4f}"
    )

    print(
        f"Mean Precision: "
        f"{summary['mean_precision']:.4f}"
    )

    print(
        f"Mean Recall   : "
        f"{summary['mean_recall']:.4f}"
    )

    print(
        f"Mean F1       : "
        f"{summary['mean_f1']:.4f}"
    )

    print(
        f"Mean ROC-AUC  : "
        f"{summary['mean_roc_auc']:.4f}"
    )

    print(
        f"Mean PR-AUC   : "
        f"{summary['mean_pr_auc']:.4f}"
    )

    print()
    print(
        "Aggregate confusion matrix:"
    )

    print(
        f"  TN = {summary['total_tn']}"
    )

    print(
        f"  FP = {summary['total_fp']}"
    )

    print(
        f"  FN = {summary['total_fn']}"
    )

    print(
        f"  TP = {summary['total_tp']}"
    )

    print()
    print(
        "Top 15 features:"
    )

    print(
        mean_importance
        .head(15)
        .to_string(
            index=False
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "SCRIPT 36 - RANDOM FOREST BASELINE "
        "WITH LEAVE-ONE-SCENARIO-OUT"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    mft_df = load_dataset(
        MFT_PATH
    )

    mft_usn_df = load_dataset(
        MFT_USN_PATH
    )

    # --------------------------------------------------------
    # Run MFT-only
    # --------------------------------------------------------

    mft_summary = run_loso(
        mft_df,
        "MFT_ONLY"
    )

    # --------------------------------------------------------
    # Run MFT + USN
    # --------------------------------------------------------

    mft_usn_summary = run_loso(
        mft_usn_df,
        "MFT_USN"
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison = pd.DataFrame([
        mft_summary,
        mft_usn_summary,
    ])

    comparison_path = (
        OUTPUT_DIR
        / "baseline_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    print(
        comparison[
            [
                "dataset",
                "features",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "mean_roc_auc",
                "mean_pr_auc",
                "total_tp",
                "total_fp",
                "total_fn",
                "total_tn",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"[SAVE] Comparison:"
        f"\n       {comparison_path}"
    )

    print()
    print("=" * 80)
    print("SCRIPT 36 SELESAI")
    print("=" * 80)

    print()
    print(
        "Random Forest baseline "
        "MFT-only dan MFT+USN "
        "sudah selesai."
    )

    print()
    print(
        "JANGAN tuning atau SHAP dulu."
    )

    print(
        "Kirim output terminal lengkap "
        "bagian SUMMARY dan BASELINE COMPARISON."
    )


if __name__ == "__main__":
    main()