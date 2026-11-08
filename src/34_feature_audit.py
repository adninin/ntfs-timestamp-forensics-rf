from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_PATH = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
    / "all_scenarios_ml_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METADATA / NON-ML COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {
    "scenario",
    "file_id",
    "label",
    "negative_type",
    "target_filename",
}


# ============================================================
# HELPER
# ============================================================

def is_identifier_like(column_name):
    """
    Identifikasi kolom yang secara nama terlihat seperti ID.
    Ini hanya flag audit, bukan otomatis dibuang.
    """

    name = column_name.lower()

    keywords = [
        "id",
        "entrynumber",
        "sequencenumber",
        "parententry",
        "parentsequence",
        "updatesecuence",
        "updatesquence",
        "sourcefile",
        "filename",
        "name",
        "path",
    ]

    return any(
        keyword in name
        for keyword in keywords
    )


def safe_numeric(series):
    """
    Convert numeric-like series ke numeric.
    """
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SCRIPT 34 - FEATURE AUDIT & LEAKAGE CHECK")
    print("=" * 80)

    print()
    print(f"Input : {INPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    print("[LOAD]")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print()

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    required_columns = [
        "scenario",
        "file_id",
        "label",
        "negative_type",
    ]

    missing_required = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: "
            f"{missing_required}"
        )

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    print("=" * 80)
    print("1. LABEL DISTRIBUTION")
    print("=" * 80)

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Label per scenario:")

    label_per_scenario = (
        df
        .groupby(
            ["scenario", "label"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        label_per_scenario
    )

    print()

    # --------------------------------------------------------
    # FEATURE CANDIDATES
    # --------------------------------------------------------

    feature_candidates = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    print("=" * 80)
    print("2. FEATURE CANDIDATES")
    print("=" * 80)

    print(
        f"Total candidate features: "
        f"{len(feature_candidates)}"
    )

    print()

    # --------------------------------------------------------
    # AUDIT TABLE
    # --------------------------------------------------------

    audit_rows = []

    for col in feature_candidates:

        series = df[col]

        dtype = str(series.dtype)

        missing_count = int(
            series.isna().sum()
        )

        missing_pct = (
            missing_count
            / len(df)
            * 100
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        unique_ratio = (
            unique_count
            / len(df)
        )

        constant = (
            unique_count <= 1
        )

        identifier_flag = (
            is_identifier_like(col)
        )

        numeric_series = safe_numeric(
            series
        )

        numeric_possible = (
            numeric_series.notna().sum()
            > 0
        )

        # ----------------------------------------------------
        # Label relationship
        # ----------------------------------------------------

        label_mean_0 = np.nan
        label_mean_1 = np.nan
        mean_difference = np.nan

        if numeric_possible:

            temp = pd.DataFrame({
                "x": numeric_series,
                "label": df["label"]
            })

            mean_by_label = (
                temp
                .groupby("label")["x"]
                .mean()
            )

            if 0 in mean_by_label.index:
                label_mean_0 = (
                    mean_by_label.loc[0]
                )

            if 1 in mean_by_label.index:
                label_mean_1 = (
                    mean_by_label.loc[1]
                )

            if (
                pd.notna(label_mean_0)
                and pd.notna(label_mean_1)
            ):
                mean_difference = (
                    label_mean_1
                    - label_mean_0
                )

        # ----------------------------------------------------
        # Correlation with label
        # ----------------------------------------------------

        correlation = np.nan

        if numeric_possible:

            corr_df = pd.DataFrame({
                "x": numeric_series,
                "label": df["label"]
            }).dropna()

            if (
                len(corr_df) > 1
                and corr_df["x"].nunique() > 1
            ):
                correlation = (
                    corr_df["x"]
                    .corr(
                        corr_df["label"]
                    )
                )

        # ----------------------------------------------------
        # Target-only value
        #
        # Karena positive hanya 12 rows,
        # kita cek apakah fitur punya nilai unik
        # khusus positive.
        # ----------------------------------------------------

        positive_values = (
            df.loc[
                df["label"] == 1,
                col
            ]
        )

        negative_values = (
            df.loc[
                df["label"] == 0,
                col
            ]
        )

        positive_unique = int(
            positive_values.nunique(
                dropna=True
            )
        )

        negative_unique = int(
            negative_values.nunique(
                dropna=True
            )
        )

        # ----------------------------------------------------
        # Numeric variance
        # ----------------------------------------------------

        std_value = np.nan

        if numeric_possible:

            std_value = (
                numeric_series.std()
            )

        # ----------------------------------------------------
        # Suspicious flags
        # ----------------------------------------------------

        flags = []

        if constant:
            flags.append(
                "CONSTANT"
            )

        if missing_pct >= 50:
            flags.append(
                "HIGH_MISSING"
            )

        if identifier_flag:
            flags.append(
                "IDENTIFIER_LIKE"
            )

        # Extremely high uniqueness can indicate ID-like data
        if (
            unique_ratio >= 0.90
            and unique_count > 20
        ):
            flags.append(
                "HIGH_CARDINALITY"
            )

        # A feature existing only in positive class
        if (
            positive_unique > 0
            and negative_unique == 0
        ):
            flags.append(
                "POSITIVE_ONLY"
            )

        audit_rows.append({
            "column": col,
            "dtype": dtype,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "constant": constant,
            "identifier_like": identifier_flag,
            "numeric_possible": numeric_possible,
            "std": std_value,
            "mean_label0": label_mean_0,
            "mean_label1": label_mean_1,
            "mean_difference": mean_difference,
            "label_correlation": correlation,
            "positive_unique": positive_unique,
            "negative_unique": negative_unique,
            "flags": "|".join(flags),
        })

    audit_df = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # SAVE FULL AUDIT
    # --------------------------------------------------------

    audit_path = (
        OUTPUT_DIR
        / "feature_audit_full.csv"
    )

    audit_df.to_csv(
        audit_path,
        index=False
    )

    print()
    print(
        f"[SAVE] Full feature audit:"
        f"\n       {audit_path}"
    )

    # ========================================================
    # 3. CONSTANT FEATURES
    # ========================================================

    print()
    print("=" * 80)
    print("3. CONSTANT FEATURES")
    print("=" * 80)

    constant_df = audit_df[
        audit_df["constant"]
    ].copy()

    if constant_df.empty:
        print("Tidak ada constant feature.")
    else:
        print(
            constant_df[
                [
                    "column",
                    "unique_count",
                    "missing_pct",
                ]
            ].to_string(
                index=False
            )
        )

    constant_path = (
        OUTPUT_DIR
        / "feature_audit_constant.csv"
    )

    constant_df.to_csv(
        constant_path,
        index=False
    )

    # ========================================================
    # 4. HIGH MISSING
    # ========================================================

    print()
    print("=" * 80)
    print("4. HIGH MISSING FEATURES")
    print("=" * 80)

    high_missing_df = audit_df[
        audit_df["missing_pct"] >= 50
    ].copy()

    if high_missing_df.empty:
        print(
            "Tidak ada feature dengan "
            "missing >= 50%."
        )
    else:
        print(
            high_missing_df[
                [
                    "column",
                    "missing_count",
                    "missing_pct",
                ]
            ]
            .sort_values(
                "missing_pct",
                ascending=False
            )
            .to_string(
                index=False
            )
        )

    high_missing_path = (
        OUTPUT_DIR
        / "feature_audit_high_missing.csv"
    )

    high_missing_df.to_csv(
        high_missing_path,
        index=False
    )

    # ========================================================
    # 5. IDENTIFIER-LIKE
    # ========================================================

    print()
    print("=" * 80)
    print("5. IDENTIFIER-LIKE FEATURES")
    print("=" * 80)

    identifier_df = audit_df[
        audit_df["identifier_like"]
    ].copy()

    print(
        identifier_df[
            [
                "column",
                "dtype",
                "unique_count",
                "unique_ratio",
                "flags",
            ]
        ]
        .sort_values(
            "unique_ratio",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    identifier_path = (
        OUTPUT_DIR
        / "feature_audit_identifier_like.csv"
    )

    identifier_df.to_csv(
        identifier_path,
        index=False
    )

    # ========================================================
    # 6. HIGH CORRELATION
    # ========================================================

    print()
    print("=" * 80)
    print("6. HIGHEST ABSOLUTE LABEL CORRELATION")
    print("=" * 80)

    corr_df = (
        audit_df[
            audit_df["label_correlation"]
            .notna()
        ]
        .copy()
    )

    corr_df["abs_correlation"] = (
        corr_df["label_correlation"]
        .abs()
    )

    corr_df = corr_df.sort_values(
        "abs_correlation",
        ascending=False
    )

    print(
        corr_df[
            [
                "column",
                "label_correlation",
                "mean_label0",
                "mean_label1",
                "flags",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    correlation_path = (
        OUTPUT_DIR
        / "feature_audit_correlation.csv"
    )

    corr_df.to_csv(
        correlation_path,
        index=False
    )

    # ========================================================
    # 7. POSITIVE-ONLY FEATURES
    # ========================================================

    print()
    print("=" * 80)
    print("7. POSITIVE-ONLY FEATURES")
    print("=" * 80)

    positive_only_df = audit_df[
        audit_df["flags"]
        .str.contains(
            "POSITIVE_ONLY",
            na=False
        )
    ].copy()

    if positive_only_df.empty:
        print(
            "Tidak ditemukan feature "
            "yang hanya memiliki nilai pada positive."
        )
    else:
        print(
            positive_only_df[
                [
                    "column",
                    "positive_unique",
                    "negative_unique",
                    "label_correlation",
                ]
            ]
            .to_string(
                index=False
            )
        )

    positive_only_path = (
        OUTPUT_DIR
        / "feature_audit_positive_only.csv"
    )

    positive_only_df.to_csv(
        positive_only_path,
        index=False
    )

    # ========================================================
    # 8. NUMERIC / NON-NUMERIC
    # ========================================================

    print()
    print("=" * 80)
    print("8. DATA TYPE SUMMARY")
    print("=" * 80)

    print(
        audit_df[
            "numeric_possible"
        ]
        .value_counts()
    )

    print()

    print(
        "Dtype:"
    )

    print(
        audit_df[
            "dtype"
        ]
        .value_counts()
    )

    # ========================================================
    # 9. PROPOSED SAFE FEATURE SET
    # ========================================================

    print()
    print("=" * 80)
    print("9. PROPOSED SAFE FEATURE SET")
    print("=" * 80)

    # --------------------------------------------------------
    # Jangan otomatis menggunakan semua identifier-like.
    #
    # Safe set awal:
    # - numeric
    # - bukan constant
    # - missing < 50%
    # - bukan identifier-like
    # - bukan positive-only
    #
    # Ini masih candidate set.
    # Final feature set akan kita review.
    # --------------------------------------------------------

    safe_mask = (
        (audit_df["numeric_possible"])
        & (~audit_df["constant"])
        & (audit_df["missing_pct"] < 50)
        & (~audit_df["identifier_like"])
        & (
            ~audit_df["flags"]
            .str.contains(
                "POSITIVE_ONLY",
                na=False
            )
        )
    )

    safe_df = audit_df[
        safe_mask
    ].copy()

    safe_df = safe_df.sort_values(
        "column"
    )

    print(
        f"Candidate safe features: "
        f"{len(safe_df)}"
    )

    print()

    print(
        safe_df[
            [
                "column",
                "dtype",
                "missing_pct",
                "unique_count",
                "label_correlation",
            ]
        ].to_string(
            index=False
        )
    )

    safe_path = (
        OUTPUT_DIR
        / "feature_audit_safe_candidates.csv"
    )

    safe_df.to_csv(
        safe_path,
        index=False
    )

    # ========================================================
    # 10. FINAL AUDIT SUMMARY
    # ========================================================

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "candidate_features": len(feature_candidates),
        "constant_features": len(constant_df),
        "high_missing_features": len(
            high_missing_df
        ),
        "identifier_like_features": len(
            identifier_df
        ),
        "positive_only_features": len(
            positive_only_df
        ),
        "safe_candidate_features": len(
            safe_df
        ),
        "positive_records": int(
            (df["label"] == 1).sum()
        ),
        "negative_records": int(
            (df["label"] == 0).sum()
        ),
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_audit_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 80)
    print("SCRIPT 34 SELESAI")
    print("=" * 80)

    print()
    print("Output:")
    print(f"1. {audit_path}")
    print(f"2. {constant_path}")
    print(f"3. {high_missing_path}")
    print(f"4. {identifier_path}")
    print(f"5. {correlation_path}")
    print(f"6. {positive_only_path}")
    print(f"7. {safe_path}")
    print(f"8. {summary_path}")

    print()
    print(
        "JANGAN training Random Forest dulu."
    )

    print(
        "Kirim output terminal bagian "
        "'3' sampai '9' untuk kita review "
        "bersama sebelum finalisasi fitur."
    )


if __name__ == "__main__":
    main()