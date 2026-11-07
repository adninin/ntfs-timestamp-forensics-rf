from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================

MFT_CSV = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\\output\mft_scenario_01"
    r"\Scenario_01_MFT.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
)

OUTPUT_FEATURES = (
    OUTPUT_DIR
    / "Scenario_01_forensic_timestamp_features.csv"
)

OUTPUT_TARGET = (
    OUTPUT_DIR
    / "Scenario_01_target_forensic_features.csv"
)

OUTPUT_DISTRIBUTION = (
    OUTPUT_DIR
    / "Scenario_01_timestamp_distribution.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET = (
    "NewFileTime_SI_C_Manipulation.dll"
)


# ============================================================
# TIMESTAMP PAIRS
# ============================================================

PAIRS = {
    "creation": (
        "Created0x10",
        "Created0x30"
    ),

    "modified": (
        "LastModified0x10",
        "LastModified0x30"
    ),

    "record_change": (
        "LastRecordChange0x10",
        "LastRecordChange0x30"
    ),

    "access": (
        "LastAccess0x10",
        "LastAccess0x30"
    )
}


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 90)
    print("LOAD MFT")
    print("=" * 90)

    df = pd.read_csv(
        MFT_CSV,
        low_memory=False
    )

    print(
        f"\nRows: {len(df):,}"
    )

    return df


# ============================================================
# KONVERSI TIMESTAMP
# ============================================================

def convert_timestamps(df):

    print("\n")
    print("=" * 90)
    print("CONVERT TIMESTAMPS")
    print("=" * 90)

    for si_col, fn_col in PAIRS.values():

        df[si_col] = pd.to_datetime(
            df[si_col],
            errors="coerce"
        )

        df[fn_col] = pd.to_datetime(
            df[fn_col],
            errors="coerce"
        )

    return df


# ============================================================
# BUAT FITUR
# ============================================================

def create_features(df):

    print("\n")
    print("=" * 90)
    print("CREATE FORENSIC TIMESTAMP FEATURES")
    print("=" * 90)

    for label, (
        si_col,
        fn_col
    ) in PAIRS.items():

        delta = (
            f"delta_{label}_seconds"
        )

        abs_delta = (
            f"abs_delta_{label}_seconds"
        )

        year_delta = (
            f"year_delta_{label}"
        )

        exact = (
            f"si_fn_{label}_exact"
        )

        si_before = (
            f"si_before_fn_{label}"
        )

        si_after = (
            f"si_after_fn_{label}"
        )

        si_1980 = (
            f"si_{label}_is_1980"
        )

        fn_1980 = (
            f"fn_{label}_is_1980"
        )

        si_2000 = (
            f"si_{label}_is_2000"
        )

        fn_2000 = (
            f"fn_{label}_is_2000"
        )

        both = (
            df[si_col].notna()
            &
            df[fn_col].notna()
        )

        # ----------------------------------------------------
        # DELTA
        # ----------------------------------------------------

        df[delta] = np.nan

        df.loc[both, delta] = (
            (
                df.loc[both, si_col]
                -
                df.loc[both, fn_col]
            )
            .dt.total_seconds()
        )

        df[abs_delta] = (
            df[delta].abs()
        )

        # ----------------------------------------------------
        # YEAR DELTA
        # ----------------------------------------------------

        df[year_delta] = np.nan

        df.loc[both, year_delta] = (
            df.loc[both, si_col].dt.year
            -
            df.loc[both, fn_col].dt.year
        )

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        df[exact] = False

        df.loc[both, exact] = (
            df.loc[
                both,
                si_col
            ]
            ==
            df.loc[
                both,
                fn_col
            ]
        )

        # ----------------------------------------------------
        # ARAH PERBEDAAN
        # ----------------------------------------------------

        df[si_before] = False

        df[si_after] = False

        df.loc[both, si_before] = (
            df.loc[
                both,
                si_col
            ]
            <
            df.loc[
                both,
                fn_col
            ]
        )

        df.loc[both, si_after] = (
            df.loc[
                both,
                si_col
            ]
            >
            df.loc[
                both,
                fn_col
            ]
        )

        # ----------------------------------------------------
        # SPECIAL EPOCH YEAR
        # ----------------------------------------------------

        df[si_1980] = (
            df[si_col].dt.year == 1980
        )

        df[fn_1980] = (
            df[fn_col].dt.year == 1980
        )

        df[si_2000] = (
            df[si_col].dt.year == 2000
        )

        df[fn_2000] = (
            df[fn_col].dt.year == 2000
        )

    # ========================================================
    # GLOBAL FEATURES
    # ========================================================

    abs_delta_columns = [
        f"abs_delta_{label}_seconds"
        for label in PAIRS
    ]

    df[
        "max_abs_timestamp_delta"
    ] = df[
        abs_delta_columns
    ].max(axis=1)

    df[
        "mean_abs_timestamp_delta"
    ] = df[
        abs_delta_columns
    ].mean(axis=1)

    exact_columns = [
        f"si_fn_{label}_exact"
        for label in PAIRS
    ]

    df[
        "si_fn_exact_count"
    ] = df[
        exact_columns
    ].sum(axis=1)

    special_columns = []

    for label in PAIRS:

        special_columns.extend([
            f"si_{label}_is_1980",
            f"fn_{label}_is_1980",
            f"si_{label}_is_2000",
            f"fn_{label}_is_2000"
        ])

    df[
        "special_timestamp_count"
    ] = df[
        special_columns
    ].sum(axis=1)

    return df


# ============================================================
# DISTRIBUSI
# ============================================================

def analyze_distribution(df):

    print("\n")
    print("=" * 90)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 90)

    rows = []

    for label in PAIRS:

        col = (
            f"abs_delta_{label}_seconds"
        )

        valid = df[col].dropna()

        if valid.empty:
            continue

        row = {
            "timestamp_type": label,
            "count": len(valid),
            "median": valid.median(),
            "mean": valid.mean(),
            "p75": valid.quantile(0.75),
            "p90": valid.quantile(0.90),
            "p95": valid.quantile(0.95),
            "p99": valid.quantile(0.99),
            "p99_9": valid.quantile(0.999),
            "max": valid.max()
        }

        rows.append(row)

        print(
            f"\n{label}"
        )

        for key, value in row.items():

            if key == "timestamp_type":
                continue

            print(
                f"  {key}: "
                f"{value:,.6f}"
            )

    distribution = pd.DataFrame(
        rows
    )

    distribution.to_csv(
        OUTPUT_DISTRIBUTION,
        index=False,
        encoding="utf-8-sig"
    )

    return distribution


# ============================================================
# TARGET
# ============================================================

def analyze_target(df):

    print("\n")
    print("=" * 90)
    print("TARGET ANALYSIS")
    print("=" * 90)

    target = df[
        df["FileName"]
        .astype(str)
        .str.lower()
        .eq(TARGET.lower())
    ].copy()

    print(
        f"\nTarget rows: "
        f"{len(target)}"
    )

    if target.empty:
        return

    feature_columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "SI<FN"
    ]

    feature_columns += [
        col
        for col in df.columns
        if (
            col.startswith("delta_")
            or col.startswith("abs_delta_")
            or col.startswith("year_delta_")
            or col.startswith("si_fn_")
            or col.startswith("si_")
            or col.startswith("fn_")
            or col in [
                "max_abs_timestamp_delta",
                "mean_abs_timestamp_delta",
                "special_timestamp_count"
            ]
        )
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col in target.columns
    ]

    print("\nTarget features:")

    print(
        target[
            feature_columns
        ].to_string(
            index=False
        )
    )

    target[
        feature_columns
    ].to_csv(
        OUTPUT_TARGET,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nOutput target:"
    )
    print(OUTPUT_TARGET)


# ============================================================
# TOP RECORD
# ============================================================

def show_top_records(df):

    print("\n")
    print("=" * 90)
    print("TOP RECORD BERDASARKAN MAX ABS DELTA")
    print("=" * 90)

    columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "SI<FN",
        "max_abs_timestamp_delta",
        "mean_abs_timestamp_delta",
        "special_timestamp_count"
    ]

    columns = [
        col
        for col in columns
        if col in df.columns
    ]

    top = df.sort_values(
        "max_abs_timestamp_delta",
        ascending=False
    ).head(30)

    print(
        top[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_features(df):

    df.to_csv(
        OUTPUT_FEATURES,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 90)
    print("OUTPUT")
    print("=" * 90)

    print(
        f"\nFeatures:"
    )
    print(OUTPUT_FEATURES)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SCRIPT 21 - FORENSIC TIMESTAMP FEATURES")
    print("=" * 90)

    df = load_data()

    df = convert_timestamps(
        df
    )

    df = create_features(
        df
    )

    analyze_distribution(
        df
    )

    analyze_target(
        df
    )

    show_top_records(
        df
    )

    save_features(
        df
    )

    print("\n")
    print("=" * 90)
    print("SCRIPT 21 SELESAI")
    print("=" * 90)


if __name__ == "__main__":
    main()