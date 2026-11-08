from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================

MFT_CSV = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
    r"\Scenario_01_MFT.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
)

OUTPUT_FEATURES = (
    OUTPUT_DIR
    / "Scenario_01_SI_FN_features.csv"
)

OUTPUT_MISMATCH = (
    OUTPUT_DIR
    / "Scenario_01_SI_FN_mismatch.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "Scenario_01_SI_FN_summary.csv"
)


# ============================================================
# KONFIGURASI TIMESTAMP
# ============================================================

TIMESTAMP_PAIRS = {
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
# LOAD DATA
# ============================================================

def load_mft():

    print("=" * 90)
    print("LOAD MFT")
    print("=" * 90)

    df = pd.read_csv(
        MFT_CSV,
        low_memory=False
    )

    print(
        f"\nJumlah record : {len(df):,}"
    )

    print(
        f"Jumlah kolom  : {len(df.columns)}"
    )

    return df


# ============================================================
# KONVERSI TIMESTAMP
# ============================================================

def convert_timestamps(df):

    print("\n")
    print("=" * 90)
    print("KONVERSI TIMESTAMP")
    print("=" * 90)

    for timestamp_pair in TIMESTAMP_PAIRS.values():

        for col in timestamp_pair:

            if col not in df.columns:
                print(
                    f"Kolom tidak ditemukan: {col}"
                )
                continue

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            print(
                f"{col}: "
                f"{df[col].notna().sum():,} "
                f"nilai valid"
            )

    return df


# ============================================================
# HITUNG DISCREPANCY
# ============================================================

def calculate_features(df):

    print("\n")
    print("=" * 90)
    print("HITUNG SI ↔ FN DISCREPANCY")
    print("=" * 90)

    for label, (
        si_col,
        fn_col
    ) in TIMESTAMP_PAIRS.items():

        delta_col = (
            f"delta_{label}_seconds"
        )

        mismatch_col = (
            f"si_fn_{label}_mismatch"
        )

        si_valid = (
            df[si_col].notna()
        )

        fn_valid = (
            df[fn_col].notna()
        )

        both_valid = (
            si_valid
            & fn_valid
        )

        # Selisih SI - FN dalam detik
        df[delta_col] = np.nan

        df.loc[
            both_valid,
            delta_col
        ] = (
            (
                df.loc[
                    both_valid,
                    si_col
                ]
                -
                df.loc[
                    both_valid,
                    fn_col
                ]
            )
            .dt.total_seconds()
        )

        # Mismatch hanya dihitung jika
        # kedua timestamp tersedia.
        df[mismatch_col] = False

        df.loc[
            both_valid,
            mismatch_col
        ] = (
            df.loc[
                both_valid,
                delta_col
            ].abs() > 0
        )

        print(
            f"\n{label}:"
        )

        print(
            f"  SI valid : "
            f"{si_valid.sum():,}"
        )

        print(
            f"  FN valid : "
            f"{fn_valid.sum():,}"
        )

        print(
            f"  Keduanya : "
            f"{both_valid.sum():,}"
        )

        print(
            f"  Mismatch : "
            f"{df[mismatch_col].sum():,}"
        )

    # --------------------------------------------------------
    # JUMLAH MISMATCH
    # --------------------------------------------------------

    mismatch_columns = [
        f"si_fn_{label}_mismatch"
        for label in TIMESTAMP_PAIRS
    ]

    df[
        "si_fn_mismatch_count"
    ] = df[
        mismatch_columns
    ].sum(axis=1)

    df[
        "si_fn_any_mismatch"
    ] = (
        df[
            "si_fn_mismatch_count"
        ] > 0
    )

    return df


# ============================================================
# RINGKASAN DISTRIBUSI
# ============================================================

def print_delta_statistics(df):

    print("\n")
    print("=" * 90)
    print("STATISTIK DELTA")
    print("=" * 90)

    for label in TIMESTAMP_PAIRS:

        col = (
            f"delta_{label}_seconds"
        )

        valid = df[
            col
        ].dropna()

        print(
            f"\n{label}"
        )

        if valid.empty:

            print(
                "  Tidak ada pasangan timestamp valid."
            )

            continue

        print(
            f"  count : {len(valid):,}"
        )

        print(
            f"  min   : {valid.min():,.6f}"
        )

        print(
            f"  median: {valid.median():,.6f}"
        )

        print(
            f"  mean  : {valid.mean():,.6f}"
        )

        print(
            f"  max   : {valid.max():,.6f}"
        )

        print(
            f"  P95   : "
            f"{valid.quantile(0.95):,.6f}"
        )

        print(
            f"  P99   : "
            f"{valid.quantile(0.99):,.6f}"
        )


# ============================================================
# TARGET ANALYSIS
# ============================================================

def analyze_target(df):

    print("\n")
    print("=" * 90)
    print("ANALISIS TARGET")
    print("=" * 90)

    target = (
        "NewFileTime_SI_C_Manipulation.dll"
    )

    target_df = df[
        df["FileName"]
        .astype(str)
        .str.lower()
        .eq(target.lower())
    ].copy()

    print(
        f"\nTarget: {target}"
    )

    print(
        f"Jumlah record: "
        f"{len(target_df)}"
    )

    if target_df.empty:

        print(
            "Target tidak ditemukan."
        )

        return

    columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "SI<FN",
        "Created0x10",
        "Created0x30",
        "LastModified0x10",
        "LastModified0x30",
        "LastRecordChange0x10",
        "LastRecordChange0x30",
        "LastAccess0x10",
        "LastAccess0x30",
        "delta_creation_seconds",
        "delta_modified_seconds",
        "delta_record_change_seconds",
        "delta_access_seconds",
        "si_fn_creation_mismatch",
        "si_fn_modified_mismatch",
        "si_fn_record_change_mismatch",
        "si_fn_access_mismatch",
        "si_fn_mismatch_count",
        "si_fn_any_mismatch"
    ]

    columns = [
        col
        for col in columns
        if col in target_df.columns
    ]

    print("\nTarget detail:")

    print(
        target_df[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# TOP MISMATCH
# ============================================================

def analyze_mismatch(df):

    print("\n")
    print("=" * 90)
    print("TOP SI ↔ FN MISMATCH")
    print("=" * 90)

    mismatch_df = df[
        df["si_fn_any_mismatch"]
    ].copy()

    print(
        f"\nRecord dengan minimal "
        f"1 mismatch: "
        f"{len(mismatch_df):,}"
    )

    if mismatch_df.empty:

        print(
            "Tidak ada mismatch."
        )

        return mismatch_df

    # Nilai discrepancy terbesar
    mismatch_df[
        "max_abs_delta_seconds"
    ] = mismatch_df[
        [
            f"delta_{label}_seconds"
            for label in TIMESTAMP_PAIRS
        ]
    ].abs().max(
        axis=1
    )

    sort_columns = [
        "max_abs_delta_seconds"
    ]

    mismatch_df = mismatch_df.sort_values(
        sort_columns,
        ascending=False
    )

    display_columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "SI<FN",
        "Created0x10",
        "Created0x30",
        "LastModified0x10",
        "LastModified0x30",
        "LastRecordChange0x10",
        "LastRecordChange0x30",
        "LastAccess0x10",
        "LastAccess0x30",
        "delta_creation_seconds",
        "delta_modified_seconds",
        "delta_record_change_seconds",
        "delta_access_seconds",
        "si_fn_mismatch_count",
        "max_abs_delta_seconds"
    ]

    display_columns = [
        col
        for col in display_columns
        if col in mismatch_df.columns
    ]

    print("\nTop 30 discrepancy:")

    print(
        mismatch_df[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    return mismatch_df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(df):

    print("\n")
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    summary_rows = []

    total = len(df)

    for label, (
        si_col,
        fn_col
    ) in TIMESTAMP_PAIRS.items():

        both_valid = (
            df[si_col].notna()
            &
            df[fn_col].notna()
        )

        mismatch_col = (
            f"si_fn_{label}_mismatch"
        )

        mismatch_count = int(
            df[mismatch_col].sum()
        )

        summary_rows.append({
            "timestamp_type": label,
            "total_mft_records": total,
            "si_valid": int(
                df[si_col].notna().sum()
            ),
            "fn_valid": int(
                df[fn_col].notna().sum()
            ),
            "both_valid": int(
                both_valid.sum()
            ),
            "mismatch_count": mismatch_count,
            "mismatch_percentage_of_both_valid": (
                (
                    mismatch_count
                    /
                    both_valid.sum()
                    * 100
                )
                if both_valid.sum() > 0
                else 0
            )
        })

    # Overall
    summary_rows.append({
        "timestamp_type": "ANY",
        "total_mft_records": total,
        "si_valid": None,
        "fn_valid": None,
        "both_valid": None,
        "mismatch_count": int(
            df["si_fn_any_mismatch"].sum()
        ),
        "mismatch_percentage_of_both_valid": None
    })

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    df,
    mismatch_df
):

    print("\n")
    print("=" * 90)
    print("SAVE OUTPUT")
    print("=" * 90)

    # Dataset lengkap
    df.to_csv(
        OUTPUT_FEATURES,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nFeatures:"
    )
    print(OUTPUT_FEATURES)

    # Dataset mismatch
    mismatch_df.to_csv(
        OUTPUT_MISMATCH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nMismatch:"
    )
    print(OUTPUT_MISMATCH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SCRIPT 20 - SI ↔ FN TIMESTAMP ANALYSIS")
    print("=" * 90)

    # 1
    df = load_mft()

    # 2
    df = convert_timestamps(
        df
    )

    # 3
    df = calculate_features(
        df
    )

    # 4
    print_delta_statistics(
        df
    )

    # 5
    analyze_target(
        df
    )

    # 6
    mismatch_df = analyze_mismatch(
        df
    )

    # 7
    create_summary(
        df
    )

    # 8
    save_outputs(
        df,
        mismatch_df
    )

    print("\n")
    print("=" * 90)
    print("SCRIPT 20 SELESAI")
    print("=" * 90)


if __name__ == "__main__":
    main()