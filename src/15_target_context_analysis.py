from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch\output\target_context"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGET FILE
# ============================================================

TARGETS = {
    "01": "NewFileTime_SI_C_Manipulation.dll",
    "02": "NewFileTime_SI_M_Manipulation.dll",
    "03": "NewFileTime_SI_MAC_Manipulation.dll",

    "04": "PowerShell_SI_C_Manipulation.dll",
    "05": "PowerShell_SI_M_Manipulation.dll",
    "06": "PowerShell_SI_MAC_Manipulation.dll",

    "07": "nTimestomp_SI_C_Manipulation.dll",
    "08": "nTimestomp_SI_M_Manipulation.dll",
    "09": "nTimestomp_SI_MACE_Manipulation.dll",

    "10": "FileSystemTunneling_SI_C_Change.dll",

    "11": "SetMACE_SI_MACE_Copy_Manipulation.dll",
    "12": "SetMACE_SI_FN_MACE_Manipulation.dll",
}


# ============================================================
# FUNGSI UTILITAS
# ============================================================

def add_file_id(df):
    """
    Membuat identitas record berbasis:
    EntryNumber + SequenceNumber
    """

    df["file_id"] = (
        df["EntryNumber"].astype(str)
        + "-"
        + df["SequenceNumber"].astype(str)
    )

    return df


def add_event_flags(df):
    """
    Membuat flag untuk jenis event USN.
    """

    reason = df["UpdateReasons"].fillna("").astype(str)

    df["has_basic_info_change"] = (
        reason.str.contains(
            "BasicInfoChange",
            regex=False
        )
    )

    df["has_close"] = (
        reason.str.contains(
            "Close",
            regex=False
        )
    )

    df["has_file_create"] = (
        reason.str.contains(
            "FileCreate",
            regex=False
        )
    )

    df["has_file_delete"] = (
        reason.str.contains(
            "FileDelete",
            regex=False
        )
    )

    df["has_rename"] = (
        reason.str.contains(
            "Rename",
            regex=False
        )
    )

    df["has_data_extend"] = (
        reason.str.contains(
            "DataExtend",
            regex=False
        )
    )

    df["has_data_overwrite"] = (
        reason.str.contains(
            "DataOverwrite",
            regex=False
        )
    )

    df["has_security_change"] = (
        reason.str.contains(
            "SecurityChange",
            regex=False
        )
    )

    return df


def add_sequence_context(df):
    """
    Mengurutkan event secara global berdasarkan USN,
    kemudian membuat konteks previous/next dalam setiap file_id.
    """

    # --------------------------------------------------------
    # Sort berdasarkan USN
    # --------------------------------------------------------

    df = df.sort_values(
        "UpdateSequenceNumber"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Event order dalam setiap file_id
    # --------------------------------------------------------

    df["event_order"] = (
        df.groupby("file_id")
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Previous event
    # --------------------------------------------------------

    df["previous_reason"] = (
        df.groupby("file_id")["UpdateReasons"]
        .shift(1)
    )

    df["previous_usn"] = (
        df.groupby("file_id")["UpdateSequenceNumber"]
        .shift(1)
    )

    df["previous_timestamp"] = (
        df.groupby("file_id")["UpdateTimestamp"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Next event
    # --------------------------------------------------------

    df["next_reason"] = (
        df.groupby("file_id")["UpdateReasons"]
        .shift(-1)
    )

    df["next_usn"] = (
        df.groupby("file_id")["UpdateSequenceNumber"]
        .shift(-1)
    )

    df["next_timestamp"] = (
        df.groupby("file_id")["UpdateTimestamp"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # USN delta
    # --------------------------------------------------------

    df["usn_delta"] = (
        df["UpdateSequenceNumber"]
        - df["previous_usn"]
    )

    # --------------------------------------------------------
    # Timestamp delta
    # --------------------------------------------------------

    df["UpdateTimestamp"] = pd.to_datetime(
        df["UpdateTimestamp"],
        errors="coerce"
    )

    df["previous_timestamp"] = pd.to_datetime(
        df["previous_timestamp"],
        errors="coerce"
    )

    df["next_timestamp"] = pd.to_datetime(
        df["next_timestamp"],
        errors="coerce"
    )

    df["time_delta_seconds"] = (
        df["UpdateTimestamp"]
        - df["previous_timestamp"]
    ).dt.total_seconds()

    df["next_time_delta_seconds"] = (
        df["next_timestamp"]
        - df["UpdateTimestamp"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Reason berubah / tidak
    # --------------------------------------------------------

    df["reason_changed"] = (
        df["UpdateReasons"].fillna("")
        != df["previous_reason"].fillna("")
    )

    return df


# ============================================================
# ANALISIS SATU SKENARIO
# ============================================================

def analyze_scenario(
    scenario,
    target_name
):

    print("\n")
    print("=" * 90)
    print(f"SCENARIO {scenario}")
    print(f"TARGET   : {target_name}")
    print("=" * 90)

    csv_path = (
        BASE_DIR
        / f"Scenario_{scenario}.csv"
    )

    if not csv_path.exists():

        print(
            f"ERROR: File tidak ditemukan: {csv_path}"
        )

        return None

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    print(
        f"Total rows dataset : {len(df):,}"
    )

    # --------------------------------------------------------
    # File ID
    # --------------------------------------------------------

    df = add_file_id(df)

    # --------------------------------------------------------
    # Cari target
    # --------------------------------------------------------

    target_mask = (
        df["Name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(target_name.lower())
    )

    target_rows = df.loc[
        target_mask
    ].copy()

    if target_rows.empty:

        print("TARGET TIDAK DITEMUKAN.")

        return None

    target_file_ids = (
        target_rows["file_id"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Target rows      : {len(target_rows):,}"
    )

    print(
        f"Target file_id   : {len(target_file_ids):,}"
    )

    print(
        f"File IDs         : {target_file_ids}"
    )

    # --------------------------------------------------------
    # Ambil SELURUH event dari file_id target
    # --------------------------------------------------------

    context_df = df[
        df["file_id"].isin(
            target_file_ids
        )
    ].copy()

    print(
        f"Total event konteks : "
        f"{len(context_df):,}"
    )

    # --------------------------------------------------------
    # Event flags
    # --------------------------------------------------------

    context_df = add_event_flags(
        context_df
    )

    # --------------------------------------------------------
    # Sequence context
    # --------------------------------------------------------

    context_df = add_sequence_context(
        context_df
    )

    # --------------------------------------------------------
    # Target marker
    # --------------------------------------------------------

    context_df["is_target_name"] = (
        context_df["Name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(target_name.lower())
    )

    # --------------------------------------------------------
    # Posisi relatif terhadap target
    # --------------------------------------------------------

    context_df["target_event_type"] = np.where(
        context_df["is_target_name"],
        "TARGET",
        "CONTEXT"
    )

    # --------------------------------------------------------
    # Simpan seluruh konteks
    # --------------------------------------------------------

    detail_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_target_context.csv"
    )

    context_df.to_csv(
        detail_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nDetail disimpan:"
    )
    print(detail_file)

    # ========================================================
    # SUMMARY PER FILE ID
    # ========================================================

    file_summary = (
        context_df
        .groupby("file_id")
        .agg(
            target_name=("Name", "first"),

            total_events=(
                "UpdateReasons",
                "count"
            ),

            basic_info_events=(
                "has_basic_info_change",
                "sum"
            ),

            close_events=(
                "has_close",
                "sum"
            ),

            file_create_events=(
                "has_file_create",
                "sum"
            ),

            file_delete_events=(
                "has_file_delete",
                "sum"
            ),

            rename_events=(
                "has_rename",
                "sum"
            ),

            data_extend_events=(
                "has_data_extend",
                "sum"
            ),

            security_change_events=(
                "has_security_change",
                "sum"
            ),

            first_usn=(
                "UpdateSequenceNumber",
                "min"
            ),

            last_usn=(
                "UpdateSequenceNumber",
                "max"
            ),

            first_timestamp=(
                "UpdateTimestamp",
                "min"
            ),

            last_timestamp=(
                "UpdateTimestamp",
                "max"
            ),
        )
        .reset_index()
    )

    file_summary["usn_span"] = (
        file_summary["last_usn"]
        - file_summary["first_usn"]
    )

    file_summary["time_span_seconds"] = (
        file_summary["last_timestamp"]
        - file_summary["first_timestamp"]
    ).dt.total_seconds()

    file_summary.insert(
        0,
        "scenario",
        scenario
    )

    summary_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_file_summary.csv"
    )

    file_summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nFILE-LEVEL SUMMARY")

    print(
        file_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # BASIC INFO CHANGE CONTEXT
    # ========================================================

    bic = context_df[
        context_df["has_basic_info_change"]
    ].copy()

    bic_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_BasicInfo_context.csv"
    )

    bic.to_csv(
        bic_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nBasicInfoChange events : "
        f"{len(bic):,}"
    )

    # --------------------------------------------------------
    # Print BasicInfoChange
    # --------------------------------------------------------

    if not bic.empty:

        display_columns = [
            "file_id",
            "event_order",
            "Name",
            "UpdateSequenceNumber",
            "UpdateTimestamp",
            "previous_reason",
            "UpdateReasons",
            "next_reason",
            "usn_delta",
            "time_delta_seconds",
            "next_time_delta_seconds",
            "FileAttributes",
        ]

        display_columns = [
            c
            for c in display_columns
            if c in bic.columns
        ]

        print("\nBasicInfoChange context:")

        print(
            bic[display_columns]
            .to_string(index=False)
        )

    # ========================================================
    # SEQUENCE PATTERN
    # ========================================================

    context_df["sequence_3_event"] = (
        context_df["previous_reason"]
        .fillna("START")
        + " -> "
        + context_df["UpdateReasons"]
        .fillna("UNKNOWN")
        + " -> "
        + context_df["next_reason"]
        .fillna("END")
    )

    pattern_summary = (
        context_df[
            context_df["has_basic_info_change"]
        ]
        ["sequence_3_event"]
        .value_counts()
        .reset_index()
    )

    pattern_summary.columns = [
        "sequence_pattern",
        "frequency"
    ]

    pattern_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_sequence_patterns.csv"
    )

    pattern_summary.to_csv(
        pattern_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop sequence patterns:")

    print(
        pattern_summary
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {
        "scenario": scenario,
        "target_name": target_name,
        "target_rows": len(target_rows),
        "target_file_ids": len(target_file_ids),
        "context_events": len(context_df),
        "basic_info_events": int(
            context_df[
                "has_basic_info_change"
            ].sum()
        ),
        "close_events": int(
            context_df[
                "has_close"
            ].sum()
        ),
        "file_create_events": int(
            context_df[
                "has_file_create"
            ].sum()
        ),
        "file_delete_events": int(
            context_df[
                "has_file_delete"
            ].sum()
        ),
        "rename_events": int(
            context_df[
                "has_rename"
            ].sum()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("TARGET CONTEXT ANALYSIS")
    print("=" * 90)

    results = []

    for scenario, target_name in TARGETS.items():

        result = analyze_scenario(
            scenario,
            target_name
        )

        if result is not None:
            results.append(result)

    # ========================================================
    # CROSS-SCENARIO SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        results
    )

    summary_file = (
        OUTPUT_DIR
        / "all_scenarios_context_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n\n")
    print("=" * 110)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 110)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSummary disimpan:")
    print(summary_file)

    print("\nSELESAI.")


if __name__ == "__main__":
    main()