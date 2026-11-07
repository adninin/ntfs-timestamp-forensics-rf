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
    r"C:\Users\atnin\Downloads\TimestampResearch\output\global_target_context"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Jumlah event sebelum dan sesudah target
WINDOW_SIZE = 20


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
# FUNGSI
# ============================================================

def add_file_id(df):
    """
    Identitas record berdasarkan EntryNumber + SequenceNumber.
    """

    df["file_id"] = (
        df["EntryNumber"].astype(str)
        + "-"
        + df["SequenceNumber"].astype(str)
    )

    return df


def add_event_flags(df):
    """
    Membuat flag berdasarkan UpdateReasons.
    """

    reason = (
        df["UpdateReasons"]
        .fillna("")
        .astype(str)
    )

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


def prepare_dataset(df):
    """
    Membersihkan dan mengurutkan dataset secara global
    menggunakan UpdateSequenceNumber.
    """

    df = add_file_id(df)

    df["UpdateTimestamp"] = pd.to_datetime(
        df["UpdateTimestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        "UpdateSequenceNumber"
    ).reset_index(drop=True)

    # Posisi global dalam USN Journal
    df["global_position"] = (
        np.arange(len(df))
    )

    # Flag event
    df = add_event_flags(df)

    return df


def get_target_rows(df, target_name):
    """
    Mendapatkan indeks global event yang mempunyai
    nama target secara exact match.
    """

    mask = (
        df["Name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(target_name.lower())
    )

    return df.index[mask].tolist()


def extract_global_window(
    df,
    target_index,
    target_name,
    scenario
):
    """
    Mengambil WINDOW_SIZE event sebelum dan sesudah
    satu event target.

    target_index adalah index setelah dataset
    diurutkan berdasarkan USN.
    """

    start = max(
        0,
        target_index - WINDOW_SIZE
    )

    end = min(
        len(df),
        target_index + WINDOW_SIZE + 1
    )

    window = df.iloc[
        start:end
    ].copy()

    # --------------------------------------------------------
    # Posisi relatif terhadap target
    # --------------------------------------------------------

    window["target_row_index"] = target_index

    window["relative_position"] = (
        window.index - target_index
    )

    window["distance_from_target"] = (
        window["relative_position"].abs()
    )

    window["is_target_event"] = (
        window.index == target_index
    )

    window["scenario"] = scenario

    window["target_name"] = target_name

    # --------------------------------------------------------
    # Apakah event berasal dari file_id yang sama?
    # --------------------------------------------------------

    target_file_id = df.loc[
        target_index,
        "file_id"
    ]

    window["target_file_id"] = target_file_id

    window["is_same_file_id"] = (
        window["file_id"]
        == target_file_id
    )

    # --------------------------------------------------------
    # Delta USN terhadap target
    # --------------------------------------------------------

    target_usn = df.loc[
        target_index,
        "UpdateSequenceNumber"
    ]

    window["usn_delta_from_target"] = (
        window["UpdateSequenceNumber"]
        - target_usn
    )

    # --------------------------------------------------------
    # Delta waktu terhadap target
    # --------------------------------------------------------

    target_time = df.loc[
        target_index,
        "UpdateTimestamp"
    ]

    window["time_delta_from_target_seconds"] = (
        window["UpdateTimestamp"]
        - target_time
    ).dt.total_seconds()

    return window


def create_target_summary(
    context_df
):
    """
    Membuat summary konteks global per target event.
    """

    summaries = []

    grouped = context_df.groupby(
        [
            "scenario",
            "target_name",
            "target_row_index"
        ]
    )

    for (
        scenario,
        target_name,
        target_row_index
    ), group in grouped:

        target_rows = group[
            group["is_target_event"]
        ]

        if target_rows.empty:
            continue

        target_row = target_rows.iloc[0]

        # ----------------------------------------------------
        # Event sekitar
        # ----------------------------------------------------

        before = group[
            group["relative_position"] < 0
        ]

        after = group[
            group["relative_position"] > 0
        ]

        # ----------------------------------------------------
        # Event target
        # ----------------------------------------------------

        target_file_id = (
            target_row["target_file_id"]
        )

        # ----------------------------------------------------
        # BasicInfoChange di sekitar target
        # ----------------------------------------------------

        basic_info_before = before[
            before["has_basic_info_change"]
        ]

        basic_info_after = after[
            after["has_basic_info_change"]
        ]

        # ----------------------------------------------------
        # Rename di sekitar target
        # ----------------------------------------------------

        rename_before = before[
            before["has_rename"]
        ]

        rename_after = after[
            after["has_rename"]
        ]

        # ----------------------------------------------------
        # File create/delete
        # ----------------------------------------------------

        create_before = before[
            before["has_file_create"]
        ]

        create_after = after[
            after["has_file_create"]
        ]

        delete_before = before[
            before["has_file_delete"]
        ]

        delete_after = after[
            after["has_file_delete"]
        ]

        # ----------------------------------------------------
        # Same file ID
        # ----------------------------------------------------

        same_file = group[
            group["is_same_file_id"]
        ]

        # ----------------------------------------------------
        # Different file ID
        # ----------------------------------------------------

        other_files = group[
            ~group["is_same_file_id"]
        ]

        # ----------------------------------------------------
        # Timestamp window
        # ----------------------------------------------------

        time_values = group[
            "time_delta_from_target_seconds"
        ].dropna()

        if len(time_values) > 0:
            min_time_delta = time_values.min()
            max_time_delta = time_values.max()
        else:
            min_time_delta = np.nan
            max_time_delta = np.nan

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summaries.append({
            "scenario": scenario,
            "target_name": target_name,
            "target_row_index": target_row_index,

            "target_file_id": target_file_id,

            "target_usn": target_row[
                "UpdateSequenceNumber"
            ],

            "target_timestamp": target_row[
                "UpdateTimestamp"
            ],

            "target_reason": target_row[
                "UpdateReasons"
            ],

            "window_size": len(group),

            "events_before": len(before),

            "events_after": len(after),

            "same_file_events": len(
                same_file
            ),

            "other_file_events": len(
                other_files
            ),

            "basic_info_before": len(
                basic_info_before
            ),

            "basic_info_after": len(
                basic_info_after
            ),

            "rename_before": len(
                rename_before
            ),

            "rename_after": len(
                rename_after
            ),

            "create_before": len(
                create_before
            ),

            "create_after": len(
                create_after
            ),

            "delete_before": len(
                delete_before
            ),

            "delete_after": len(
                delete_after
            ),

            "min_time_delta_seconds": (
                min_time_delta
            ),

            "max_time_delta_seconds": (
                max_time_delta
            ),
        })

    return pd.DataFrame(
        summaries
    )


# ============================================================
# ANALISIS SATU SKENARIO
# ============================================================

def analyze_scenario(
    scenario,
    target_name
):

    print("\n")
    print("=" * 100)
    print(f"SCENARIO {scenario}")
    print(f"TARGET   : {target_name}")
    print("=" * 100)

    csv_path = (
        BASE_DIR
        / f"Scenario_{scenario}.csv"
    )

    if not csv_path.exists():

        print(
            f"ERROR: File tidak ditemukan:"
        )
        print(csv_path)

        return None

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    print(
        f"Total rows : {len(df):,}"
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_dataset(
        df
    )

    # --------------------------------------------------------
    # TARGET INDEX
    # --------------------------------------------------------

    target_indices = get_target_rows(
        df,
        target_name
    )

    print(
        f"Target events : "
        f"{len(target_indices):,}"
    )

    if not target_indices:

        print(
            "TARGET TIDAK DITEMUKAN."
        )

        return None

    # --------------------------------------------------------
    # EXTRACT GLOBAL WINDOWS
    # --------------------------------------------------------

    windows = []

    for target_index in target_indices:

        window = extract_global_window(
            df=df,
            target_index=target_index,
            target_name=target_name,
            scenario=scenario
        )

        windows.append(
            window
        )

    context_df = pd.concat(
        windows,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Hilangkan kemungkinan duplikasi
    #
    # Satu event bisa masuk window beberapa target.
    # Kita pertahankan semua kemunculannya karena
    # relative_position berbeda terhadap target.
    # --------------------------------------------------------

    print(
        f"Total window records : "
        f"{len(context_df):,}"
    )

    # --------------------------------------------------------
    # SIMPAN DETAIL
    # --------------------------------------------------------

    detail_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_global_context.csv"
    )

    context_df.to_csv(
        detail_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nDetail disimpan:"
    )
    print(detail_file)

    # --------------------------------------------------------
    # SUMMARY TARGET EVENT
    # --------------------------------------------------------

    summary_df = create_target_summary(
        context_df
    )

    summary_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_global_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nTarget context summary:"
    )

    display_columns = [
        "scenario",
        "target_name",
        "target_row_index",
        "target_file_id",
        "target_usn",
        "target_reason",
        "window_size",
        "events_before",
        "events_after",
        "same_file_events",
        "other_file_events",
        "basic_info_before",
        "basic_info_after",
        "rename_before",
        "rename_after",
        "create_before",
        "create_after",
        "delete_before",
        "delete_after",
        "min_time_delta_seconds",
        "max_time_delta_seconds",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in summary_df.columns
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # EVENT REASON DISTRIBUTION
    # --------------------------------------------------------

    reason_summary = (
        context_df[
            ~context_df["is_target_event"]
        ]["UpdateReasons"]
        .fillna("UNKNOWN")
        .value_counts()
        .reset_index()
    )

    reason_summary.columns = [
        "UpdateReasons",
        "frequency"
    ]

    reason_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_global_reason_summary.csv"
    )

    reason_summary.to_csv(
        reason_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nTop global context UpdateReasons:"
    )

    print(
        reason_summary
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # EVENT DENGAN DISTANCE TERDEKAT
    # --------------------------------------------------------

    nearest_context = context_df[
        ~context_df["is_target_event"]
    ].copy()

    nearest_context = (
        nearest_context
        .sort_values(
            [
                "target_row_index",
                "distance_from_target"
            ]
        )
    )

    nearest_file = (
        OUTPUT_DIR
        / f"Scenario_{scenario}_nearest_context.csv"
    )

    nearest_context.to_csv(
        nearest_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("GLOBAL TARGET CONTEXT ANALYSIS")
    print("=" * 100)

    all_summaries = []

    for scenario, target_name in TARGETS.items():

        summary_df = analyze_scenario(
            scenario,
            target_name
        )

        if (
            summary_df is not None
            and not summary_df.empty
        ):

            all_summaries.append(
                summary_df
            )

    # ========================================================
    # GABUNGKAN 12 SKENARIO
    # ========================================================

    if not all_summaries:

        print(
            "\nTidak ada hasil."
        )

        return

    all_summary_df = pd.concat(
        all_summaries,
        ignore_index=True
    )

    all_summary_file = (
        OUTPUT_DIR
        / "all_scenarios_global_summary.csv"
    )

    all_summary_df.to_csv(
        all_summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # CROSS-SCENARIO
    # ========================================================

    print("\n\n")
    print("=" * 120)
    print("CROSS-SCENARIO GLOBAL CONTEXT SUMMARY")
    print("=" * 120)

    cross_summary = (
        all_summary_df
        .groupby(
            [
                "scenario",
                "target_name"
            ],
            as_index=False
        )
        .agg(
            target_events=(
                "target_row_index",
                "count"
            ),

            mean_same_file_events=(
                "same_file_events",
                "mean"
            ),

            mean_other_file_events=(
                "other_file_events",
                "mean"
            ),

            mean_basic_info_before=(
                "basic_info_before",
                "mean"
            ),

            mean_basic_info_after=(
                "basic_info_after",
                "mean"
            ),

            mean_rename_before=(
                "rename_before",
                "mean"
            ),

            mean_rename_after=(
                "rename_after",
                "mean"
            ),

            mean_create_before=(
                "create_before",
                "mean"
            ),

            mean_create_after=(
                "create_after",
                "mean"
            ),

            mean_delete_before=(
                "delete_before",
                "mean"
            ),

            mean_delete_after=(
                "delete_after",
                "mean"
            ),

            mean_min_time_delta=(
                "min_time_delta_seconds",
                "mean"
            ),

            mean_max_time_delta=(
                "max_time_delta_seconds",
                "mean"
            ),
        )
    )

    cross_file = (
        OUTPUT_DIR
        / "cross_scenario_global_summary.csv"
    )

    cross_summary.to_csv(
        cross_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        cross_summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        "All summary:"
    )
    print(all_summary_file)

    print(
        "\nCross summary:"
    )
    print(cross_file)

    print("\nSELESAI.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()