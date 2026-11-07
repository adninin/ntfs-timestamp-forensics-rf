import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

MFT_BASE = (
    BASE_DIR
    / "output"
    / "mft_record_features_all"
)

USN_BASE = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
) / "CSV_PERFORMANCE"

OUTPUT_BASE = (
    BASE_DIR
    / "output"
    / "integrated_dataset_all"
)

OUTPUT_BASE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCTION
# ============================================================

def integrate_scenario(scenario_number):

    scenario = (
        f"Scenario_{scenario_number:02d}"
    )

    print("\n" + "=" * 70)
    print(scenario)
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    mft_file = (
        MFT_BASE
        / f"{scenario}_MFT_record_features.csv"
    )

    usn_file = (
        USN_BASE
        / f"{scenario}.csv"
    )

    output_file = (
        OUTPUT_BASE
        / f"{scenario}_integrated.csv"
    )

    print("\nMFT:")
    print(mft_file)

    print("\nUSN:")
    print(usn_file)

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if not mft_file.exists():

        print(
            "[FAILED] MFT tidak ditemukan."
        )

        return {
            "scenario": scenario,
            "status": "mft_not_found"
        }

    if not usn_file.exists():

        print(
            "[FAILED] USN tidak ditemukan."
        )

        return {
            "scenario": scenario,
            "status": "usn_not_found"
        }

    # --------------------------------------------------------
    # SKIP
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "[SKIP] Output sudah ada:"
        )

        print(output_file)

        return {
            "scenario": scenario,
            "status": "already_exists",
            "output": str(output_file)
        }

    # ========================================================
    # LOAD MFT
    # ========================================================

    print(
        "\n[1] Loading MFT record features..."
    )

    mft = pd.read_csv(
        mft_file,
        low_memory=False
    )

    print(
        f"MFT rows    : {len(mft):,}"
    )

    print(
        f"MFT columns : {len(mft.columns)}"
    )

    # ========================================================
    # LOAD USN
    # ========================================================

    print(
        "\n[2] Loading raw USN..."
    )

    usn = pd.read_csv(
        usn_file,
        low_memory=False
    )

    print(
        f"USN rows    : {len(usn):,}"
    )

    print(
        f"USN columns : {len(usn.columns)}"
    )

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    required_usn_columns = [
        "EntryNumber",
        "SequenceNumber",
        "UpdateSequenceNumber",
        "UpdateTimestamp",
        "UpdateReasons",
        "Name",
    ]

    missing = [
        col
        for col in required_usn_columns
        if col not in usn.columns
    ]

    if missing:

        raise ValueError(
            f"Kolom USN tidak ditemukan: {missing}"
        )

    # ========================================================
    # CREATE FILE ID
    # ========================================================

    print(
        "\n[3] Creating USN file_id..."
    )

    usn["EntryNumber"] = pd.to_numeric(
        usn["EntryNumber"],
        errors="coerce"
    )

    usn["SequenceNumber"] = pd.to_numeric(
        usn["SequenceNumber"],
        errors="coerce"
    )

    usn["file_id"] = (
        usn["EntryNumber"]
        .astype("Int64")
        .astype(str)
        + "-"
        + usn["SequenceNumber"]
        .astype("Int64")
        .astype(str)
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    print(
        "\n[4] Preparing timestamps..."
    )

    usn["UpdateTimestamp"] = pd.to_datetime(
        usn["UpdateTimestamp"],
        errors="coerce"
    )

    usn["UpdateSequenceNumber"] = pd.to_numeric(
        usn["UpdateSequenceNumber"],
        errors="coerce"
    )

    # ========================================================
    # EVENT COUNT
    # ========================================================

    print(
        "\n[5] Creating USN event features..."
    )

    usn["usn_event_count"] = 1

    reason_mapping = {

        "basic_info":
            "BasicInfoChange",

        "close":
            "Close",

        "file_create":
            "FileCreate",

        "file_delete":
            "FileDelete",

        "rename_old":
            "RenameOldName",

        "rename_new":
            "RenameNewName",

        "data_extend":
            "DataExtend",

        "data_overwrite":
            "DataOverwrite",

        "data_truncation":
            "DataTruncation",

        "security_change":
            "SecurityChange",

        "indexable_change":
            "IndexableChange",

        "named_data_extend":
            "NamedDataExtend",

        "reparse_point_change":
            "ReparsePointChange",

        "stream_change":
            "StreamChange",
    }

    for feature_name, reason in (
        reason_mapping.items()
    ):

        column_name = (
            f"usn_{feature_name}_count"
        )

        usn[column_name] = (
            usn["UpdateReasons"]
            .astype(str)
            .str.contains(
                reason,
                regex=False,
                na=False
            )
            .astype(int)
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    print(
        "\n[6] Aggregating USN by MFT record..."
    )

    agg_dict = {

        "usn_event_count": (
            "usn_event_count",
            "sum"
        ),

        "usn_first_sequence": (
            "UpdateSequenceNumber",
            "min"
        ),

        "usn_last_sequence": (
            "UpdateSequenceNumber",
            "max"
        ),

        "usn_first_timestamp": (
            "UpdateTimestamp",
            "min"
        ),

        "usn_last_timestamp": (
            "UpdateTimestamp",
            "max"
        ),
    }

    for feature_name in (
        reason_mapping.keys()
    ):

        column_name = (
            f"usn_{feature_name}_count"
        )

        agg_dict[column_name] = (
            column_name,
            "sum"
        )

    usn_agg = (
        usn
        .groupby(
            "file_id",
            sort=False
        )
        .agg(**agg_dict)
        .reset_index()
    )

    print(
        f"USN aggregated records : "
        f"{len(usn_agg):,}"
    )

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    print(
        "\n[7] Creating derived USN features..."
    )

    usn_agg["usn_sequence_span"] = (
        usn_agg["usn_last_sequence"]
        -
        usn_agg["usn_first_sequence"]
    )

    usn_agg["usn_time_span_seconds"] = (
        (
            usn_agg["usn_last_timestamp"]
            -
            usn_agg["usn_first_timestamp"]
        )
        .dt.total_seconds()
    )

    # ========================================================
    # RATIOS
    # ========================================================

    print(
        "\n[8] Creating USN ratios..."
    )

    for feature_name in (
        reason_mapping.keys()
    ):

        count_col = (
            f"usn_{feature_name}_count"
        )

        ratio_col = (
            f"usn_{feature_name}_ratio"
        )

        usn_agg[ratio_col] = (
            usn_agg[count_col]
            /
            usn_agg["usn_event_count"]
        )

    # ========================================================
    # MERGE
    # ========================================================

    print(
        "\n[9] Merging MFT + USN..."
    )

    integrated = mft.merge(
        usn_agg,
        on="file_id",
        how="left",
        validate="one_to_one"
    )

    print(
        f"Integrated rows : "
        f"{len(integrated):,}"
    )

    # ========================================================
    # HAS USN
    # ========================================================

    integrated["has_usn"] = (
        integrated[
            "usn_event_count"
        ]
        .notna()
        .astype(int)
    )

    with_usn = (
        integrated["has_usn"]
        .eq(1)
        .sum()
    )

    without_usn = (
        integrated["has_usn"]
        .eq(0)
        .sum()
    )

    print(
        f"Records with USN    : "
        f"{with_usn:,}"
    )

    print(
        f"Records without USN : "
        f"{without_usn:,}"
    )

    # ========================================================
    # FILL COUNT / RATIO
    # ========================================================

    print(
        "\n[10] Filling missing USN features..."
    )

    count_columns = [
        col
        for col in integrated.columns
        if (
            col.startswith("usn_")
            and col.endswith("_count")
        )
    ]

    for col in count_columns:

        integrated[col] = (
            integrated[col]
            .fillna(0)
        )

    ratio_columns = [
        col
        for col in integrated.columns
        if (
            col.startswith("usn_")
            and col.endswith("_ratio")
        )
    ]

    for col in ratio_columns:

        integrated[col] = (
            integrated[col]
            .fillna(0)
        )

    # ========================================================
    # QUALITY CHECK
    # ========================================================

    print(
        "\n[11] QUALITY CHECK"
    )

    print("-" * 70)

    print(
        f"MFT input rows      : "
        f"{len(mft):,}"
    )

    print(
        f"USN input rows      : "
        f"{len(usn):,}"
    )

    print(
        f"USN aggregated rows : "
        f"{len(usn_agg):,}"
    )

    print(
        f"Integrated rows     : "
        f"{len(integrated):,}"
    )

    print(
        f"Integrated columns  : "
        f"{len(integrated.columns)}"
    )

    print(
        f"Unique file_id      : "
        f"{integrated['file_id'].nunique():,}"
    )

    print(
        f"Duplicate file_id   : "
        f"{integrated['file_id'].duplicated().sum():,}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        "\n[12] Saving..."
    )

    integrated.to_csv(
        output_file,
        index=False
    )

    print(
        "Output:"
    )

    print(
        output_file
    )

    return {
        "scenario": scenario,
        "status": "success",
        "mft_rows": len(mft),
        "usn_rows": len(usn),
        "usn_records": len(usn_agg),
        "integrated_rows":
            len(integrated),
        "integrated_columns":
            len(integrated.columns),
        "with_usn":
            int(with_usn),
        "without_usn":
            int(without_usn),
        "duplicate_file_id":
            int(
                integrated[
                    "file_id"
                ].duplicated().sum()
            ),
        "output":
            str(output_file)
    }


# ============================================================
# MAIN
# ============================================================

results = []

for scenario_number in range(1, 13):

    try:

        result = integrate_scenario(
            scenario_number
        )

        results.append(result)

    except Exception as e:

        print(
            "\n[ERROR]"
        )

        print(
            repr(e)
        )

        results.append({
            "scenario":
                f"Scenario_{scenario_number:02d}",

            "status":
                "error",

            "error":
                repr(e)
        })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH INTEGRATION SUMMARY")
print("=" * 70)

summary = pd.DataFrame(
    results
)

print(
    summary.to_string(
        index=False
    )
)

summary_file = (
    OUTPUT_BASE
    / "batch_integration_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    "\nSummary saved:"
)

print(
    summary_file
)

success_count = (
    summary["status"]
    .isin([
        "success",
        "already_exists"
    ])
    .sum()
)

print(
    f"\nCompleted: "
    f"{success_count}/12"
)

print(
    "\n" + "=" * 70
)

print(
    "SCRIPT 30 SELESAI"
)

print(
    "=" * 70
)