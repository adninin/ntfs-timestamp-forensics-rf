import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

MFT_FILE = (
    BASE_DIR
    / "output"
    / "mft_record_features"
    / "Scenario_01_MFT_record_features.csv"
)

USN_FILE = (
    BASE_DIR
    / "output"
    / "sequence_features_scenario_01.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "integrated_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "Scenario_01_MFT_USN_integrated.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SCRIPT 27 - INTEGRASI MFT + USN")
print("=" * 70)


# ============================================================
# 1. LOAD MFT
# ============================================================

print("\n[1] Loading MFT record features...")

mft = pd.read_csv(
    MFT_FILE,
    low_memory=False
)

print(
    f"MFT rows    : {len(mft):,}"
)

print(
    f"MFT columns : {len(mft.columns)}"
)


# ============================================================
# 2. LOAD USN
# ============================================================

print("\n[2] Loading USN sequence features...")

usn = pd.read_csv(
    USN_FILE,
    low_memory=False
)

print(
    f"USN rows    : {len(usn):,}"
)

print(
    f"USN columns : {len(usn.columns)}"
)


# ============================================================
# 3. VALIDATE FILE_ID
# ============================================================

print("\n[3] Validating file_id...")


if "file_id" not in mft.columns:

    raise ValueError(
        "Kolom file_id tidak ditemukan pada MFT."
    )


if "file_id" not in usn.columns:

    raise ValueError(
        "Kolom file_id tidak ditemukan pada USN."
    )


print(
    f"MFT unique file_id : "
    f"{mft['file_id'].nunique():,}"
)

print(
    f"USN unique file_id : "
    f"{usn['file_id'].nunique():,}"
)


# ============================================================
# 4. CHECK MFT UNIQUENESS
# ============================================================

print("\n[4] Checking MFT uniqueness...")

mft_duplicates = (
    mft["file_id"]
    .duplicated()
    .sum()
)

print(
    f"MFT duplicate file_id : "
    f"{mft_duplicates:,}"
)

if mft_duplicates != 0:

    raise ValueError(
        "MFT tidak unik berdasarkan file_id."
    )


# ============================================================
# 5. PREPARE USN TIMESTAMP
# ============================================================

print("\n[5] Preparing USN timestamps...")


if "UpdateTimestamp" in usn.columns:

    usn["UpdateTimestamp"] = pd.to_datetime(
        usn["UpdateTimestamp"],
        errors="coerce"
    )


# ============================================================
# 6. USN BASIC COUNTS
# ============================================================

print("\n[6] Aggregating USN event counts...")


usn["usn_event_count"] = 1


# ------------------------------------------------------------
# Event reason indicators
# ------------------------------------------------------------

reason_mapping = {

    "basic_info": "BasicInfoChange",
    "close": "Close",
    "file_create": "FileCreate",
    "file_delete": "FileDelete",
    "rename_old": "RenameOldName",
    "rename_new": "RenameNewName",
    "data_extend": "DataExtend",
    "data_overwrite": "DataOverwrite",
    "data_truncation": "DataTruncation",
    "security_change": "SecurityChange",
    "indexable_change": "IndexableChange",
    "named_data_extend": "NamedDataExtend",
    "reparse_point_change": "ReparsePointChange",
    "stream_change": "StreamChange",
}


for feature_name, reason in reason_mapping.items():

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


# ============================================================
# 7. USN AGGREGATION
# ============================================================

print("\n[7] Aggregating USN by MFT record...")


agg_dict = {

    "usn_event_count": (
        "usn_event_count",
        "sum"
    ),

}


for feature_name in reason_mapping.keys():

    column_name = (
        f"usn_{feature_name}_count"
    )

    agg_dict[column_name] = (
        column_name,
        "sum"
    )


# ------------------------------------------------------------
# USN order
# ------------------------------------------------------------

if "UpdateSequenceNumber" in usn.columns:

    usn["UpdateSequenceNumber"] = pd.to_numeric(
        usn["UpdateSequenceNumber"],
        errors="coerce"
    )

    agg_dict["usn_first_sequence"] = (
        "UpdateSequenceNumber",
        "min"
    )

    agg_dict["usn_last_sequence"] = (
        "UpdateSequenceNumber",
        "max"
    )


# ------------------------------------------------------------
# Timestamp
# ------------------------------------------------------------

if "UpdateTimestamp" in usn.columns:

    agg_dict["usn_first_timestamp"] = (
        "UpdateTimestamp",
        "min"
    )

    agg_dict["usn_last_timestamp"] = (
        "UpdateTimestamp",
        "max"
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


# ============================================================
# 8. USN DERIVED FEATURES
# ============================================================

print("\n[8] Creating USN derived features...")


# ------------------------------------------------------------
# USN span
# ------------------------------------------------------------

if (
    "usn_first_sequence" in usn_agg.columns
    and
    "usn_last_sequence" in usn_agg.columns
):

    usn_agg["usn_sequence_span"] = (
        usn_agg["usn_last_sequence"]
        - usn_agg["usn_first_sequence"]
    )


# ------------------------------------------------------------
# Timestamp span
# ------------------------------------------------------------

if (
    "usn_first_timestamp" in usn_agg.columns
    and
    "usn_last_timestamp" in usn_agg.columns
):

    usn_agg["usn_time_span_seconds"] = (
        (
            usn_agg["usn_last_timestamp"]
            -
            usn_agg["usn_first_timestamp"]
        )
        .dt.total_seconds()
    )


# ============================================================
# 9. USN REASON RATIOS
# ============================================================

print("\n[9] Creating USN reason ratios...")


for feature_name in reason_mapping.keys():

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


# ============================================================
# 10. MERGE MFT + USN
# ============================================================

print("\n[10] Merging MFT + USN...")


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


# ============================================================
# 11. RECORDS WITHOUT USN
# ============================================================

print("\n[11] Checking MFT records without USN...")


integrated["has_usn"] = (
    integrated["usn_event_count"]
    .notna()
    .astype(int)
)


without_usn = (
    integrated["has_usn"]
    .eq(0)
    .sum()
)

with_usn = (
    integrated["has_usn"]
    .eq(1)
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


# ============================================================
# 12. FILL USN COUNT FEATURES
# ============================================================

print("\n[12] Filling missing USN count features...")


usn_count_columns = [
    col
    for col in integrated.columns
    if col.startswith("usn_")
    and col.endswith("_count")
]


for col in usn_count_columns:

    integrated[col] = (
        integrated[col]
        .fillna(0)
    )


# ============================================================
# 13. FILL USN RATIO FEATURES
# ============================================================

usn_ratio_columns = [
    col
    for col in integrated.columns
    if col.startswith("usn_")
    and col.endswith("_ratio")
]


for col in usn_ratio_columns:

    integrated[col] = (
        integrated[col]
        .fillna(0)
    )


# ============================================================
# 14. QUALITY CHECK
# ============================================================

print("\n[13] QUALITY CHECK")
print("-" * 70)


print(
    f"MFT input rows       : "
    f"{len(mft):,}"
)

print(
    f"USN input rows       : "
    f"{len(usn):,}"
)

print(
    f"USN aggregated rows  : "
    f"{len(usn_agg):,}"
)

print(
    f"Integrated rows      : "
    f"{len(integrated):,}"
)

print(
    f"Integrated columns   : "
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


# ============================================================
# 15. TARGET CHECK
# ============================================================

print("\n[14] TARGET CHECK")
print("-" * 70)


target_name = (
    "NewFileTime_SI_C_Manipulation.dll"
)


target_usn = usn[
    usn["Name"]
    .astype(str)
    .str.lower()
    .eq(target_name.lower())
]


print(
    f"Target USN rows : "
    f"{len(target_usn)}"
)


if len(target_usn) > 0:

    target_ids = (
        target_usn["file_id"]
        .unique()
    )

    print(
        f"Target file_id(s): "
        f"{list(target_ids)}"
    )

    target_integrated = integrated[
        integrated["file_id"].isin(
            target_ids
        )
    ]

    print(
        "\nTarget integrated record:"
    )

    print(
        target_integrated[
            [
                "file_id",
                "EntryNumber",
                "SequenceNumber",
                "raw_row_count",
                "filename_count",
                "file_size",
                "si_fn_true",
                "usn_event_count",
                "usn_basic_info_count",
                "usn_close_count",
                "usn_file_create_count",
                "usn_file_delete_count",
                "usn_data_extend_count",
                "usn_data_overwrite_count",
                "usn_first_sequence",
                "usn_last_sequence",
                "usn_sequence_span",
                "usn_time_span_seconds",
                "has_usn",
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# 16. SAVE
# ============================================================

print("\n[15] Saving integrated dataset...")


integrated.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nOutput saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("SCRIPT 27 SELESAI")
print("=" * 70)