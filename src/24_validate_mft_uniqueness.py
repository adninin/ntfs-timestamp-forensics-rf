from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

MFT_CSV = (
    PROJECT_DIR
    / "output"
    / "mft_scenario_01"
    / "Scenario_01_MFT.csv"
)

FEATURE_CSV = (
    PROJECT_DIR
    / "output"
    / "mft_scenario_01"
    / "Scenario_01_forensic_timestamp_features.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 100)
print("SCRIPT 24 - VALIDATE MFT UNIQUENESS")
print("=" * 100)


# ============================================================
# LOAD MFT
# ============================================================

print("\nLOAD MFT")
print("=" * 100)

mft = pd.read_csv(
    MFT_CSV,
    low_memory=False
)

print(
    f"Rows: {len(mft):,}"
)


# ============================================================
# CREATE FILE ID
# ============================================================

mft["EntryNumber"] = pd.to_numeric(
    mft["EntryNumber"],
    errors="coerce"
)

mft["SequenceNumber"] = pd.to_numeric(
    mft["SequenceNumber"],
    errors="coerce"
)

mft["file_id"] = (
    mft["EntryNumber"].astype("Int64").astype(str)
    + "-"
    + mft["SequenceNumber"].astype("Int64").astype(str)
)


# ============================================================
# UNIQUENESS
# ============================================================

unique_ids = (
    mft["file_id"]
    .nunique()
)

duplicate_rows = (
    len(mft) - unique_ids
)

print("\n")
print("=" * 100)
print("MFT UNIQUENESS")
print("=" * 100)

print(
    f"Total rows          : {len(mft):,}"
)

print(
    f"Unique file_id      : {unique_ids:,}"
)

print(
    f"Duplicate rows      : {duplicate_rows:,}"
)


# ============================================================
# DUPLICATE FILE IDS
# ============================================================

dup_counts = (
    mft["file_id"]
    .value_counts()
)

duplicated_ids = (
    dup_counts[
        dup_counts > 1
    ]
)

print(
    f"\nfile_id with duplicates: "
    f"{len(duplicated_ids):,}"
)


if len(duplicated_ids) > 0:

    print("\nTOP DUPLICATE FILE IDs")
    print("=" * 100)

    print(
        duplicated_ids
        .head(20)
        .to_string()
    )


# ============================================================
# TARGET CHECK
# ============================================================

TARGET = (
    "NewFileTime_SI_C_Manipulation.dll"
)

target = mft[
    mft["FileName"]
    .astype(str)
    .str.strip()
    .str.lower()
    == TARGET.lower()
].copy()

print("\n")
print("=" * 100)
print("TARGET UNIQUENESS")
print("=" * 100)

print(
    f"Target rows: {len(target)}"
)

print(
    target[
        [
            "EntryNumber",
            "SequenceNumber",
            "file_id",
            "FileName",
            "SourceFile"
        ]
    ].to_string(index=False)
)


# ============================================================
# LOAD FORENSIC FEATURES
# ============================================================

print("\n")
print("=" * 100)
print("FORENSIC FEATURE DATASET")
print("=" * 100)

features = pd.read_csv(
    FEATURE_CSV,
    low_memory=False
)

features["EntryNumber"] = pd.to_numeric(
    features["EntryNumber"],
    errors="coerce"
)

features["SequenceNumber"] = pd.to_numeric(
    features["SequenceNumber"],
    errors="coerce"
)

features["file_id"] = (
    features["EntryNumber"].astype("Int64").astype(str)
    + "-"
    + features["SequenceNumber"].astype("Int64").astype(str)
)

feature_unique = (
    features["file_id"]
    .nunique()
)

feature_duplicate_rows = (
    len(features) - feature_unique
)

print(
    f"Feature rows       : {len(features):,}"
)

print(
    f"Unique file_id     : {feature_unique:,}"
)

print(
    f"Duplicate rows     : {feature_duplicate_rows:,}"
)


# ============================================================
# DUPLICATE FEATURE IDS
# ============================================================

feature_dup_counts = (
    features["file_id"]
    .value_counts()
)

feature_duplicated_ids = (
    feature_dup_counts[
        feature_dup_counts > 1
    ]
)

print(
    f"Feature file_id with duplicates: "
    f"{len(feature_duplicated_ids):,}"
)


if len(feature_duplicated_ids) > 0:

    print("\nTOP DUPLICATE FEATURE FILE IDs")
    print("=" * 100)

    print(
        feature_duplicated_ids
        .head(20)
        .to_string()
    )


# ============================================================
# TARGET IN FEATURE DATASET
# ============================================================

target_feature = features[
    features["file_id"]
    == "123518-548"
].copy()

print("\n")
print("=" * 100)
print("TARGET IN FORENSIC FEATURE DATASET")
print("=" * 100)

print(
    f"Rows for target file_id: "
    f"{len(target_feature)}"
)

if len(target_feature) > 0:

    print(
        target_feature[
            [
                "EntryNumber",
                "SequenceNumber",
                "FileName",
                "file_id",
                "max_abs_timestamp_delta",
                "mean_abs_timestamp_delta"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE DUPLICATE LIST
# ============================================================

if len(duplicated_ids) > 0:

    duplicate_output = (
        OUTPUT_DIR
        / "Scenario_01_mft_duplicate_file_ids.csv"
    )

    (
        duplicated_ids
        .rename("row_count")
        .reset_index()
        .rename(
            columns={
                "index": "file_id"
            }
        )
        .to_csv(
            duplicate_output,
            index=False
        )
    )

    print(
        f"\nDuplicate list:"
        f"\n{duplicate_output}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 100)
print("SCRIPT 24 SELESAI")
print("=" * 100)