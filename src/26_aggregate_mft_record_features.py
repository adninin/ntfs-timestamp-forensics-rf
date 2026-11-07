import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_MFT = (
    BASE_DIR
    / "output"
    / "mft_scenario_01"
    / "Scenario_01_MFT.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "mft_record_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "Scenario_01_MFT_record_features.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SCRIPT 26 - AGGREGATE MFT RECORD FEATURES (OPTIMIZED)")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1] Loading MFT...")

df = pd.read_csv(
    INPUT_MFT,
    low_memory=False
)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. CREATE MFT RECORD ID
# ============================================================

print("\n[2] Creating MFT record ID...")

df["EntryNumber"] = pd.to_numeric(
    df["EntryNumber"],
    errors="coerce"
)

df["SequenceNumber"] = pd.to_numeric(
    df["SequenceNumber"],
    errors="coerce"
)

df["file_id"] = (
    df["EntryNumber"].astype("Int64").astype(str)
    + "-"
    + df["SequenceNumber"].astype("Int64").astype(str)
)

print(
    f"Unique MFT records: "
    f"{df['file_id'].nunique():,}"
)


# ============================================================
# 3. TIMESTAMP CONVERSION
# ============================================================

print("\n[3] Converting timestamps...")

timestamp_columns = [
    "Created0x10",
    "Created0x30",
    "LastModified0x10",
    "LastModified0x30",
    "LastRecordChange0x10",
    "LastRecordChange0x30",
    "LastAccess0x10",
    "LastAccess0x30",
]

for col in timestamp_columns:

    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )


# ============================================================
# 4. TIMESTAMP DELTA
# ============================================================

print("\n[4] Creating timestamp delta features...")

pairs = {
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
    ),
}

for name, (si, fn) in pairs.items():

    delta = (
        df[si] - df[fn]
    ).dt.total_seconds()

    df[f"delta_{name}_seconds"] = delta

    df[f"abs_delta_{name}_seconds"] = (
        delta.abs()
    )

    df[f"si_fn_exact_{name}"] = (
        delta == 0
    )

    df[f"si_before_fn_{name}"] = (
        delta < 0
    )

    df[f"si_after_fn_{name}"] = (
        delta > 0
    )


# ============================================================
# 5. BOOLEAN NORMALIZATION
# ============================================================

print("\n[5] Normalizing boolean fields...")


def to_bool(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )


boolean_columns = [
    "SI<FN",
    "InUse",
    "Copied",
    "IsDirectory",
    "HasAds",
    "IsAds",
]

for col in boolean_columns:

    if col in df.columns:

        df[col] = to_bool(
            df[col]
        )


# ============================================================
# 6. GROUP OBJECT
# ============================================================

print("\n[6] Creating groupby object...")

group = df.groupby(
    "file_id",
    sort=False
)


# ============================================================
# 7. CORE FEATURES
# ============================================================

print("\n[7] Aggregating core MFT features...")


features = group.agg(

    EntryNumber=(
        "EntryNumber",
        "first"
    ),

    SequenceNumber=(
        "SequenceNumber",
        "first"
    ),

    raw_row_count=(
        "file_id",
        "size"
    ),

    filename_count=(
        "FileName",
        "count"
    ),

    unique_filename_count=(
        "FileName",
        "nunique"
    ),

    parent_count=(
        "ParentEntryNumber",
        "nunique"
    ),

    parent_sequence_count=(
        "ParentSequenceNumber",
        "nunique"
    ),

    parent_path_count=(
        "ParentPath",
        "nunique"
    ),

    name_type_count=(
        "NameType",
        "nunique"
    ),

    extension_count=(
        "Extension",
        "nunique"
    ),

    source_file_count=(
        "SourceFile",
        "nunique"
    ),

    file_size=(
        "FileSize",
        "first"
    ),

    reference_count=(
        "ReferenceCount",
        "first"
    ),

    is_directory=(
        "IsDirectory",
        "max"
    ),

    has_ads=(
        "HasAds",
        "max"
    ),

    is_ads=(
        "IsAds",
        "max"
    ),

    si_fn_true_count=(
        "SI<FN",
        "sum"
    ),

    in_use_count=(
        "InUse",
        "sum"
    ),

    copied_count=(
        "Copied",
        "sum"
    ),

).reset_index()


print(
    f"Aggregated records: "
    f"{len(features):,}"
)


# ============================================================
# 8. NAME TYPE FEATURES
# ============================================================

print("\n[8] Creating NameType features...")


name_type = (
    df["NameType"]
    .astype(str)
    .str.lower()
)

df["_is_windows_name"] = (
    name_type == "windows"
)

df["_is_posix_name"] = (
    name_type == "posix"
)

df["_is_dos_name"] = (
    name_type == "dos"
)


name_features = group.agg(

    has_windows_name=(
        "_is_windows_name",
        "max"
    ),

    has_posix_name=(
        "_is_posix_name",
        "max"
    ),

    has_dos_name=(
        "_is_dos_name",
        "max"
    ),

).reset_index()


features = features.merge(
    name_features,
    on="file_id",
    how="left"
)


# ============================================================
# 9. TIMESTAMP AGGREGATION
# ============================================================

print("\n[9] Aggregating timestamp features...")


timestamp_feature_columns = [
    "abs_delta_creation_seconds",
    "abs_delta_modified_seconds",
    "abs_delta_record_change_seconds",
    "abs_delta_access_seconds",
]


timestamp_agg = group[
    timestamp_feature_columns
].agg([
    "min",
    "max",
    "mean"
])


timestamp_agg.columns = [
    f"{col}_{stat}"
    for col, stat
    in timestamp_agg.columns
]

timestamp_agg = (
    timestamp_agg
    .reset_index()
)


features = features.merge(
    timestamp_agg,
    on="file_id",
    how="left"
)


# ============================================================
# 10. TIMESTAMP RANGE
# ============================================================

print("\n[10] Creating timestamp range features...")


for name in [
    "creation",
    "modified",
    "record_change",
    "access",
]:

    min_col = (
        f"abs_delta_{name}_seconds_min"
    )

    max_col = (
        f"abs_delta_{name}_seconds_max"
    )

    range_col = (
        f"abs_delta_{name}_range"
    )

    features[range_col] = (
        features[max_col]
        - features[min_col]
    )


# ============================================================
# 11. SI/FN COUNTS
# ============================================================

print("\n[11] Aggregating SI/FN relationships...")


for name in [
    "creation",
    "modified",
    "record_change",
    "access",
]:

    exact_col = (
        f"si_fn_exact_{name}"
    )

    before_col = (
        f"si_before_fn_{name}"
    )

    after_col = (
        f"si_after_fn_{name}"
    )

    temp = group.agg(

        **{
            f"si_fn_exact_count_{name}":
                (exact_col, "sum"),

            f"si_before_fn_count_{name}":
                (before_col, "sum"),

            f"si_after_fn_count_{name}":
                (after_col, "sum"),
        }

    ).reset_index()

    features = features.merge(
        temp,
        on="file_id",
        how="left"
    )


# ============================================================
# 12. TIMESTAMP YEAR FEATURES
# ============================================================

print("\n[12] Creating timestamp year features...")


for col in timestamp_columns:

    year_col = f"{col}_year"

    df[year_col] = (
        df[col]
        .dt.year
    )

    temp = group.agg(
        **{
            f"{year_col}_min":
                (year_col, "min"),

            f"{year_col}_max":
                (year_col, "max"),
        }
    ).reset_index()

    features = features.merge(
        temp,
        on="file_id",
        how="left"
    )


# ============================================================
# 13. SI/FN OVERALL FEATURES
# ============================================================

print("\n[13] Creating overall SI/FN features...")


features["si_fn_true"] = (
    features["si_fn_true_count"] > 0
)

features["si_fn_false_count"] = (
    features["raw_row_count"]
    - features["si_fn_true_count"]
)


features["si_fn_true_ratio"] = (
    features["si_fn_true_count"]
    / features["raw_row_count"]
)


# ============================================================
# 14. NORMALIZE BOOLEAN TO INTEGER
# ============================================================

boolean_output_columns = [
    "is_directory",
    "has_ads",
    "is_ads",
    "has_windows_name",
    "has_posix_name",
    "has_dos_name",
    "si_fn_true",
]

for col in boolean_output_columns:

    if col in features.columns:

        features[col] = (
            features[col]
            .fillna(False)
            .astype(int)
        )


# ============================================================
# 15. SORT
# ============================================================

features = features.sort_values(
    "EntryNumber"
).reset_index(
    drop=True
)


# ============================================================
# 16. QUALITY CHECK
# ============================================================

print("\n[14] QUALITY CHECK")
print("-" * 70)

print(
    f"Raw MFT rows       : "
    f"{len(df):,}"
)

print(
    f"Aggregated records : "
    f"{len(features):,}"
)

print(
    f"Unique file_id     : "
    f"{features['file_id'].nunique():,}"
)

print(
    f"Final columns      : "
    f"{len(features.columns)}"
)

print(
    f"Duplicate file_id  : "
    f"{features['file_id'].duplicated().sum():,}"
)


# ============================================================
# 17. TARGET CHECK
# ============================================================

target_name = (
    "NewFileTime_SI_C_Manipulation.dll"
)

target_rows = df[
    df["FileName"].astype(str).str.lower()
    == target_name.lower()
]

print("\n[15] TARGET CHECK")
print("-" * 70)

print(
    f"Target filename : {target_name}"
)

print(
    f"Raw target rows : {len(target_rows)}"
)

if len(target_rows) > 0:

    target_ids = (
        target_rows["file_id"]
        .unique()
    )

    print(
        f"Target MFT records : "
        f"{len(target_ids)}"
    )

    target_features = features[
        features["file_id"].isin(
            target_ids
        )
    ]

    print("\nTarget aggregated record:")

    print(
        target_features.to_string(
            index=False
        )
    )


# ============================================================
# 18. SAVE
# ============================================================

print("\n[16] Saving output...")

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nOutput saved to:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("SCRIPT 26 SELESAI")
print("=" * 70)