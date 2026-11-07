from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

GROUND_TRUTH_CSV = (
    PROJECT_DIR
    / "output"
    / "ground_truth"
    / "Scenario_01_ground_truth_full.csv"
)

FORENSIC_FEATURE_CSV = (
    PROJECT_DIR
    / "output"
    / "mft_scenario_01"
    / "Scenario_01_forensic_timestamp_features.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "negative_candidates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGET
# ============================================================

TARGET_FILENAME = (
    "NewFileTime_SI_C_Manipulation.dll"
)


# ============================================================
# START
# ============================================================

print("=" * 100)
print("SCRIPT 23 - NEGATIVE CANDIDATE ANALYSIS")
print("=" * 100)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLOAD GROUND TRUTH")
print("=" * 100)

df = pd.read_csv(
    GROUND_TRUTH_CSV,
    low_memory=False
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# LOAD FORENSIC FEATURES
# ============================================================

print("\nLOAD FORENSIC TIMESTAMP FEATURES")
print("=" * 100)

features = pd.read_csv(
    FORENSIC_FEATURE_CSV,
    low_memory=False
)

print(f"Rows: {len(features):,}")
print(f"Columns: {len(features.columns)}")


# ============================================================
# CREATE FILE ID
# ============================================================

for data in [df, features]:

    data["EntryNumber"] = pd.to_numeric(
        data["EntryNumber"],
        errors="coerce"
    )

    data["SequenceNumber"] = pd.to_numeric(
        data["SequenceNumber"],
        errors="coerce"
    )

    data["file_id"] = (
        data["EntryNumber"].astype("Int64").astype(str)
        + "-"
        + data["SequenceNumber"].astype("Int64").astype(str)
    )


# ============================================================
# MERGE GROUND TRUTH + FORENSIC FEATURES
# ============================================================

feature_columns = [
    "file_id",
    "max_abs_timestamp_delta",
    "mean_abs_timestamp_delta",
    "si_fn_exact_count",
    "special_timestamp_count",
    "abs_delta_creation_seconds",
    "abs_delta_modified_seconds",
    "abs_delta_record_change_seconds",
    "abs_delta_access_seconds",
    "year_delta_creation",
    "year_delta_modified",
    "year_delta_record_change",
    "year_delta_access",
]

feature_columns = [
    c for c in feature_columns
    if c in features.columns
]

features_small = features[
    feature_columns
].drop_duplicates(
    subset=["file_id"]
)

df = df.merge(
    features_small,
    on="file_id",
    how="left",
    suffixes=("", "_feature")
)

print(
    f"\nAfter merge: {len(df):,} rows"
)


# ============================================================
# FIND TARGET
# ============================================================

target_rows = df[
    df["FileName"]
    .astype(str)
    .str.strip()
    .str.lower()
    == TARGET_FILENAME.lower()
].copy()


if len(target_rows) == 0:

    raise RuntimeError(
        "Target tidak ditemukan."
    )


target = target_rows.iloc[0]


target_entry = target["EntryNumber"]
target_seq = target["SequenceNumber"]
target_size = target["FileSize"]
target_extension = str(
    target["Extension"]
).strip().lower()


print("\nTARGET")
print("=" * 100)

print(
    f"FileName       : {target['FileName']}"
)

print(
    f"EntryNumber    : {target_entry}"
)

print(
    f"SequenceNumber : {target_seq}"
)

print(
    f"File ID        : {target['file_id']}"
)

print(
    f"Extension      : {target_extension}"
)

print(
    f"FileSize       : {target_size:,}"
)


# ============================================================
# REMOVE TARGET
# ============================================================

candidates = df[
    ~(
        (df["EntryNumber"] == target_entry)
        &
        (df["SequenceNumber"] == target_seq)
    )
].copy()


# ============================================================
# NORMALIZE
# ============================================================

candidates["extension_norm"] = (
    candidates["Extension"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# SIZE FEATURES
# ============================================================

candidates["size_difference"] = (
    pd.to_numeric(
        candidates["FileSize"],
        errors="coerce"
    )
    - target_size
)

candidates["abs_size_difference"] = (
    candidates["size_difference"]
    .abs()
)

candidates["size_difference_percent"] = (
    candidates["abs_size_difference"]
    / target_size
    * 100
)


# ============================================================
# EXTENSION MATCH
# ============================================================

candidates["same_extension"] = (
    candidates["extension_norm"]
    == target_extension
)


# ============================================================
# CANDIDATE GROUPS
# ============================================================

same_extension = candidates[
    candidates["same_extension"]
].copy()

exact_size = candidates[
    candidates["same_extension"]
    &
    (
        candidates["FileSize"]
        == target_size
    )
].copy()

within_5 = candidates[
    candidates["same_extension"]
    &
    (
        candidates["size_difference_percent"]
        <= 5
    )
].copy()

within_10 = candidates[
    candidates["same_extension"]
    &
    (
        candidates["size_difference_percent"]
        <= 10
    )
].copy()


# ============================================================
# COUNTS
# ============================================================

print("\n")
print("=" * 100)
print("CANDIDATE COUNTS")
print("=" * 100)

print(
    f"Same extension (.dll)           : "
    f"{len(same_extension):,}"
)

print(
    f"Same extension + exact size     : "
    f"{len(exact_size):,}"
)

print(
    f"Same extension + size ±5%       : "
    f"{len(within_5):,}"
)

print(
    f"Same extension + size ±10%      : "
    f"{len(within_10):,}"
)


# ============================================================
# DISPLAY FUNCTION
# ============================================================

def show_candidates(group, title):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    if len(group) == 0:

        print("Tidak ada kandidat.")

        return

    # Sort BEFORE selecting display columns
    result = (
        group
        .sort_values(
            by="abs_size_difference",
            ascending=True
        )
        .head(20)
    )

    display_columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "Extension",
        "FileSize",
        "size_difference",
        "abs_size_difference",
        "size_difference_percent",
        "SI<FN",
        "max_abs_timestamp_delta",
        "mean_abs_timestamp_delta",
        "si_fn_exact_count",
        "special_timestamp_count",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in result.columns
    ]

    print(
        result[
            display_columns
        ].to_string(index=False)
    )

# ============================================================
# DISPLAY
# ============================================================

show_candidates(
    exact_size,
    "EXACT SIZE + SAME EXTENSION"
)

show_candidates(
    within_5,
    "WITHIN ±5% SIZE + SAME EXTENSION"
)

show_candidates(
    within_10,
    "WITHIN ±10% SIZE + SAME EXTENSION"
)


# ============================================================
# TARGET FEATURES
# ============================================================

print("\n")
print("=" * 100)
print("TARGET FORENSIC FEATURES")
print("=" * 100)

target_feature_columns = [
    "file_id",
    "FileName",
    "max_abs_timestamp_delta",
    "mean_abs_timestamp_delta",
    "si_fn_exact_count",
    "special_timestamp_count",
    "abs_delta_creation_seconds",
    "abs_delta_modified_seconds",
    "abs_delta_record_change_seconds",
    "abs_delta_access_seconds",
    "year_delta_creation",
    "year_delta_modified",
    "year_delta_record_change",
    "year_delta_access",
]

target_feature_columns = [
    c
    for c in target_feature_columns
    if c in target.columns
]

print(
    target[target_feature_columns]
    .to_string()
)


# ============================================================
# COMPARE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 100)
print("TIMESTAMP COMPARISON")
print("=" * 100)


comparison_groups = [
    ("TARGET", target_rows),
    ("EXACT_SIZE", exact_size),
    ("WITHIN_5_PERCENT", within_5),
    ("WITHIN_10_PERCENT", within_10),
]


for name, group in comparison_groups:

    print("\n" + name)

    if len(group) == 0:

        print("No records")

        continue

    if "max_abs_timestamp_delta" not in group.columns:

        print(
            "max_abs_timestamp_delta tidak tersedia"
        )

        continue

    values = pd.to_numeric(
        group[
            "max_abs_timestamp_delta"
        ],
        errors="coerce"
    ).dropna()

    if len(values) == 0:

        print("Tidak ada nilai valid")

        continue

    print(
        f"count  : {len(values):,}"
    )

    print(
        f"median : {values.median():,.3f}"
    )

    print(
        f"p90    : {values.quantile(0.90):,.3f}"
    )

    print(
        f"p95    : {values.quantile(0.95):,.3f}"
    )

    print(
        f"p99    : {values.quantile(0.99):,.3f}"
    )

    print(
        f"max    : {values.max():,.3f}"
    )


# ============================================================
# SAVE
# ============================================================

exact_output = (
    OUTPUT_DIR
    / "Scenario_01_exact_size_candidates.csv"
)

within5_output = (
    OUTPUT_DIR
    / "Scenario_01_within5_candidates.csv"
)

within10_output = (
    OUTPUT_DIR
    / "Scenario_01_within10_candidates.csv"
)

same_ext_output = (
    OUTPUT_DIR
    / "Scenario_01_same_extension_candidates.csv"
)


exact_size.to_csv(
    exact_output,
    index=False
)

within_5.to_csv(
    within5_output,
    index=False
)

within_10.to_csv(
    within10_output,
    index=False
)

same_extension.to_csv(
    same_ext_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "target_filename": TARGET_FILENAME,
            "target_entry": target_entry,
            "target_sequence": target_seq,
            "target_size": target_size,
            "target_extension": target_extension,
            "same_extension_count": len(same_extension),
            "exact_size_count": len(exact_size),
            "within_5_percent_count": len(within_5),
            "within_10_percent_count": len(within_10),
        }
    ]
)

summary_output = (
    OUTPUT_DIR
    / "Scenario_01_negative_candidate_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n")
print("=" * 100)
print("SCRIPT 23 SELESAI")
print("=" * 100)