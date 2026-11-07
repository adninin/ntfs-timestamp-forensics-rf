import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_BASE = (
    BASE_DIR
    / "output"
    / "mft_scenarios"
)

OUTPUT_BASE = (
    BASE_DIR
    / "output"
    / "mft_record_features_all"
)

OUTPUT_BASE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCTION: PROCESS ONE SCENARIO
# ============================================================

def process_scenario(scenario_number):

    scenario = f"Scenario_{scenario_number:02d}"

    print("\n" + "=" * 70)
    print(f"{scenario}")
    print("=" * 70)

    input_file = (
        INPUT_BASE
        / scenario
        / f"{scenario}_MFT.csv"
    )

    output_file = (
        OUTPUT_BASE
        / f"{scenario}_MFT_record_features.csv"
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_file.exists():

        print(
            f"[FAILED] Input tidak ditemukan:"
        )

        print(input_file)

        return {
            "scenario": scenario,
            "status": "input_not_found"
        }

    # --------------------------------------------------------
    # Skip existing
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\n[1] Loading MFT...")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Identifier
    # --------------------------------------------------------

    print(
        "\n[2] Creating MFT record ID..."
    )

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

    unique_records = (
        df["file_id"].nunique()
    )

    print(
        f"Unique MFT records : "
        f"{unique_records:,}"
    )

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    print(
        "\n[3] Converting timestamps..."
    )

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

    # --------------------------------------------------------
    # Timestamp delta
    # --------------------------------------------------------

    print(
        "\n[4] Creating timestamp delta features..."
    )

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

        df[
            f"delta_{name}_seconds"
        ] = delta

        df[
            f"abs_delta_{name}_seconds"
        ] = delta.abs()

        df[
            f"si_fn_exact_{name}"
        ] = delta == 0

        df[
            f"si_before_fn_{name}"
        ] = delta < 0

        df[
            f"si_after_fn_{name}"
        ] = delta > 0

    # --------------------------------------------------------
    # Boolean conversion
    # --------------------------------------------------------

    print(
        "\n[5] Normalizing boolean fields..."
    )

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

    # --------------------------------------------------------
    # Group
    # --------------------------------------------------------

    print(
        "\n[6] Creating groupby..."
    )

    group = df.groupby(
        "file_id",
        sort=False
    )

    # --------------------------------------------------------
    # Core aggregation
    # --------------------------------------------------------

    print(
        "\n[7] Aggregating core MFT features..."
    )

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

    # --------------------------------------------------------
    # NameType
    # --------------------------------------------------------

    print(
        "\n[8] Creating NameType features..."
    )

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

    # --------------------------------------------------------
    # Timestamp aggregation
    # --------------------------------------------------------

    print(
        "\n[9] Aggregating timestamp features..."
    )

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

    # --------------------------------------------------------
    # Timestamp range
    # --------------------------------------------------------

    print(
        "\n[10] Creating timestamp ranges..."
    )

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

        features[
            f"abs_delta_{name}_range"
        ] = (
            features[max_col]
            - features[min_col]
        )

    # --------------------------------------------------------
    # SI/FN relationships
    # --------------------------------------------------------

    print(
        "\n[11] Aggregating SI/FN relationships..."
    )

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

    # --------------------------------------------------------
    # Timestamp years
    # --------------------------------------------------------

    print(
        "\n[12] Creating timestamp year features..."
    )

    for col in timestamp_columns:

        year_col = f"{col}_year"

        df[year_col] = (
            df[col].dt.year
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

    # --------------------------------------------------------
    # SI/FN overall
    # --------------------------------------------------------

    print(
        "\n[13] Creating overall SI/FN features..."
    )

    features["si_fn_true"] = (
        features["si_fn_true_count"] > 0
    )

    features["si_fn_false_count"] = (
        features["raw_row_count"]
        -
        features["si_fn_true_count"]
    )

    features["si_fn_true_ratio"] = (
        features["si_fn_true_count"]
        /
        features["raw_row_count"]
    )

    # --------------------------------------------------------
    # Boolean output
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    features = (
        features
        .sort_values("EntryNumber")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    print(
        "\n[14] QUALITY CHECK"
    )

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[15] Saving..."
    )

    features.to_csv(
        output_file,
        index=False
    )

    print(
        f"Output:"
    )

    print(
        output_file
    )

    return {
        "scenario": scenario,
        "status": "success",
        "raw_rows": len(df),
        "mft_records": len(features),
        "columns": len(features.columns),
        "duplicate_file_id":
            int(
                features["file_id"]
                .duplicated()
                .sum()
            ),
        "output": str(output_file)
    }


# ============================================================
# MAIN
# ============================================================

results = []


for scenario_number in range(1, 13):

    try:

        result = process_scenario(
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
print("BATCH MFT FEATURE SUMMARY")
print("=" * 70)


summary = pd.DataFrame(
    results
)


if len(summary) > 0:

    print(
        summary[
            [
                "scenario",
                "status"
            ]
        ].to_string(
            index=False
        )
    )


summary_file = (
    OUTPUT_BASE
    / "batch_mft_features_summary.csv"
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


print("\n" + "=" * 70)
print("SCRIPT 29 SELESAI")
print("=" * 70)