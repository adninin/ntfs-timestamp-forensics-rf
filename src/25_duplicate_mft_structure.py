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
print("SCRIPT 25 - DUPLICATE MFT STRUCTURE ANALYSIS")
print("=" * 100)


# ============================================================
# LOAD
# ============================================================

print("\nLOAD MFT")
print("=" * 100)

df = pd.read_csv(
    MFT_CSV,
    low_memory=False
)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# FILE ID
# ============================================================

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


# ============================================================
# DUPLICATE IDS
# ============================================================

counts = (
    df["file_id"]
    .value_counts()
)

duplicate_ids = (
    counts[counts > 1]
)


print("\n")
print("=" * 100)
print("DUPLICATE SUMMARY")
print("=" * 100)

print(
    f"Unique file_id: {df['file_id'].nunique():,}"
)

print(
    f"Duplicated file_id: {len(duplicate_ids):,}"
)


# ============================================================
# SELECT TOP DUPLICATES
# ============================================================

top_ids = (
    duplicate_ids
    .head(10)
    .index
    .tolist()
)


# ============================================================
# ANALYZE EACH
# ============================================================

summary_rows = []


for file_id in top_ids:

    group = df[
        df["file_id"] == file_id
    ].copy()

    print("\n")
    print("=" * 100)
    print(f"FILE ID: {file_id}")
    print("=" * 100)

    print(
        f"Rows: {len(group)}"
    )

    # --------------------------------------------------------
    # BASIC
    # --------------------------------------------------------

    basic_columns = [
        "EntryNumber",
        "SequenceNumber",
        "FileName",
        "Extension",
        "FileSize",
        "ReferenceCount",
        "IsDirectory",
        "HasAds",
        "IsAds",
        "SI<FN",
        "NameType",
        "SourceFile",
    ]

    basic_columns = [
        c for c in basic_columns
        if c in group.columns
    ]

    print("\nBASIC STRUCTURE")

    print(
        group[basic_columns]
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # NAME TYPES
    # --------------------------------------------------------

    if "NameType" in group.columns:

        print("\nNameType:")

        print(
            group["NameType"]
            .value_counts(dropna=False)
            .to_string()
        )


    # --------------------------------------------------------
    # FILE NAMES
    # --------------------------------------------------------

    if "FileName" in group.columns:

        print("\nFileName:")

        print(
            group["FileName"]
            .value_counts(dropna=False)
            .to_string()
        )


    # --------------------------------------------------------
    # SI<FN
    # --------------------------------------------------------

    if "SI<FN" in group.columns:

        print("\nSI<FN:")

        print(
            group["SI<FN"]
            .value_counts(dropna=False)
            .to_string()
        )


    # --------------------------------------------------------
    # TIMESTAMP DIFFERENCES
    # --------------------------------------------------------

    timestamp_pairs = [
        ("Created0x10", "Created0x30"),
        ("LastModified0x10", "LastModified0x30"),
        ("LastRecordChange0x10", "LastRecordChange0x30"),
        ("LastAccess0x10", "LastAccess0x30"),
    ]

    for si_col, fn_col in timestamp_pairs:

        if (
            si_col in group.columns
            and fn_col in group.columns
        ):

            si = pd.to_datetime(
                group[si_col],
                errors="coerce"
            )

            fn = pd.to_datetime(
                group[fn_col],
                errors="coerce"
            )

            delta = (
                si - fn
            ).dt.total_seconds()

            print(
                f"\n{si_col} vs {fn_col}:"
            )

            print(
                delta.to_string(index=False)
            )


    # --------------------------------------------------------
    # DUPLICATE COMPARISON
    # --------------------------------------------------------

    print("\nCOLUMN VARIABILITY")

    variability = []

    for col in df.columns:

        unique_values = (
            group[col]
            .astype(str)
            .nunique(
                dropna=False
            )
        )

        if unique_values > 1:

            variability.append(
                {
                    "column": col,
                    "unique_values": unique_values,
                }
            )

    variability_df = pd.DataFrame(
        variability
    )

    if len(variability_df) > 0:

        print(
            variability_df
            .sort_values(
                "unique_values",
                ascending=False
            )
            .to_string(index=False)
        )

    else:

        print(
            "Semua kolom identik."
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows.append(
        {
            "file_id": file_id,
            "row_count": len(group),
            "unique_filename": group["FileName"].nunique()
                if "FileName" in group.columns
                else None,
            "unique_nametype": group["NameType"].nunique()
                if "NameType" in group.columns
                else None,
            "unique_sourcefile": group["SourceFile"].nunique()
                if "SourceFile" in group.columns
                else None,
            "unique_si_fn": group["SI<FN"].nunique()
                if "SI<FN" in group.columns
                else None,
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

output = (
    OUTPUT_DIR
    / "Scenario_01_duplicate_structure_summary.csv"
)

summary.to_csv(
    output,
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
    summary.to_string(index=False)
)

print(
    f"\nOutput:"
    f"\n{output}"
)

print("\n")
print("=" * 100)
print("SCRIPT 25 SELESAI")
print("=" * 100)