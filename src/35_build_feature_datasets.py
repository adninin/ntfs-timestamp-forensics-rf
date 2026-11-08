from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_PATH = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
    / "all_scenarios_ml_dataset.csv"
)

AUDIT_PATH = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
    / "feature_audit_full.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "ml_dataset_all"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METADATA COLUMNS
# ============================================================

METADATA_COLUMNS = {
    "scenario",
    "file_id",
    "label",
    "negative_type",
    "target_filename",
}


# ============================================================
# IDENTIFIER / LEAKAGE COLUMNS
# ============================================================

EXCLUDE_IDENTIFIER_COLUMNS = {
    "EntryNumber",
    "SequenceNumber",
}


# ============================================================
# FEATURES YANG AKAN DIKELOMPOKKAN
# ============================================================

def classify_feature(column):

    name = column.lower()

    # --------------------------------------------------------
    # USN
    # --------------------------------------------------------

    if name.startswith("usn_"):

        return "USN"

    # --------------------------------------------------------
    # Timestamp / SI-FN
    # --------------------------------------------------------

    timestamp_keywords = [
        "delta_creation",
        "delta_modified",
        "delta_record_change",
        "delta_access",
        "created0x",
        "lastmodified0x",
        "lastrecordchange0x",
        "lastaccess0x",
        "si_fn",
        "si_before_fn",
        "si_after_fn",
    ]

    if any(
        keyword in name
        for keyword in timestamp_keywords
    ):

        return "MFT_TIMESTAMP"

    # --------------------------------------------------------
    # MFT structural
    # --------------------------------------------------------

    mft_keywords = [
        "file_size",
        "raw_row_count",
        "filename_count",
        "unique_filename_count",
        "parent_count",
        "parent_sequence_count",
        "parent_path_count",
        "name_type_count",
        "extension_count",
        "reference_count",
        "is_directory",
        "has_ads",
        "is_ads",
        "in_use_count",
        "copied_count",
        "has_windows_name",
        "has_posix_name",
        "has_dos_name",
    ]

    if any(
        keyword in name
        for keyword in mft_keywords
    ):

        return "MFT_STRUCTURE"

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    return "OTHER"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SCRIPT 35 - FEATURE SELECTION & DATASET V2")
    print("=" * 80)

    print()
    print(f"Input dataset : {INPUT_PATH}")
    print(f"Audit file    : {AUDIT_PATH}")
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{INPUT_PATH}"
        )

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Audit file tidak ditemukan:\n{AUDIT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    audit = pd.read_csv(
        AUDIT_PATH,
        low_memory=False
    )

    print("[LOAD]")
    print(f"Dataset rows    : {len(df):,}")
    print(f"Dataset columns : {len(df.columns)}")
    print(f"Audit rows      : {len(audit):,}")
    print()

    # --------------------------------------------------------
    # AUDIT COLUMN VALIDATION
    # --------------------------------------------------------

    if "column" not in audit.columns:
        raise ValueError(
            "Kolom 'column' tidak ditemukan "
            "di feature_audit_full.csv"
        )

    # --------------------------------------------------------
    # CANDIDATE FEATURES
    # --------------------------------------------------------

    audit = audit.copy()

    # Hanya numeric
    audit = audit[
        audit["numeric_possible"] == True
    ].copy()

    # Buang constant
    audit = audit[
        audit["constant"] == False
    ].copy()

    candidate_features = (
        audit["column"]
        .tolist()
    )

    # --------------------------------------------------------
    # Remove metadata
    # --------------------------------------------------------

    candidate_features = [
        c
        for c in candidate_features
        if c not in METADATA_COLUMNS
    ]

    # --------------------------------------------------------
    # Remove identifiers
    # --------------------------------------------------------

    candidate_features = [
        c
        for c in candidate_features
        if c not in EXCLUDE_IDENTIFIER_COLUMNS
    ]

    # --------------------------------------------------------
    # Pastikan kolom masih ada
    # --------------------------------------------------------

    candidate_features = [
        c
        for c in candidate_features
        if c in df.columns
    ]

    # --------------------------------------------------------
    # Buat classification
    # --------------------------------------------------------

    feature_groups = []

    for col in candidate_features:

        group = classify_feature(
            col
        )

        feature_groups.append({
            "feature": col,
            "group": group,
        })

    feature_groups_df = pd.DataFrame(
        feature_groups
    )

    print("=" * 80)
    print("1. FEATURE GROUPS")
    print("=" * 80)

    print(
        feature_groups_df[
            "group"
        ]
        .value_counts()
    )

    print()

    print(
        feature_groups_df
        .sort_values(
            ["group", "feature"]
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save feature groups
    # --------------------------------------------------------

    feature_groups_path = (
        OUTPUT_DIR
        / "feature_groups_v2.csv"
    )

    feature_groups_df.to_csv(
        feature_groups_path,
        index=False
    )

    # ========================================================
    # FEATURE SETS
    # ========================================================

    mft_timestamp_features = (
        feature_groups_df.loc[
            feature_groups_df["group"]
            == "MFT_TIMESTAMP",
            "feature"
        ]
        .tolist()
    )

    mft_structure_features = (
        feature_groups_df.loc[
            feature_groups_df["group"]
            == "MFT_STRUCTURE",
            "feature"
        ]
        .tolist()
    )

    usn_features = (
        feature_groups_df.loc[
            feature_groups_df["group"]
            == "USN",
            "feature"
        ]
        .tolist()
    )

    other_features = (
        feature_groups_df.loc[
            feature_groups_df["group"]
            == "OTHER",
            "feature"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Model feature sets
    # --------------------------------------------------------

    # MFT-only:
    # timestamp + structural MFT
    mft_only_features = (
        mft_timestamp_features
        + mft_structure_features
    )

    # MFT + USN:
    # semua MFT + USN
    mft_usn_features = (
        mft_only_features
        + usn_features
    )

    # Remove duplicates preserving order
    mft_only_features = list(
        dict.fromkeys(
            mft_only_features
        )
    )

    mft_usn_features = list(
        dict.fromkeys(
            mft_usn_features
        )
    )

    # ========================================================
    # PRINT COUNTS
    # ========================================================

    print()
    print("=" * 80)
    print("2. FEATURE SET COUNTS")
    print("=" * 80)

    print(
        f"MFT timestamp features : "
        f"{len(mft_timestamp_features)}"
    )

    print(
        f"MFT structural features: "
        f"{len(mft_structure_features)}"
    )

    print(
        f"USN features            : "
        f"{len(usn_features)}"
    )

    print(
        f"Other features          : "
        f"{len(other_features)}"
    )

    print(
        f"MFT-only total          : "
        f"{len(mft_only_features)}"
    )

    print(
        f"MFT + USN total         : "
        f"{len(mft_usn_features)}"
    )

    # ========================================================
    # CREATE DATASETS
    # ========================================================

    metadata = [
        "scenario",
        "file_id",
        "label",
        "negative_type",
    ]

    # --------------------------------------------------------
    # MFT ONLY
    # --------------------------------------------------------

    mft_columns = (
        metadata
        + mft_only_features
    )

    mft_dataset = (
        df[mft_columns]
        .copy()
    )

    # --------------------------------------------------------
    # MFT + USN
    # --------------------------------------------------------

    mft_usn_columns = (
        metadata
        + mft_usn_features
    )

    mft_usn_dataset = (
        df[mft_usn_columns]
        .copy()
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_dataset(
        dataset,
        name
    ):

        print()
        print(
            "=" * 80
        )
        print(
            f"VALIDATION - {name}"
        )
        print(
            "=" * 80
        )

        print(
            f"Rows    : {len(dataset):,}"
        )

        print(
            f"Columns : {len(dataset.columns)}"
        )

        print()
        print(
            "Labels:"
        )

        print(
            dataset["label"]
            .value_counts()
            .sort_index()
        )

        # ----------------------------------------------------
        # Duplicate scenario/file_id
        # ----------------------------------------------------

        duplicates = (
            dataset
            .duplicated(
                subset=[
                    "scenario",
                    "file_id"
                ]
            )
            .sum()
        )

        print()
        print(
            f"Duplicate "
            f"(scenario,file_id): "
            f"{duplicates}"
        )

        if duplicates > 0:
            raise ValueError(
                f"{name}: duplicate record."
            )

        # ----------------------------------------------------
        # Missing summary
        # ----------------------------------------------------

        feature_cols = [
            c
            for c in dataset.columns
            if c not in metadata
        ]

        missing = (
            dataset[feature_cols]
            .isna()
            .mean()
            * 100
        )

        print()
        print(
            "Top missing features:"
        )

        print(
            missing
            .sort_values(
                ascending=False
            )
            .head(10)
            .to_string()
        )

        # ----------------------------------------------------
        # Positive count
        # ----------------------------------------------------

        positive = dataset[
            dataset["label"] == 1
        ]

        print()
        print(
            f"Positive records: "
            f"{len(positive)}"
        )

        # ----------------------------------------------------
        # Positive per scenario
        # ----------------------------------------------------

        pos_scenario = (
            positive
            .groupby("scenario")
            .size()
        )

        print(
            "Positive per scenario:"
        )

        print(
            pos_scenario
            .to_string()
        )

        if len(pos_scenario) != 12:
            raise ValueError(
                f"{name}: jumlah scenario "
                f"positive bukan 12."
            )

    validate_dataset(
        mft_dataset,
        "MFT-ONLY"
    )

    validate_dataset(
        mft_usn_dataset,
        "MFT + USN"
    )

    # ========================================================
    # SAVE DATASETS
    # ========================================================

    mft_path = (
        OUTPUT_DIR
        / "dataset_mft_only.csv"
    )

    mft_usn_path = (
        OUTPUT_DIR
        / "dataset_mft_usn.csv"
    )

    mft_dataset.to_csv(
        mft_path,
        index=False
    )

    mft_usn_dataset.to_csv(
        mft_usn_path,
        index=False
    )

    print()
    print("=" * 80)
    print("3. FILES SAVED")
    print("=" * 80)

    print(
        f"MFT-only:"
        f"\n{mft_path}"
    )

    print()
    print(
        f"MFT + USN:"
        f"\n{mft_usn_path}"
    )

    # ========================================================
    # FEATURE MANIFEST
    # ========================================================

    manifest_rows = []

    for col in mft_only_features:

        manifest_rows.append({
            "feature": col,
            "group": (
                feature_groups_df
                .loc[
                    feature_groups_df["feature"]
                    == col,
                    "group"
                ]
                .iloc[0]
            ),
            "included_mft_only": True,
            "included_mft_usn": True,
        })

    for col in usn_features:

        if col not in mft_only_features:

            manifest_rows.append({
                "feature": col,
                "group": "USN",
                "included_mft_only": False,
                "included_mft_usn": True,
            })

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        OUTPUT_DIR
        / "feature_manifest_v2.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    print()
    print(
        f"Feature manifest:"
        f"\n{manifest_path}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([
        {
            "dataset": "MFT_ONLY",
            "rows": len(mft_dataset),
            "columns_total": len(
                mft_dataset.columns
            ),
            "feature_count": len(
                mft_only_features
            ),
            "timestamp_features": len(
                mft_timestamp_features
            ),
            "mft_structure_features": len(
                mft_structure_features
            ),
            "usn_features": 0,
            "positive": int(
                (mft_dataset["label"] == 1)
                .sum()
            ),
            "negative": int(
                (mft_dataset["label"] == 0)
                .sum()
            ),
        },
        {
            "dataset": "MFT_USN",
            "rows": len(mft_usn_dataset),
            "columns_total": len(
                mft_usn_dataset.columns
            ),
            "feature_count": len(
                mft_usn_features
            ),
            "timestamp_features": len(
                mft_timestamp_features
            ),
            "mft_structure_features": len(
                mft_structure_features
            ),
            "usn_features": len(
                usn_features
            ),
            "positive": int(
                (mft_usn_dataset["label"] == 1)
                .sum()
            ),
            "negative": int(
                (mft_usn_dataset["label"] == 0)
                .sum()
            ),
        },
    ])

    summary_path = (
        OUTPUT_DIR
        / "feature_dataset_summary_v2.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print(
        f"Summary:"
        f"\n{summary_path}"
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 80)
    print("SCRIPT 35 SELESAI")
    print("=" * 80)

    print()
    print(
        "Dataset MFT-only dan MFT+USN "
        "sudah siap untuk tahap modeling."
    )

    print()
    print(
        "JANGAN training RF dulu."
    )

    print(
        "Kirim output terminal Script 35 "
        "supaya kita review jumlah fitur "
        "dan kelompok fiturnya sebelum "
        "masuk Script 36."
    )


if __name__ == "__main__":
    main()