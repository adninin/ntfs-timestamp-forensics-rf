from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SCRIPT 31
# BUILD GROUND TRUTH FOR ALL 12 PERFORMANCE SCENARIOS
# ============================================================


# ============================================================
# 1. BASE DIRECTORY
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)


# ============================================================
# 2. RAW MFT
# ============================================================

RAW_MFT_DIR = (
    BASE_DIR
    / "output"
    / "mft_scenarios"
)


# ============================================================
# 3. AGGREGATED MFT FEATURES
# ============================================================

MFT_FEATURE_DIR = (
    BASE_DIR
    / "output"
    / "mft_record_features_all"
)


# ============================================================
# 4. INTEGRATED DATASET
# ============================================================

INTEGRATED_DIR = (
    BASE_DIR
    / "output"
    / "integrated_dataset_all"
)


# ============================================================
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "ground_truth_all"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. SCENARIO MAPPING
# ============================================================

SCENARIOS = {

    1: {
        "target_filename":
            "NewFileTime_SI_C_Manipulation.dll",
        "method":
            "NewFileTime",
        "api":
            "SetFileTime() API",
        "manipulation":
            "$SI-C",
    },

    2: {
        "target_filename":
            "NewFileTime_SI_M_Manipulation.dll",
        "method":
            "NewFileTime",
        "api":
            "SetFileTime() API",
        "manipulation":
            "$SI-M",
    },

    3: {
        "target_filename":
            "NewFileTime_SI_MAC_Manipulation.dll",
        "method":
            "NewFileTime",
        "api":
            "SetFileTime() API",
        "manipulation":
            "$SI-MAC",
    },

    4: {
        "target_filename":
            "PowerShell_SI_C_Manipulation.dll",
        "method":
            "PowerShell",
        "api":
            "Get-Item Cmdlet",
        "manipulation":
            "$SI-C",
    },

    5: {
        "target_filename":
            "PowerShell_SI_M_Manipulation.dll",
        "method":
            "PowerShell",
        "api":
            "Get-Item Cmdlet",
        "manipulation":
            "$SI-M",
    },

    6: {
        "target_filename":
            "PowerShell_SI_MAC_Manipulation.dll",
        "method":
            "PowerShell",
        "api":
            "Get-Item Cmdlet",
        "manipulation":
            "$SI-MAC",
    },

    7: {
        "target_filename":
            "nTimestomp_SI_C_Manipulation.dll",
        "method":
            "nTimestomp",
        "api":
            "NtSetInformationFile() API",
        "manipulation":
            "$SI-C",
    },

    8: {
        "target_filename":
            "nTimestomp_SI_M_Manipulation.dll",
        "method":
            "nTimestomp",
        "api":
            "NtSetInformationFile() API",
        "manipulation":
            "$SI-M",
    },

    9: {
        "target_filename":
            "nTimestomp_SI_MACE_Manipulation.dll",
        "method":
            "nTimestomp",
        "api":
            "NtSetInformationFile() API",
        "manipulation":
            "$SI-MACE",
    },

    10: {
        "target_filename":
            "FileSystemTunneling_SI_C_Change.dll",
        "method":
            "File System Tunneling",
        "api":
            "File System Tunneling",
        "manipulation":
            "$SI-C",
    },

    11: {
        "target_filename":
            "SetMACE_SI_MACE_Copy_Manipulation.dll",
        "method":
            "SetMACE",
        "api":
            "NtSetInformationFile() API",
        "manipulation":
            "$SI-MACE",
    },

    12: {
        "target_filename":
            "SetMACE_SI_FN_MACE_Manipulation.dll",
        "method":
            "SetMACE",
        "api":
            "NtSetInformationFile() API",
        "manipulation":
            "$SI-MACE + $FN-MACE",
    },
}


# ============================================================
# 7. HELPER
# ============================================================

def normalize_filename(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ============================================================
# 8. LOAD RAW MFT
# ============================================================

def load_raw_mft(scenario):

    path = (
        RAW_MFT_DIR
        / f"Scenario_{scenario:02d}"
        / f"Scenario_{scenario:02d}_MFT.csv"
    )

    print("\nRaw MFT:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nRaw MFT tidak ditemukan:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


# ============================================================
# 9. LOAD MFT FEATURES
# ============================================================

def load_mft_features(scenario):

    path = (
        MFT_FEATURE_DIR
        / f"Scenario_{scenario:02d}_"
          f"MFT_record_features.csv"
    )

    print("\nMFT record features:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nMFT record features tidak ditemukan:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


# ============================================================
# 10. LOAD INTEGRATED DATASET
# ============================================================

def load_integrated(scenario):

    path = (
        INTEGRATED_DIR
        / f"Scenario_{scenario:02d}_integrated.csv"
    )

    print("\nIntegrated dataset:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nIntegrated dataset tidak ditemukan:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


# ============================================================
# 11. CHECK COLUMNS
# ============================================================

def check_columns(
    df,
    columns,
    dataset_name
):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:

        raise KeyError(
            f"\nKolom berikut tidak ditemukan "
            f"pada {dataset_name}:\n"
            +
            "\n".join(
                f"  - {col}"
                for col in missing
            )
        )


# ============================================================
# 12. SUMMARY
# ============================================================

summary_rows = []


# ============================================================
# 13. PROCESS SCENARIOS
# ============================================================

for scenario, meta in SCENARIOS.items():

    print("\n")
    print("=" * 90)
    print(
        f"SCENARIO {scenario:02d}"
    )
    print("=" * 90)

    target_filename = (
        meta["target_filename"]
    )

    method = (
        meta["method"]
    )

    api = (
        meta["api"]
    )

    manipulation = (
        meta["manipulation"]
    )

    print(
        f"Target       : {target_filename}"
    )

    print(
        f"Method       : {method}"
    )

    print(
        f"API          : {api}"
    )

    print(
        f"Manipulation : {manipulation}"
    )


    # ========================================================
    # LOAD
    # ========================================================

    raw_mft = load_raw_mft(
        scenario
    )

    mft_features = load_mft_features(
        scenario
    )

    integrated = load_integrated(
        scenario
    )


    print(
        f"\nRaw MFT rows       : "
        f"{len(raw_mft):,}"
    )

    print(
        f"MFT records        : "
        f"{len(mft_features):,}"
    )

    print(
        f"Integrated records : "
        f"{len(integrated):,}"
    )


    # ========================================================
    # CHECK
    # ========================================================

    check_columns(
        raw_mft,
        [
            "EntryNumber",
            "SequenceNumber",
            "FileName"
        ],
        f"RAW MFT Scenario {scenario:02d}"
    )

    check_columns(
        mft_features,
        [
            "file_id"
        ],
        f"MFT features Scenario {scenario:02d}"
    )

    check_columns(
        integrated,
        [
            "file_id"
        ],
        f"Integrated Scenario {scenario:02d}"
    )


    # ========================================================
    # BUILD FILE ID
    # ========================================================

    raw_mft["file_id"] = (

        raw_mft["EntryNumber"]
        .astype(str)

        + "-"

        + raw_mft["SequenceNumber"]
        .astype(str)
    )


    # ========================================================
    # NORMALIZE FILENAME
    # ========================================================

    target_normalized = (
        normalize_filename(
            target_filename
        )
    )

    raw_mft["_filename_normalized"] = (

        raw_mft["FileName"]
        .map(
            normalize_filename
        )
    )


    # ========================================================
    # FIND TARGET
    # ========================================================

    target_raw = raw_mft[
        raw_mft["_filename_normalized"]
        == target_normalized
    ].copy()


    print(
        "\nTarget raw MFT rows found:"
        f" {len(target_raw):,}"
    )


    # ========================================================
    # TARGET NOT FOUND
    # ========================================================

    if target_raw.empty:

        print(
            "WARNING: Target tidak ditemukan."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "method":
                method,

            "api":
                api,

            "manipulation":
                manipulation,

            "raw_target_rows":
                0,

            "target_mft_records":
                0,

            "target_integrated_records":
                0,

            "background_records":
                len(integrated),

            "target_file_ids":
                "",

            "status":
                "TARGET_NOT_FOUND"
        })

        continue


    # ========================================================
    # GET TARGET FILE IDS
    # ========================================================

    target_file_ids = (

        target_raw["file_id"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()
    )


    print(
        "\nTarget file_id:"
    )

    for file_id in target_file_ids:

        print(
            f"  - {file_id}"
        )


    # ========================================================
    # TARGET RAW DETAILS
    # ========================================================

    raw_display_columns = [

        "EntryNumber",

        "SequenceNumber",

        "file_id",

        "FileName",

        "Extension",

        "FileSize",

        "ReferenceCount",

        "SI<FN",

        "NameType",

        "Created0x10",

        "Created0x30",

        "LastModified0x10",

        "LastModified0x30",

        "LastRecordChange0x10",

        "LastRecordChange0x30",

        "LastAccess0x10",

        "LastAccess0x30",

        "UpdateSequenceNumber",

        "LogfileSequenceNumber",
    ]


    raw_display_columns = [

        col
        for col in raw_display_columns
        if col in target_raw.columns
    ]


    print(
        "\nTarget raw MFT details:"
    )

    print(
        target_raw[
            raw_display_columns
        ].to_string(index=False)
    )


    # ========================================================
    # NORMALIZE FILE ID
    # ========================================================

    mft_features["file_id"] = (

        mft_features["file_id"]
        .astype(str)
    )

    integrated["file_id"] = (

        integrated["file_id"]
        .astype(str)
    )


    # ========================================================
    # TARGET MFT FEATURES
    # ========================================================

    target_mft_records = (

        mft_features[
            mft_features["file_id"]
            .isin(target_file_ids)
        ]
        .copy()
    )


    # ========================================================
    # TARGET INTEGRATED
    # ========================================================

    integrated_target = (

        integrated[
            integrated["file_id"]
            .isin(target_file_ids)
        ]
        .copy()
    )


    print(
        "\nTarget MFT records:"
        f" {len(target_mft_records):,}"
    )

    print(
        "Target integrated records:"
        f" {len(integrated_target):,}"
    )


    # ========================================================
    # LABEL
    # ========================================================

    target_id_set = set(
        target_file_ids
    )


    integrated["label"] = (

        integrated["file_id"]
        .isin(target_id_set)
        .astype(int)
    )


    # ========================================================
    # METADATA
    # ========================================================

    integrated["scenario"] = (
        scenario
    )

    integrated["target_filename"] = (
        target_filename
    )

    integrated["method"] = (
        method
    )

    integrated["api"] = (
        api
    )

    integrated["manipulation"] = (
        manipulation
    )


    # ========================================================
    # LABEL SOURCE
    # ========================================================

    integrated["label_source"] = np.where(

        integrated["label"] == 1,

        "experimental_target_identity",

        "background_non_target"
    )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    positive_count = int(

        (
            integrated["label"] == 1
        ).sum()
    )

    negative_count = int(

        (
            integrated["label"] == 0
        ).sum()
    )


    print(
        "\nLabel distribution:"
    )

    print(
        f"  label 1 target     : "
        f"{positive_count:,}"
    )

    print(
        f"  label 0 background : "
        f"{negative_count:,}"
    )


    # ========================================================
    # SHOW TARGET INTEGRATED
    # ========================================================

    target_columns = [

        "file_id",

        "EntryNumber",

        "SequenceNumber",

        "FileName",

        "FileSize",

        "si_fn_true",

        "raw_row_count",

        "filename_count",

        "unique_filename_count",

        "parent_count",

        "parent_sequence_count",

        "parent_path_count",

        "name_type_count",

        "has_windows_name",

        "has_posix_name",

        "abs_delta_creation_seconds",

        "abs_delta_modified_seconds",

        "abs_delta_record_change_seconds",

        "abs_delta_access_seconds",

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


    target_columns = [

        col
        for col in target_columns
        if col in integrated_target.columns
    ]


    print(
        "\nIntegrated target details:"
    )

    print(
        integrated_target[
            target_columns
        ].to_string(index=False)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if (
        len(target_mft_records)
        == len(target_file_ids)

        and

        len(integrated_target)
        == len(target_file_ids)
    ):

        status = "OK"

    else:

        status = "CHECK_REQUIRED"


    print(
        "\nScenario status:"
    )

    print(
        status
    )


    # ========================================================
    # SAVE FULL DATASET
    # ========================================================

    full_output = (

        OUTPUT_DIR
        / f"Scenario_{scenario:02d}_"
          f"ground_truth.csv"
    )


    integrated.to_csv(
        full_output,
        index=False
    )


    print(
        "\nGround truth saved:"
    )

    print(
        full_output
    )


    # ========================================================
    # SAVE TARGET ONLY
    # ========================================================

    target_output = (

        OUTPUT_DIR
        / f"Scenario_{scenario:02d}_"
          f"target_records.csv"
    )


    integrated_target.to_csv(
        target_output,
        index=False
    )


    # ========================================================
    # SAVE RAW TARGET
    # ========================================================

    raw_target_output = (

        OUTPUT_DIR
        / f"Scenario_{scenario:02d}_"
          f"target_raw_mft.csv"
    )


    target_raw.drop(
        columns=[
            "_filename_normalized"
        ],
        errors="ignore"
    ).to_csv(
        raw_target_output,
        index=False
    )


    # ========================================================
    # SAVE TARGET MFT FEATURES
    # ========================================================

    target_mft_output = (

        OUTPUT_DIR
        / f"Scenario_{scenario:02d}_"
          f"target_mft_features.csv"
    )


    target_mft_records.to_csv(
        target_mft_output,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append({

        "scenario":
            scenario,

        "target_filename":
            target_filename,

        "method":
            method,

        "api":
            api,

        "manipulation":
            manipulation,

        "raw_target_rows":
            len(target_raw),

        "target_mft_records":
            len(target_mft_records),

        "target_integrated_records":
            len(integrated_target),

        "background_records":
            negative_count,

        "target_file_ids":
            ";".join(
                target_file_ids
            ),

        "status":
            status
    })


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (

    OUTPUT_DIR
    / "ground_truth_all_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 15. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print(
    "FINAL GROUND TRUTH SUMMARY"
)
print("=" * 90)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 16. STATUS
# ============================================================

print("\n")
print("=" * 90)
print(
    "FINAL STATUS"
)
print("=" * 90)


ok_count = int(

    (
        summary_df["status"]
        == "OK"
    ).sum()
)


check_count = int(

    (
        summary_df["status"]
        == "CHECK_REQUIRED"
    ).sum()
)


not_found_count = int(

    (
        summary_df["status"]
        == "TARGET_NOT_FOUND"
    ).sum()
)


print(
    f"OK               : "
    f"{ok_count}/12"
)

print(
    f"CHECK_REQUIRED   : "
    f"{check_count}/12"
)

print(
    f"TARGET_NOT_FOUND : "
    f"{not_found_count}/12"
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nScript 31 selesai."
)