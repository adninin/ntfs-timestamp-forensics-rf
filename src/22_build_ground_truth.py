from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

MFT_DIR = PROJECT_DIR / "output" / "mft_scenario_01"

MFT_CSV = MFT_DIR / "Scenario_01_MFT.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "ground_truth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET MAPPING
# Berdasarkan 12 skenario eksperimen
# ============================================================

SCENARIOS = {
    1: {
        "target": "NewFileTime_SI_C_Manipulation.dll",
        "method": "NewFileTime",
        "api": "SetFileTime() API",
        "manipulation": "$SI-C",
    },
    2: {
        "target": "NewFileTime_SI_M_Manipulation.dll",
        "method": "NewFileTime",
        "api": "SetFileTime() API",
        "manipulation": "$SI-M",
    },
    3: {
        "target": "NewFileTime_SI_MAC_Manipulation.dll",
        "method": "NewFileTime",
        "api": "SetFileTime() API",
        "manipulation": "$SI-MAC",
    },
    4: {
        "target": "PowerShell_SI_C_Manipulation.dll",
        "method": "PowerShell",
        "api": "Get-Item Cmdlet",
        "manipulation": "$SI-C",
    },
    5: {
        "target": "PowerShell_SI_M_Manipulation.dll",
        "method": "PowerShell",
        "api": "Get-Item Cmdlet",
        "manipulation": "$SI-M",
    },
    6: {
        "target": "PowerShell_SI_MAC_Manipulation.dll",
        "method": "PowerShell",
        "api": "Get-Item Cmdlet",
        "manipulation": "$SI-MAC",
    },
    7: {
        "target": "nTimestomp_SI_C_Manipulation.dll",
        "method": "nTimestomp",
        "api": "NtSetInformationFile() API",
        "manipulation": "$SI-C",
    },
    8: {
        "target": "nTimestomp_SI_M_Manipulation.dll",
        "method": "nTimestomp",
        "api": "NtSetInformationFile() API",
        "manipulation": "$SI-M",
    },
    9: {
        "target": "nTimestomp_SI_MACE_Manipulation.dll",
        "method": "nTimestomp",
        "api": "NtSetInformationFile() API",
        "manipulation": "$SI-MACE",
    },
    10: {
        "target": "FileSystemTunneling_SI_C_Change.dll",
        "method": "File System Tunneling",
        "api": "File System Tunneling",
        "manipulation": "$SI-C",
    },
    11: {
        "target": "SetMACE_SI_MACE_Copy_Manipulation.dll",
        "method": "SetMACE",
        "api": "NtSetInformationFile() API",
        "manipulation": "$SI-MACE",
    },
    12: {
        "target": "SetMACE_SI_FN_MACE_Manipulation.dll",
        "method": "SetMACE",
        "api": "NtSetInformationFile() API",
        "manipulation": "$SI-MACE + $FN-MACE",
    },
}


# ============================================================
# HELPER
# ============================================================

def find_target_records(mft, target_name):

    result = mft[
        mft["FileName"]
        .astype(str)
        .str.strip()
        .str.lower()
        == target_name.lower()
    ].copy()

    return result


# ============================================================
# LOAD MFT
# ============================================================

print("=" * 100)
print("SCRIPT 22 - BUILD GROUND TRUTH")
print("=" * 100)

print("\nLOAD MFT")
print("=" * 100)

print(f"MFT: {MFT_CSV}")

mft = pd.read_csv(
    MFT_CSV,
    low_memory=False
)

print(f"Rows: {len(mft):,}")
print(f"Columns: {len(mft.columns)}")


# ============================================================
# NORMALIZE KEY
# ============================================================

mft["EntryNumber"] = pd.to_numeric(
    mft["EntryNumber"],
    errors="coerce"
)

mft["SequenceNumber"] = pd.to_numeric(
    mft["SequenceNumber"],
    errors="coerce"
)

# MFT record identity
mft["file_id"] = (
    mft["EntryNumber"].astype("Int64").astype(str)
    + "-"
    + mft["SequenceNumber"].astype("Int64").astype(str)
)


# ============================================================
# INITIAL LABEL
# ============================================================

mft["label_target"] = 0
mft["scenario"] = pd.NA
mft["target_filename"] = pd.NA
mft["method"] = pd.NA
mft["api"] = pd.NA
mft["manipulation_type"] = pd.NA


# ============================================================
# PROCESS SCENARIO 01
# ============================================================

scenario_no = 1
info = SCENARIOS[scenario_no]

target_name = info["target"]

print("\n")
print("=" * 100)
print(f"SCENARIO {scenario_no}")
print("=" * 100)

print(f"Target       : {target_name}")
print(f"Method       : {info['method']}")
print(f"API          : {info['api']}")
print(f"Manipulation : {info['manipulation']}")


target = find_target_records(
    mft,
    target_name
)

print(f"\nTarget MFT records found: {len(target)}")


if len(target) == 0:

    print("\nWARNING: target tidak ditemukan.")

else:

    target_ids = set(target["file_id"])

    mask = mft["file_id"].isin(target_ids)

    mft.loc[mask, "label_target"] = 1
    mft.loc[mask, "scenario"] = scenario_no
    mft.loc[mask, "target_filename"] = target_name
    mft.loc[mask, "method"] = info["method"]
    mft.loc[mask, "api"] = info["api"]
    mft.loc[mask, "manipulation_type"] = info["manipulation"]


    print("\nTARGET RECORD")
    print(
        target[
            [
                "EntryNumber",
                "SequenceNumber",
                "FileName",
                "Extension",
                "FileSize",
                "SI<FN",
                "Created0x10",
                "Created0x30",
                "LastModified0x10",
                "LastModified0x30",
                "LastRecordChange0x10",
                "LastRecordChange0x30",
                "LastAccess0x10",
                "LastAccess0x30",
            ]
        ].to_string(index=False)
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n")
print("=" * 100)
print("LABEL DISTRIBUTION")
print("=" * 100)

print(
    mft["label_target"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nTarget records:")
print(
    mft[mft["label_target"] == 1][
        [
            "EntryNumber",
            "SequenceNumber",
            "FileName",
            "label_target",
            "scenario",
            "method",
            "manipulation_type",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE FULL DATASET
# ============================================================

full_output = (
    OUTPUT_DIR
    / "Scenario_01_ground_truth_full.csv"
)

mft.to_csv(
    full_output,
    index=False
)

print("\nFull ground truth:")
print(full_output)


# ============================================================
# SAVE TARGET ONLY
# ============================================================

target_output = (
    OUTPUT_DIR
    / "Scenario_01_ground_truth_target.csv"
)

mft[
    mft["label_target"] == 1
].to_csv(
    target_output,
    index=False
)

print("\nTarget ground truth:")
print(target_output)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "scenario": scenario_no,
            "target_filename": target_name,
            "method": info["method"],
            "api": info["api"],
            "manipulation_type": info["manipulation"],
            "mft_total_records": len(mft),
            "target_records": int(
                (mft["label_target"] == 1).sum()
            ),
            "non_target_records": int(
                (mft["label_target"] == 0).sum()
            ),
        }
    ]
)

summary_output = (
    OUTPUT_DIR
    / "ground_truth_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSummary output:")
print(summary_output)


print("\n")
print("=" * 100)
print("SCRIPT 22 SELESAI")
print("=" * 100)