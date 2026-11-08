from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

PATH = (
    ROOT
    / "output"
    / "hard_negative_usn_integrated"
    / "hard_negative_usn_integrated_all.csv"
)

print("=" * 80)
print("VALIDATE HARD NEGATIVE DATASET")
print("=" * 80)

df = pd.read_csv(PATH)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabels:")
print(df["label"].value_counts())

print("\nNegative types:")
print(df["negative_type"].value_counts())

# scenario mungkin sudah ada
if "scenario" in df.columns:

    print("\nPer scenario:")
    print(
        df.groupby(
            ["scenario", "label"]
        ).size()
    )

# ============================================================
# USN CHECK
# ============================================================

negative = df[
    df["label"] == 0
].copy()

print("\nHard negative USN check:")

print(
    "Negative has_usn=1:",
    (
        negative["has_usn"] == 1
    ).sum()
)

print(
    "Negative event_count > 0:",
    (
        negative["usn_event_count"] > 0
    ).sum()
)

# ============================================================
# DUPLICATE
# ============================================================

print("\nDuplicate file_id:")

print(
    df["file_id"].duplicated().sum()
)

print("\nDuplicate scenario + file_id:")

if "scenario" in df.columns:

    print(
        df.duplicated(
            subset=[
                "scenario",
                "file_id"
            ]
        ).sum()
    )

# ============================================================
# TARGET
# ============================================================

print("\nTargets:")

print(
    df[
        df["label"] == 1
    ][
        [
            "scenario",
            "file_id",
            "usn_event_count",
            "usn_basic_info_count",
            "usn_close_count",
            "usn_data_extend_count",
            "usn_file_create_count",
        ]
    ].to_string(index=False)
)

print("\nDONE.")