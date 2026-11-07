import pandas as pd

FILE_INPUT = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"

print("Membaca dataset...")
df = pd.read_csv(FILE_INPUT)

# =========================================================
# 1. IDENTITAS
# =========================================================

df["file_id"] = (
    df["EntryNumber"].astype(str)
    + "-"
    + df["SequenceNumber"].astype(str)
)

# =========================================================
# 2. TIMESTAMP
# =========================================================

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)

# =========================================================
# 3. INFORMASI DASAR USN
# =========================================================

print("\n=== UNIQUE UPDATE SEQUENCE NUMBER ===")

print(
    "Jumlah record:",
    len(df)
)

print(
    "Jumlah UpdateSequenceNumber unik:",
    df["UpdateSequenceNumber"].nunique()
)

print(
    "Nilai kosong:",
    df["UpdateSequenceNumber"].isna().sum()
)

# =========================================================
# 4. CEK DUPLIKASI UPDATE SEQUENCE NUMBER
# =========================================================

duplicate_usn = (
    df["UpdateSequenceNumber"]
    .duplicated(keep=False)
)

print("\n=== DUPLIKASI UPDATE SEQUENCE NUMBER ===")

print(
    "Jumlah record yang memiliki USN duplicate:",
    duplicate_usn.sum()
)

print(
    "Jumlah nilai USN yang duplicate:",
    df.loc[
        duplicate_usn,
        "UpdateSequenceNumber"
    ].nunique()
)

# =========================================================
# 5. CONTOH RECORD
# =========================================================

print("\n=== CONTOH DATA ===")

print(
    df[
        [
            "file_id",
            "Name",
            "UpdateSequenceNumber",
            "UpdateTimestamp",
            "UpdateReasons"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

# =========================================================
# 6. SATU FILE_ID
# =========================================================

print("\n=== CONTOH SATU FILE ID ===")

# Ambil file dengan event terbanyak
top_file = (
    df.groupby("file_id")
    .size()
    .sort_values(ascending=False)
    .index[0]
)

print("File ID:", top_file)

subset = df[
    df["file_id"] == top_file
].copy()

subset = subset.sort_values(
    [
        "UpdateTimestamp",
        "UpdateSequenceNumber"
    ]
)

print(
    subset[
        [
            "Name",
            "EntryNumber",
            "SequenceNumber",
            "UpdateSequenceNumber",
            "UpdateTimestamp",
            "UpdateReasons"
        ]
    ]
    .head(100)
    .to_string(index=False)
)

# =========================================================
# 7. CEK MONOTONICITY DALAM FILE ID
# =========================================================

print("\n=== MONOTONICITY UPDATE SEQUENCE NUMBER ===")

ordered = df.sort_values(
    [
        "file_id",
        "UpdateTimestamp",
        "UpdateSequenceNumber"
    ]
).copy()

ordered["previous_usn"] = (
    ordered.groupby("file_id")[
        "UpdateSequenceNumber"
    ].shift(1)
)

ordered["usn_difference"] = (
    ordered["UpdateSequenceNumber"]
    - ordered["previous_usn"]
)

valid_diff = ordered[
    ordered["usn_difference"].notna()
]

print(
    "USN naik:",
    (
        valid_diff["usn_difference"] > 0
    ).sum()
)

print(
    "USN sama:",
    (
        valid_diff["usn_difference"] == 0
    ).sum()
)

print(
    "USN turun:",
    (
        valid_diff["usn_difference"] < 0
    ).sum()
)