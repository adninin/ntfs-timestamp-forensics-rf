import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

FILE_INPUT = r"C:\Users\atnin\Downloads\TimestampResearch\output\features_scenario_01.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("Membaca feature dataset...")

df = pd.read_csv(FILE_INPUT)

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)

# Urutkan berdasarkan file dan waktu
df = df.sort_values(
    ["Name", "UpdateTimestamp"]
).reset_index(drop=True)


# ==========================================
# EVENT SEBELUMNYA
# ==========================================

df["previous_reason"] = (
    df.groupby("Name")["UpdateReasons"]
    .shift(1)
)


df["previous_timestamp"] = (
    df.groupby("Name")["UpdateTimestamp"]
    .shift(1)
)


# ==========================================
# TIME DELTA
# ==========================================

df["sequence_delta"] = (
    df["UpdateTimestamp"] -
    df["previous_timestamp"]
).dt.total_seconds()


# ==========================================
# BASIC INFO CHANGE
# ==========================================

basic = df[
    df["has_basic_info_change"] == 1
].copy()


# ==========================================
# TAMPILKAN SEQUENCE
# BASIC INFO CHANGE
# ==========================================

print("\n=== BASIC INFO CHANGE DENGAN EVENT SEBELUMNYA ===")

columns = [
    "Name",
    "Extension",
    "previous_timestamp",
    "UpdateTimestamp",
    "sequence_delta",
    "previous_reason",
    "UpdateReasons"
]

print(
    basic[columns]
    .head(50)
    .to_string(index=False)
)


# ==========================================
# POLA PREVIOUS EVENT
# ==========================================

print("\n=== EVENT SEBELUM BASIC INFO CHANGE ===")

print(
    basic["previous_reason"]
    .value_counts()
    .head(30)
)


# ==========================================
# SIMPAN
# ==========================================

output = r"C:\Users\atnin\Downloads\TimestampResearch\output\sequence_scenario_01.csv"

basic.to_csv(
    output,
    index=False
)

print(f"\nHasil disimpan ke:")
print(output)