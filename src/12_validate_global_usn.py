import pandas as pd
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(r"C:\Users\atnin\Downloads\TimestampResearch")
INPUT_FILE = BASE_DIR / "output" / "features_scenario_01.csv"
OUTPUT_FILE = BASE_DIR / "output" / "global_usn_validation_scenario_01.csv"


# ============================================================
# 1. MEMBACA DATA
# ============================================================

print("Membaca dataset...")

df = pd.read_csv(INPUT_FILE)

df["file_id"] = (
    df["EntryNumber"].astype(str)
    + "-"
    + df["SequenceNumber"].astype(str)
)

print("\n=== DATASET ===")
print(f"Jumlah baris : {len(df):,}")
print(f"Jumlah kolom : {len(df.columns)}")


# ============================================================
# 2. KONVERSI TIMESTAMP
# ============================================================

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)

print("\n=== TIMESTAMP ===")
print(f"Timestamp invalid : {df['UpdateTimestamp'].isna().sum():,}")


# ============================================================
# 3. VALIDASI USN GLOBAL
# ============================================================

print("\n=== VALIDASI UPDATE SEQUENCE NUMBER ===")

usn = df["UpdateSequenceNumber"]

print(f"Jumlah USN              : {len(usn):,}")
print(f"USN unique              : {usn.nunique():,}")
print(f"USN duplicate           : {usn.duplicated().sum():,}")
print(f"USN missing             : {usn.isna().sum():,}")

if usn.nunique() == len(usn):
    print("STATUS: Setiap USN unik secara global.")
else:
    print("STATUS: Ada USN yang duplikat.")


# ============================================================
# 4. CEK URUTAN USN PADA URUTAN DATASET ASLI
# ============================================================

print("\n=== URUTAN USN PADA DATASET ASLI ===")

usn_diff_raw = usn.diff()

raw_increase = (usn_diff_raw > 0).sum()
raw_same = (usn_diff_raw == 0).sum()
raw_decrease = (usn_diff_raw < 0).sum()

print(f"USN naik   : {raw_increase:,}")
print(f"USN sama   : {raw_same:,}")
print(f"USN turun  : {raw_decrease:,}")

if raw_decrease == 0:
    print("STATUS: Tidak ditemukan penurunan USN pada urutan dataset.")
else:
    print("STATUS: Ditemukan penurunan USN pada urutan dataset.")


# ============================================================
# 5. URUTKAN SELURUH DATA BERDASARKAN USN
# ============================================================

print("\nMengurutkan seluruh dataset berdasarkan USN...")

df_usn = df.sort_values(
    by="UpdateSequenceNumber"
).reset_index(drop=True)

usn_diff_sorted = df_usn["UpdateSequenceNumber"].diff()

sorted_decrease = (usn_diff_sorted < 0).sum()
sorted_same = (usn_diff_sorted == 0).sum()

print("\n=== URUTAN SETELAH SORTING USN ===")
print(f"USN sama   : {sorted_same:,}")
print(f"USN turun  : {sorted_decrease:,}")

if sorted_decrease == 0:
    print("STATUS: Urutan USN valid setelah sorting.")
else:
    print("STATUS: Ada masalah pada urutan USN.")


# ============================================================
# 6. HUBUNGAN USN DENGAN TIMESTAMP
# ============================================================

print("\n=== HUBUNGAN USN DAN TIMESTAMP ===")

df_usn["timestamp_diff_seconds"] = (
    df_usn["UpdateTimestamp"]
    .diff()
    .dt.total_seconds()
)

timestamp_increase = (
    df_usn["timestamp_diff_seconds"] > 0
).sum()

timestamp_same = (
    df_usn["timestamp_diff_seconds"] == 0
).sum()

timestamp_decrease = (
    df_usn["timestamp_diff_seconds"] < 0
).sum()

print(f"Timestamp naik   : {timestamp_increase:,}")
print(f"Timestamp sama   : {timestamp_same:,}")
print(f"Timestamp turun  : {timestamp_decrease:,}")


# ============================================================
# 7. CARI CONTOH TIMESTAMP TURUN
# ============================================================

timestamp_inversion = df_usn[
    df_usn["timestamp_diff_seconds"] < 0
].copy()

print("\n=== TIMESTAMP INVERSION ===")
print(
    f"Jumlah event dengan timestamp lebih kecil "
    f"dari event USN sebelumnya: {len(timestamp_inversion):,}"
)

if len(timestamp_inversion) > 0:

    cols = [
        "file_id",
        "UpdateSequenceNumber",
        "UpdateTimestamp",
        "timestamp_diff_seconds",
        "UpdateReasons"
    ]

    print("\nContoh:")
    print(
        timestamp_inversion[cols]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 8. EVENT DENGAN TIMESTAMP SAMA TETAPI USN BERBEDA
# ============================================================

print("\n=== TIMESTAMP SAMA, USN BERBEDA ===")

same_timestamp = df_usn[
    df_usn["timestamp_diff_seconds"] == 0
].copy()

print(
    f"Event dengan timestamp sama "
    f"dengan event sebelumnya: {len(same_timestamp):,}"
)

if len(same_timestamp) > 0:

    cols = [
        "file_id",
        "UpdateSequenceNumber",
        "UpdateTimestamp",
        "UpdateReasons"
    ]

    print("\nContoh:")
    print(
        same_timestamp[cols]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 9. VALIDASI ORDER DALAM FILE_ID
# ============================================================

print("\n=== VALIDASI ORDER PER FILE_ID ===")

df_usn["usn_diff_within_file"] = (
    df_usn
    .groupby("file_id")["UpdateSequenceNumber"]
    .diff()
)

within_file_decrease = (
    df_usn["usn_diff_within_file"] < 0
).sum()

within_file_same = (
    df_usn["usn_diff_within_file"] == 0
).sum()

print(f"USN turun dalam file_id : {within_file_decrease:,}")
print(f"USN sama dalam file_id  : {within_file_same:,}")

if within_file_decrease == 0:
    print(
        "STATUS: Tidak ditemukan penurunan USN "
        "dalam urutan event setiap file_id."
    )
else:
    print(
        "STATUS: Ditemukan penurunan USN "
        "dalam beberapa file_id."
    )


# ============================================================
# 10. SIMPAN HASIL VALIDASI
# ============================================================

validation_columns = [
    "file_id",
    "UpdateSequenceNumber",
    "UpdateTimestamp",
    "timestamp_diff_seconds",
    "usn_diff_within_file",
    "UpdateReasons"
]

df_usn[validation_columns].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n=== SELESAI ===")
print("Hasil disimpan ke:")
print(OUTPUT_FILE)