import pandas as pd
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "features_scenario_01.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "sequence_features_scenario_01.csv"
)


# ============================================================
# 1. MEMBACA DATASET
# ============================================================

print("Membaca dataset...")

df = pd.read_csv(INPUT_FILE)

print("\n=== DATASET AWAL ===")
print(f"Jumlah baris : {len(df):,}")
print(f"Jumlah kolom : {len(df.columns)}")


# ============================================================
# 2. MEMBUAT FILE ID
# ============================================================
#
# file_id = EntryNumber + SequenceNumber
#
# Contoh:
# EntryNumber = 518230
# SequenceNumber = 2
#
# file_id = 518230-2
#
# Ini digunakan sebagai identitas record MFT.
# ============================================================

df["file_id"] = (
    df["EntryNumber"].astype(str)
    + "-"
    + df["SequenceNumber"].astype(str)
)


# ============================================================
# 3. KONVERSI TIMESTAMP
# ============================================================

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)

invalid_timestamp = df["UpdateTimestamp"].isna().sum()

print("\n=== TIMESTAMP ===")
print(f"Timestamp invalid : {invalid_timestamp:,}")


# ============================================================
# 4. VALIDASI DASAR USN
# ============================================================

print("\n=== VALIDASI USN ===")

usn_unique = df["UpdateSequenceNumber"].nunique()
usn_total = len(df)
usn_duplicate = df["UpdateSequenceNumber"].duplicated().sum()
usn_missing = df["UpdateSequenceNumber"].isna().sum()

print(f"Jumlah record : {usn_total:,}")
print(f"USN unique   : {usn_unique:,}")
print(f"USN duplicate: {usn_duplicate:,}")
print(f"USN missing  : {usn_missing:,}")


# ============================================================
# 5. GLOBAL EVENT ORDER
# ============================================================
#
# PENTING:
#
# Jangan mengurutkan berdasarkan file_id terlebih dahulu.
#
# Berdasarkan hasil validasi:
# UpdateSequenceNumber unik secara global.
#
# Karena itu USN digunakan sebagai dasar ordering event.
# ============================================================

print("\nMengurutkan dataset berdasarkan UpdateSequenceNumber...")

df = df.sort_values(
    by="UpdateSequenceNumber"
).reset_index(drop=True)


# ============================================================
# 6. VALIDASI SETELAH SORTING
# ============================================================

usn_diff = df["UpdateSequenceNumber"].diff()

usn_decrease = (
    usn_diff < 0
).sum()

usn_same = (
    usn_diff == 0
).sum()

print("\n=== GLOBAL USN ORDER ===")
print(f"USN turun : {usn_decrease:,}")
print(f"USN sama  : {usn_same:,}")

if usn_decrease == 0 and usn_same == 0:
    print("STATUS: Global USN ordering valid.")
else:
    print("PERINGATAN: Periksa kembali urutan USN.")


# ============================================================
# 7. EVENT ORDER DALAM FILE_ID
# ============================================================
#
# Setelah dataset memiliki global order berdasarkan USN,
# kita membuat nomor urut event untuk masing-masing file_id.
#
# Contoh:
#
# file_id     USN       event_order
# 518230-2    1000      1
# 518230-2    1100      2
# 518230-2    1200      3
#
# ============================================================

df["event_order"] = (
    df.groupby("file_id")
    .cumcount()
    + 1
)


# ============================================================
# 8. EVENT SEBELUMNYA
# ============================================================
#
# previous_reason:
# reason event sebelumnya pada file_id yang sama.
#
# previous_usn:
# USN event sebelumnya.
#
# previous_timestamp:
# timestamp event sebelumnya.
# ============================================================

df["previous_reason"] = (
    df.groupby("file_id")["UpdateReasons"]
    .shift(1)
)

df["previous_usn"] = (
    df.groupby("file_id")["UpdateSequenceNumber"]
    .shift(1)
)

df["previous_timestamp"] = (
    df.groupby("file_id")["UpdateTimestamp"]
    .shift(1)
)


# ============================================================
# 9. USN DELTA
# ============================================================
#
# Selisih USN dengan event sebelumnya
# pada file_id yang sama.
#
# ============================================================

df["usn_delta"] = (
    df["UpdateSequenceNumber"]
    - df["previous_usn"]
)


# ============================================================
# 10. TIME DELTA
# ============================================================
#
# Selisih waktu antara event sekarang dan event sebelumnya
# pada file_id yang sama.
#
# Satuan: detik.
#
# ============================================================

df["time_delta_seconds"] = (
    df["UpdateTimestamp"]
    - df["previous_timestamp"]
).dt.total_seconds()


# ============================================================
# 11. EVENT FLAGS
# ============================================================
#
# Mengubah UpdateReasons menjadi fitur binary.
#
# Contoh:
#
# UpdateReasons =
# DataExtend|FileCreate|BasicInfoChange
#
# maka:
#
# has_data_extend       = 1
# has_file_create       = 1
# has_basic_info_change = 1
#
# ============================================================

df["has_basic_info_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "BasicInfoChange",
        regex=False
    )
    .astype(int)
)

df["has_close"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "Close",
        regex=False
    )
    .astype(int)
)

df["has_file_create"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "FileCreate",
        regex=False
    )
    .astype(int)
)

df["has_file_delete"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "FileDelete",
        regex=False
    )
    .astype(int)
)

df["has_data_extend"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "DataExtend",
        regex=False
    )
    .astype(int)
)

df["has_data_overwrite"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "DataOverwrite",
        regex=False
    )
    .astype(int)
)

df["has_rename"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "Rename",
        regex=False
    )
    .astype(int)
)

df["has_security_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains(
        "SecurityChange",
        regex=False
    )
    .astype(int)
)


# ============================================================
# 12. PREVIOUS EVENT FLAGS
# ============================================================
#
# Fitur ini menjawab pertanyaan:
#
# "Apa yang terjadi tepat sebelum event sekarang?"
#
# Contoh:
#
# previous_reason =
# DataExtend|FileCreate
#
# maka:
#
# previous_has_file_create = 1
# previous_has_data_extend = 1
#
# ============================================================

previous_reason = (
    df["previous_reason"]
    .fillna("")
)


df["previous_has_basic_info_change"] = (
    previous_reason
    .str.contains(
        "BasicInfoChange",
        regex=False
    )
    .astype(int)
)

df["previous_has_close"] = (
    previous_reason
    .str.contains(
        "Close",
        regex=False
    )
    .astype(int)
)

df["previous_has_file_create"] = (
    previous_reason
    .str.contains(
        "FileCreate",
        regex=False
    )
    .astype(int)
)

df["previous_has_file_delete"] = (
    previous_reason
    .str.contains(
        "FileDelete",
        regex=False
    )
    .astype(int)
)

df["previous_has_data_extend"] = (
    previous_reason
    .str.contains(
        "DataExtend",
        regex=False
    )
    .astype(int)
)

df["previous_has_data_overwrite"] = (
    previous_reason
    .str.contains(
        "DataOverwrite",
        regex=False
    )
    .astype(int)
)

df["previous_has_rename"] = (
    previous_reason
    .str.contains(
        "Rename",
        regex=False
    )
    .astype(int)
)

df["previous_has_security_change"] = (
    previous_reason
    .str.contains(
        "SecurityChange",
        regex=False
    )
    .astype(int)
)


# ============================================================
# 13. PERUBAHAN REASON
# ============================================================
#
# Apakah UpdateReasons berubah dibanding event sebelumnya?
#
# Contoh:
#
# Event 1:
# DataExtend
#
# Event 2:
# DataExtend|Close
#
# maka:
# reason_changed = 1
#
# ============================================================

df["reason_changed"] = (
    df["UpdateReasons"]
    != df["previous_reason"]
).astype(int)


# ============================================================
# 14. USN DELTA VALIDATION
# ============================================================
#
# Cek apakah USN dalam file_id selalu meningkat.
# ============================================================

within_file_usn_decrease = (
    df.groupby("file_id")["UpdateSequenceNumber"]
    .diff()
    .lt(0)
    .sum()
)

within_file_usn_same = (
    df.groupby("file_id")["UpdateSequenceNumber"]
    .diff()
    .eq(0)
    .sum()
)

print("\n=== VALIDASI SEQUENCE DALAM FILE_ID ===")
print(
    f"USN turun dalam file_id : "
    f"{within_file_usn_decrease:,}"
)

print(
    f"USN sama dalam file_id  : "
    f"{within_file_usn_same:,}"
)


# ============================================================
# 15. RINGKASAN FEATURE
# ============================================================

print("\n=== SEQUENCE DATASET ===")
print(f"Jumlah baris : {len(df):,}")
print(f"Jumlah kolom : {len(df.columns)}")

print("\n=== CONTOH SEQUENCE ===")

example_columns = [
    "file_id",
    "event_order",
    "UpdateSequenceNumber",
    "UpdateTimestamp",
    "usn_delta",
    "time_delta_seconds",
    "previous_reason",
    "UpdateReasons",
]

print(
    df[example_columns]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 16. CONTOH BASIC INFO CHANGE
# ============================================================

print("\n=== BASIC INFO CHANGE DENGAN CONTEXT ===")

basic_info = df[
    df["has_basic_info_change"] == 1
]

basic_info_columns = [
    "file_id",
    "event_order",
    "UpdateSequenceNumber",
    "usn_delta",
    "time_delta_seconds",
    "previous_reason",
    "UpdateReasons",
]

print(
    basic_info[basic_info_columns]
    .head(50)
    .to_string(index=False)
)


# ============================================================
# 17. SIMPAN DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n=== SELESAI ===")
print("Hasil disimpan ke:")
print(OUTPUT_FILE)