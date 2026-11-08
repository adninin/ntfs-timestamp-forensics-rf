import pandas as pd
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "sequence_features_scenario_01.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "ground_truth_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. MEMBACA DATASET
# ============================================================

print("Membaca sequence dataset...")

df = pd.read_csv(INPUT_FILE)

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

df["previous_timestamp"] = pd.to_datetime(
    df["previous_timestamp"],
    errors="coerce"
)


# ============================================================
# 3. INFORMASI DASAR DATASET
# ============================================================

print("\n=== INFORMASI DASAR ===")

print(
    f"Jumlah file_id unik : "
    f"{df['file_id'].nunique():,}"
)

print(
    f"Jumlah Name unik    : "
    f"{df['Name'].nunique():,}"
)

print(
    f"Jumlah BasicInfoChange : "
    f"{df['has_basic_info_change'].sum():,}"
)


# ============================================================
# 4. KANDIDAT BASIC INFO CHANGE
# ============================================================
#
# Ini BUKAN ground truth.
#
# Hanya kandidat yang akan dianalisis lebih lanjut.
# ============================================================

basic_info = df[
    df["has_basic_info_change"] == 1
].copy()

print("\n=== BASIC INFO CHANGE ===")

print(
    f"Jumlah event BasicInfoChange : "
    f"{len(basic_info):,}"
)

print(
    f"Jumlah file_id yang memiliki "
    f"BasicInfoChange : "
    f"{basic_info['file_id'].nunique():,}"
)


# ============================================================
# 5. TAMBAHKAN NEXT EVENT
# ============================================================
#
# Kita membutuhkan:
#
# previous_reason
# current UpdateReasons
# next_reason
#
# supaya bisa melihat konteks:
#
# previous → BasicInfoChange → next
#
# ============================================================

df["next_reason"] = (
    df.groupby("file_id")["UpdateReasons"]
    .shift(-1)
)

df["next_usn"] = (
    df.groupby("file_id")["UpdateSequenceNumber"]
    .shift(-1)
)

df["next_timestamp"] = (
    df.groupby("file_id")["UpdateTimestamp"]
    .shift(-1)
)


# Ambil kembali kandidat setelah kolom next tersedia

basic_info = df[
    df["has_basic_info_change"] == 1
].copy()


# ============================================================
# 6. SELISIH USN KE EVENT BERIKUTNYA
# ============================================================

basic_info["next_usn_delta"] = (
    basic_info["next_usn"]
    - basic_info["UpdateSequenceNumber"]
)


# ============================================================
# 7. SELISIH WAKTU KE EVENT BERIKUTNYA
# ============================================================

basic_info["next_time_delta_seconds"] = (
    basic_info["next_timestamp"]
    - basic_info["UpdateTimestamp"]
).dt.total_seconds()


# ============================================================
# 8. LAPORAN EVENT BASIC INFO CHANGE
# ============================================================

candidate_columns = [
    "file_id",
    "Name",
    "Extension",
    "EntryNumber",
    "SequenceNumber",
    "event_order",
    "UpdateSequenceNumber",
    "UpdateTimestamp",
    "usn_delta",
    "time_delta_seconds",
    "previous_reason",
    "UpdateReasons",
    "next_reason",
    "next_usn_delta",
    "next_time_delta_seconds",
    "FileAttributes",
    "UpdateReasons"
]

# Hilangkan duplicate nama kolom jika ada
candidate_columns = list(
    dict.fromkeys(candidate_columns)
)

basic_info_candidates = basic_info[
    candidate_columns
].copy()


# ============================================================
# 9. SIMPAN SEMUA KANDIDAT BASIC INFO CHANGE
# ============================================================

output_basic_info = (
    OUTPUT_DIR
    / "scenario_01_basic_info_candidates.csv"
)

basic_info_candidates.to_csv(
    output_basic_info,
    index=False
)

print("\nHasil kandidat BasicInfoChange:")
print(output_basic_info)


# ============================================================
# 10. KANDIDAT DENGAN BASICINFOCHANGE + CLOSE
# ============================================================
#
# Ini hanya pola yang menarik untuk diperiksa.
# BUKAN label manipulasi.
# ============================================================

basic_close = basic_info[
    basic_info["has_close"] == 1
].copy()

print("\n=== BASIC INFO CHANGE + CLOSE ===")

print(
    f"Jumlah event : "
    f"{len(basic_close):,}"
)

print(
    f"Jumlah file_id : "
    f"{basic_close['file_id'].nunique():,}"
)


# ============================================================
# 11. SIMPAN BASIC INFO + CLOSE
# ============================================================

output_basic_close = (
    OUTPUT_DIR
    / "scenario_01_basic_info_close_candidates.csv"
)

basic_close[
    candidate_columns
].to_csv(
    output_basic_close,
    index=False
)

print(
    "Hasil BasicInfoChange + Close:"
)

print(output_basic_close)


# ============================================================
# 12. KANDIDAT BASIC INFO CHANGE YANG MERUPAKAN EVENT
#    PERTAMA PADA FILE_ID
# ============================================================

basic_first = basic_info[
    basic_info["event_order"] == 1
].copy()

print(
    "\n=== BASIC INFO CHANGE SEBAGAI EVENT PERTAMA ==="
)

print(
    f"Jumlah : {len(basic_first):,}"
)


# ============================================================
# 13. KANDIDAT BASIC INFO CHANGE DENGAN FILECREATE
# ============================================================

basic_file_create = basic_info[
    basic_info["has_file_create"] == 1
].copy()

print(
    "\n=== BASIC INFO CHANGE + FILE CREATE ==="
)

print(
    f"Jumlah : {len(basic_file_create):,}"
)


# ============================================================
# 14. KANDIDAT BASIC INFO CHANGE TANPA FILECREATE
# ============================================================
#
# Ini menarik karena BasicInfoChange tidak selalu muncul
# bersamaan dengan FileCreate.
#
# Tetapi sekali lagi:
# BUKAN berarti manipulasi.
# ============================================================

basic_without_create = basic_info[
    basic_info["has_file_create"] == 0
].copy()

print(
    "\n=== BASIC INFO CHANGE TANPA FILE CREATE ==="
)

print(
    f"Jumlah : {len(basic_without_create):,}"
)


# ============================================================
# 15. RINGKASAN PER FILE_ID
# ============================================================
#
# Kita sekarang membuat satu baris untuk setiap file_id.
#
# Tujuannya:
# melihat file mana yang memiliki banyak aktivitas
# BasicInfoChange.
# ============================================================

file_summary = (
    df.groupby(
        [
            "file_id",
            "Name",
            "Extension",
            "EntryNumber",
            "SequenceNumber"
        ],
        dropna=False
    )
    .agg(
        event_count=(
            "UpdateSequenceNumber",
            "count"
        ),
        basic_info_count=(
            "has_basic_info_change",
            "sum"
        ),
        close_count=(
            "has_close",
            "sum"
        ),
        file_create_count=(
            "has_file_create",
            "sum"
        ),
        file_delete_count=(
            "has_file_delete",
            "sum"
        ),
        rename_count=(
            "has_rename",
            "sum"
        ),
        security_change_count=(
            "has_security_change",
            "sum"
        ),
        first_usn=(
            "UpdateSequenceNumber",
            "min"
        ),
        last_usn=(
            "UpdateSequenceNumber",
            "max"
        ),
        first_timestamp=(
            "UpdateTimestamp",
            "min"
        ),
        last_timestamp=(
            "UpdateTimestamp",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# 16. HITUNG TIME SPAN
# ============================================================

file_summary["time_span_seconds"] = (
    file_summary["last_timestamp"]
    - file_summary["first_timestamp"]
).dt.total_seconds()


# ============================================================
# 17. FLAG FILE YANG MEMILIKI BASIC INFO CHANGE
# ============================================================

file_summary["has_any_basic_info_change"] = (
    file_summary["basic_info_count"] > 0
).astype(int)


# ============================================================
# 18. URUTKAN BERDASARKAN BASIC INFO CHANGE
# ============================================================

file_summary = file_summary.sort_values(
    by=[
        "basic_info_count",
        "event_count"
    ],
    ascending=False
)


# ============================================================
# 19. SIMPAN RINGKASAN FILE
# ============================================================

output_file_summary = (
    OUTPUT_DIR
    / "scenario_01_file_summary.csv"
)

file_summary.to_csv(
    output_file_summary,
    index=False
)

print(
    "\nHasil ringkasan per file:"
)

print(output_file_summary)


# ============================================================
# 20. TOP FILE DENGAN BASIC INFO CHANGE
# ============================================================

print(
    "\n=== TOP FILE DENGAN BASIC INFO CHANGE ==="
)

top_columns = [
    "file_id",
    "Name",
    "Extension",
    "event_count",
    "basic_info_count",
    "close_count",
    "file_create_count",
    "file_delete_count",
    "rename_count",
    "security_change_count",
    "time_span_seconds"
]

print(
    file_summary[
        top_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 21. ANALISIS POLA SEBELUM → BASICINFO → SESUDAH
# ============================================================

pattern_summary = (
    basic_info
    .groupby(
        [
            "previous_reason",
            "UpdateReasons",
            "next_reason"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="frequency"
    )
    .sort_values(
        "frequency",
        ascending=False
    )
)


# ============================================================
# 22. SIMPAN POLA SEQUENCE
# ============================================================

output_pattern = (
    OUTPUT_DIR
    / "scenario_01_basic_info_patterns.csv"
)

pattern_summary.to_csv(
    output_pattern,
    index=False
)

print(
    "\nHasil pola sequence:"
)

print(output_pattern)


# ============================================================
# 23. TAMPILKAN POLA TERBANYAK
# ============================================================

print(
    "\n=== TOP POLA SEQUENCE ==="
)

print(
    pattern_summary
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 24. DISTRIBUSI EVENT ORDER BASIC INFO CHANGE
# ============================================================

event_order_summary = (
    basic_info[
        "event_order"
    ]
    .describe()
)

print(
    "\n=== EVENT ORDER BASIC INFO CHANGE ==="
)

print(event_order_summary)


# ============================================================
# 25. DISTRIBUSI TIME DELTA
# ============================================================

print(
    "\n=== TIME DELTA BASIC INFO CHANGE ==="
)

print(
    basic_info[
        "time_delta_seconds"
    ]
    .describe()
)


# ============================================================
# 26. CONTOH DETAIL KANDIDAT
# ============================================================

print(
    "\n=== CONTOH DETAIL KANDIDAT ==="
)

detail_columns = [
    "file_id",
    "Name",
    "event_order",
    "UpdateSequenceNumber",
    "UpdateTimestamp",
    "previous_reason",
    "UpdateReasons",
    "next_reason",
    "time_delta_seconds",
    "next_time_delta_seconds"
]

print(
    basic_info[
        detail_columns
    ]
    .head(50)
    .to_string(index=False)
)


# ============================================================
# 27. SELESAI
# ============================================================

print("\n=== SELESAI ===")

print(
    "Semua hasil disimpan di:"
)

print(OUTPUT_DIR)