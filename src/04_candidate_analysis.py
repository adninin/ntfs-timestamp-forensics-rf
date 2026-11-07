import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

FILE_INPUT = r"C:\Users\atnin\Downloads\TimestampResearch\output\features_scenario_01.csv"


# ==========================================
# LOAD
# ==========================================

print("Membaca feature dataset...")

df = pd.read_csv(FILE_INPUT)

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)


# ==========================================
# CANDIDATE 1
# BASIC INFO CHANGE
# ==========================================

candidate_basic = df[
    df["has_basic_info_change"] == 1
].copy()

print("\n=== CANDIDATE BASIC INFO CHANGE ===")
print(f"Jumlah: {len(candidate_basic):,}")


# ==========================================
# CANDIDATE 2
# BASIC INFO CHANGE + CLOSE
# ==========================================

candidate_basic_close = df[
    (df["has_basic_info_change"] == 1) &
    (df["has_close"] == 1)
].copy()

print("\n=== BASIC INFO CHANGE + CLOSE ===")
print(f"Jumlah: {len(candidate_basic_close):,}")


# ==========================================
# CANDIDATE 3
# BASIC INFO CHANGE DENGAN
# TIME DELTA KECIL
# ==========================================

candidate_fast = df[
    (df["has_basic_info_change"] == 1) &
    (df["time_delta_seconds"].notna()) &
    (df["time_delta_seconds"] <= 1)
].copy()

print("\n=== BASIC INFO CHANGE + DELTA <= 1 SECOND ===")
print(f"Jumlah: {len(candidate_fast):,}")


# ==========================================
# CANDIDATE 4
# BASIC INFO CHANGE + CLOSE
# + DELTA <= 1 SECOND
# ==========================================

candidate_strong = df[
    (df["has_basic_info_change"] == 1) &
    (df["has_close"] == 1) &
    (df["time_delta_seconds"].notna()) &
    (df["time_delta_seconds"] <= 1)
].copy()

print("\n=== STRONG CANDIDATE ===")
print(f"Jumlah: {len(candidate_strong):,}")


# ==========================================
# TAMPILKAN CONTOH
# ==========================================

columns = [
    "Name",
    "Extension",
    "UpdateTimestamp",
    "UpdateReasons",
    "time_delta_seconds",
    "event_count_per_file",
    "basic_info_count_per_file",
    "close_count_per_file"
]

print("\n=== CONTOH STRONG CANDIDATE ===")

print(
    candidate_strong[
        columns
    ]
    .head(20)
    .to_string(index=False)
)