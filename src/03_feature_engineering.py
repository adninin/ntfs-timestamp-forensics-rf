import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

FILE_SCENARIO = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"

OUTPUT_FILE = r"C:\Users\atnin\Downloads\TimestampResearch\output\features_scenario_01.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("Membaca Scenario 01...")

df = pd.read_csv(FILE_SCENARIO)

print(f"Jumlah record awal: {len(df):,}")


# ==========================================
# PARSE TIMESTAMP
# ==========================================

df["UpdateTimestamp"] = pd.to_datetime(
    df["UpdateTimestamp"],
    errors="coerce"
)


# ==========================================
# FITUR UPDATE REASONS
# ==========================================

df["has_basic_info_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("BasicInfoChange", case=False)
    .astype(int)
)

df["has_close"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Close", case=False)
    .astype(int)
)

df["has_file_create"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileCreate", case=False)
    .astype(int)
)

df["has_file_delete"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileDelete", case=False)
    .astype(int)
)

df["has_data_extend"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("DataExtend", case=False)
    .astype(int)
)

df["has_data_overwrite"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("DataOverwrite", case=False)
    .astype(int)
)

df["has_rename"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Rename", case=False)
    .astype(int)
)

df["has_security_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("SecurityChange", case=False)
    .astype(int)
)


# ==========================================
# JUMLAH REASON DALAM SATU EVENT
# ==========================================

df["reason_count"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.split("|")
    .str.len()
)


# ==========================================
# FITUR WAKTU
# ==========================================

df["hour"] = df["UpdateTimestamp"].dt.hour

df["minute"] = df["UpdateTimestamp"].dt.minute

df["second"] = df["UpdateTimestamp"].dt.second

df["weekday"] = df["UpdateTimestamp"].dt.weekday


# ==========================================
# URUTKAN BERDASARKAN WAKTU
# ==========================================

df = df.sort_values(
    ["Name", "UpdateTimestamp"]
).reset_index(drop=True)


# ==========================================
# JEDA WAKTU EVENT FILE YANG SAMA
# ==========================================

df["time_delta_seconds"] = (
    df.groupby("Name")["UpdateTimestamp"]
    .diff()
    .dt.total_seconds()
)


# ==========================================
# JUMLAH EVENT PER FILE
# ==========================================

df["event_count_per_file"] = (
    df.groupby("Name")["Name"]
    .transform("count")
)


# ==========================================
# JUMLAH BASIC INFO CHANGE PER FILE
# ==========================================

df["basic_info_count_per_file"] = (
    df.groupby("Name")["has_basic_info_change"]
    .transform("sum")
)


# ==========================================
# JUMLAH CLOSE PER FILE
# ==========================================

df["close_count_per_file"] = (
    df.groupby("Name")["has_close"]
    .transform("sum")
)


# ==========================================
# SIMPAN
# ==========================================

print("\nMenyimpan feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Output: {OUTPUT_FILE}")

print("\n=== FITUR YANG DIBUAT ===")

features = [
    "has_basic_info_change",
    "has_close",
    "has_file_create",
    "has_file_delete",
    "has_data_extend",
    "has_data_overwrite",
    "has_rename",
    "has_security_change",
    "reason_count",
    "hour",
    "minute",
    "second",
    "weekday",
    "time_delta_seconds",
    "event_count_per_file",
    "basic_info_count_per_file",
    "close_count_per_file"
]

for feature in features:
    print("-", feature)