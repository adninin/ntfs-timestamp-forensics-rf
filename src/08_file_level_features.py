import pandas as pd

FILE_INPUT = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"

print("Membaca dataset...")
df = pd.read_csv(FILE_INPUT)

# =========================================================
# 1. IDENTITAS RECORD
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
# 3. FEATURE EVENT
# =========================================================

df["has_basic_info_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("BasicInfoChange")
    .astype(int)
)

df["has_close"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Close")
    .astype(int)
)

df["has_file_create"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileCreate")
    .astype(int)
)

df["has_file_delete"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileDelete")
    .astype(int)
)

df["has_rename"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Rename")
    .astype(int)
)

df["has_data_extend"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("DataExtend")
    .astype(int)
)

df["has_data_truncation"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("DataTruncation")
    .astype(int)
)

df["has_security_change"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("SecurityChange")
    .astype(int)
)

df["has_data_overwrite"] = (
    df["UpdateReasons"]
    .fillna("")
    .str.contains("DataOverwrite")
    .astype(int)
)

# =========================================================
# 4. AGGREGASI LEVEL FILE / RECORD
# =========================================================

features = (
    df.groupby("file_id")
    .agg(
        name=("Name", "first"),
        extension=("Extension", "first"),

        entry_number=("EntryNumber", "first"),
        sequence_number=("SequenceNumber", "first"),

        event_count=("file_id", "size"),

        basic_info_count=("has_basic_info_change", "sum"),
        close_count=("has_close", "sum"),
        file_create_count=("has_file_create", "sum"),
        file_delete_count=("has_file_delete", "sum"),
        rename_count=("has_rename", "sum"),
        data_extend_count=("has_data_extend", "sum"),
        data_truncation_count=("has_data_truncation", "sum"),
        data_overwrite_count=("has_data_overwrite", "sum"),
        security_change_count=("has_security_change", "sum"),

        first_event=("UpdateTimestamp", "min"),
        last_event=("UpdateTimestamp", "max")
    )
    .reset_index()
)

# =========================================================
# 5. FITUR TEMPORAL
# =========================================================

features["time_span_seconds"] = (
    features["last_event"] -
    features["first_event"]
).dt.total_seconds()

features["basic_info_ratio"] = (
    features["basic_info_count"] /
    features["event_count"]
)

features["close_ratio"] = (
    features["close_count"] /
    features["event_count"]
)

features["rename_ratio"] = (
    features["rename_count"] /
    features["event_count"]
)

# =========================================================
# 6. TAMPILKAN HASIL
# =========================================================

print("\n=== FILE-LEVEL DATASET ===")
print(features.shape)

print("\n=== STATISTIK EVENT COUNT ===")
print(features["event_count"].describe())

print("\n=== STATISTIK BASIC INFO COUNT ===")
print(features["basic_info_count"].describe())

print("\n=== FILE DENGAN BASIC INFO CHANGE TERBANYAK ===")
print(
    features[
        features["basic_info_count"] > 0
    ]
    .sort_values("basic_info_count", ascending=False)
    [
        [
            "file_id",
            "name",
            "event_count",
            "basic_info_count",
            "close_count",
            "rename_count",
            "time_span_seconds"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# =========================================================
# 7. SIMPAN
# =========================================================

output = (
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\file_level_features_scenario_01.csv"
)

features.to_csv(output, index=False)

print("\nHasil disimpan ke:")
print(output)