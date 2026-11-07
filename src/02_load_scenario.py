import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

FILE_SCENARIO = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"


# ==========================================
# MEMBACA DATA
# ==========================================

print("Membaca Scenario 01...")

df = pd.read_csv(FILE_SCENARIO)

print("\n=== INFORMASI DATA ===")
print(f"Jumlah baris  : {len(df):,}")
print(f"Jumlah kolom  : {len(df.columns)}")

print("\n=== NAMA KOLOM ===")
for column in df.columns:
    print("-", column)


# ==========================================
# INFORMASI TIPE DATA
# ==========================================

print("\n=== TIPE DATA ===")
print(df.dtypes)


# ==========================================
# UPDATE REASONS
# ==========================================

print("\n=== TOP UPDATE REASONS ===")

reason_counts = df["UpdateReasons"].value_counts()

print(reason_counts.head(20))


# ==========================================
# BASIC INFO CHANGE
# ==========================================

basic_info = df[
    df["UpdateReasons"]
    .fillna("")
    .str.contains("BasicInfoChange", case=False)
]

print("\n=== BASIC INFO CHANGE ===")
print(f"Jumlah record : {len(basic_info):,}")


# ==========================================
# CLOSE
# ==========================================

close = df[
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Close", case=False)
]

print("\n=== CLOSE ===")
print(f"Jumlah record : {len(close):,}")


# ==========================================
# FILE CREATE
# ==========================================

file_create = df[
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileCreate", case=False)
]

print("\n=== FILE CREATE ===")
print(f"Jumlah record : {len(file_create):,}")


# ==========================================
# FILE DELETE
# ==========================================

file_delete = df[
    df["UpdateReasons"]
    .fillna("")
    .str.contains("FileDelete", case=False)
]

print("\n=== FILE DELETE ===")
print(f"Jumlah record : {len(file_delete):,}")


# ==========================================
# RENAME
# ==========================================

rename = df[
    df["UpdateReasons"]
    .fillna("")
    .str.contains("Rename", case=False)
]

print("\n=== RENAME ===")
print(f"Jumlah record : {len(rename):,}")


# ==========================================
# EXTENSION
# ==========================================

print("\n=== TOP EXTENSION ===")
print(df["Extension"].value_counts().head(20))


# ==========================================
# FILE ATTRIBUTES
# ==========================================

print("\n=== TOP FILE ATTRIBUTES ===")
print(df["FileAttributes"].value_counts().head(20))


# ==========================================
# CONTOH BASIC INFO CHANGE
# ==========================================

print("\n=== CONTOH BASIC INFO CHANGE ===")

print(
    basic_info[
        [
            "Name",
            "Extension",
            "UpdateTimestamp",
            "UpdateReasons",
            "FileAttributes"
        ]
    ].head(20).to_string(index=False)
)