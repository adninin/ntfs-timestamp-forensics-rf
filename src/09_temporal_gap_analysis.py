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

df = df.dropna(subset=["UpdateTimestamp"])

# =========================================================
# 3. SORTING
# =========================================================

df = df.sort_values(
    ["file_id", "UpdateTimestamp"]
).reset_index(drop=True)

# =========================================================
# 4. HITUNG JARAK ANTAR EVENT
# =========================================================

df["previous_timestamp"] = (
    df.groupby("file_id")["UpdateTimestamp"]
    .shift(1)
)

df["gap_seconds"] = (
    df["UpdateTimestamp"] -
    df["previous_timestamp"]
).dt.total_seconds()

gaps = df["gap_seconds"].dropna()

# =========================================================
# 5. STATISTIK
# =========================================================

print("\n=== DISTRIBUSI GAP ANTAR EVENT ===")

print(
    gaps.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)

print("\n=== GAP PERSENTIL ===")

for p in [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
    print(
        f"P{p*100:.1f}: "
        f"{gaps.quantile(p):.6f} detik"
    )

# =========================================================
# 6. JUMLAH GAP BERDASARKAN INTERVAL
# =========================================================

print("\n=== JUMLAH GAP ===")

thresholds = [
    0.001,
    0.01,
    0.1,
    1,
    5,
    10,
    30,
    60,
    300,
    600,
    3600
]

for threshold in thresholds:

    count = (gaps > threshold).sum()

    percentage = (
        count / len(gaps) * 100
    )

    print(
        f"> {threshold:8} detik : "
        f"{count:8} "
        f"({percentage:.2f}%)"
    )

# =========================================================
# 7. GAP TERBESAR
# =========================================================

print("\n=== GAP TERBESAR ===")

largest = (
    df[
        df["gap_seconds"].notna()
    ]
    .sort_values(
        "gap_seconds",
        ascending=False
    )
    [
        [
            "file_id",
            "Name",
            "UpdateTimestamp",
            "previous_timestamp",
            "gap_seconds",
            "UpdateReasons"
        ]
    ]
    .head(30)
)

print(
    largest.to_string(index=False)
)

# =========================================================
# 8. SIMPAN
# =========================================================

output = (
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\temporal_gap_scenario_01.csv"
)

df.to_csv(output, index=False)

print("\nHasil disimpan ke:")
print(output)