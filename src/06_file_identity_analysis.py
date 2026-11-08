import pandas as pd

FILE_INPUT = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"

print("Membaca dataset...")
df = pd.read_csv(FILE_INPUT)

print("\n=== STRUKTUR DATA ===")
print(df.shape)
print(df.columns.tolist())

# Gabungkan EntryNumber + SequenceNumber
df["file_id"] = (
    df["EntryNumber"].astype(str)
    + "-"
    + df["SequenceNumber"].astype(str)
)

print("\n=== JUMLAH IDENTITAS FILE ===")
print(df["file_id"].nunique())

print("\n=== CONTOH FILE ID ===")
print(
    df[
        [
            "Name",
            "EntryNumber",
            "SequenceNumber",
            "ParentEntryNumber",
            "ParentSequenceNumber",
            "file_id"
        ]
    ].head(20).to_string(index=False)
)

# Cek apakah satu Name memiliki banyak EntryNumber
name_entry = (
    df.groupby("Name")["EntryNumber"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n=== NAME DENGAN BANYAK ENTRY NUMBER ===")
print(name_entry.head(20))

# Cek apakah satu EntryNumber memiliki banyak Name
entry_name = (
    df.groupby("EntryNumber")["Name"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n=== ENTRY NUMBER DENGAN BANYAK NAME ===")
print(entry_name.head(20))

output = r"C:\Users\atnin\Downloads\TimestampResearch\output\identity_scenario_01.csv"

df.to_csv(output, index=False)

print(f"\nHasil disimpan ke:")
print(output)