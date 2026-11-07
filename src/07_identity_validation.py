import pandas as pd

FILE_INPUT = r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE\Scenario_01.csv"

print("Membaca dataset...")
df = pd.read_csv(FILE_INPUT)

# Membuat identitas record MFT
df["file_id"] = (
    df["EntryNumber"].astype(str)
    + "-"
    + df["SequenceNumber"].astype(str)
)

print("\n=== JUMLAH FILE ID ===")
print(df["file_id"].nunique())

print("\n=== JUMLAH SOURCE FILE PER FILE ID ===")

source_per_id = (
    df.groupby("file_id")["SourceFile"]
    .nunique()
    .sort_values(ascending=False)
)

print(source_per_id.head(20))

print("\n=== FILE ID DENGAN >1 SOURCE FILE ===")

multi_source = source_per_id[source_per_id > 1]

print("Jumlah:", len(multi_source))

if len(multi_source) > 0:
    print(multi_source.head(20))

print("\n=== JUMLAH EVENT PER FILE ID ===")

events_per_id = (
    df.groupby("file_id")
    .size()
    .sort_values(ascending=False)
)

print(events_per_id.describe())

print("\n=== FILE ID DENGAN EVENT TERBANYAK ===")

top_ids = events_per_id.head(10)

for file_id, count in top_ids.items():

    print("\n--------------------------------")
    print("FILE ID:", file_id)
    print("EVENT:", count)

    subset = df[df["file_id"] == file_id].copy()

    subset["UpdateTimestamp"] = pd.to_datetime(
        subset["UpdateTimestamp"],
        errors="coerce"
    )

    subset = subset.sort_values("UpdateTimestamp")

    print(
        subset[
            [
                "Name",
                "EntryNumber",
                "SequenceNumber",
                "UpdateTimestamp",
                "UpdateReasons"
            ]
        ].head(20).to_string(index=False)
    )