from pathlib import Path
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

MFT_CSV = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
    r"\Scenario_01_MFT.csv"
)

USN_CSV = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
    r"\CSV_PERFORMANCE"
    r"\Scenario_01.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = (
    "NewFileTime_SI_C_Manipulation.dll"
)

OUTPUT_TARGET = (
    OUTPUT_DIR
    / "Scenario_01_target_MFT_analysis.csv"
)

OUTPUT_COMPARE = (
    OUTPUT_DIR
    / "Scenario_01_target_USN_MFT_analysis.csv"
)


# ============================================================
# LOAD MFT
# ============================================================

def load_mft():

    print("=" * 90)
    print("LOAD MFT")
    print("=" * 90)

    mft = pd.read_csv(
        MFT_CSV,
        low_memory=False
    )

    print(
        f"\nJumlah MFT record: "
        f"{len(mft):,}"
    )

    print(
        f"Jumlah kolom: "
        f"{len(mft.columns)}"
    )

    return mft


# ============================================================
# CARI TARGET
# ============================================================

def find_target(mft):

    print("\n")
    print("=" * 90)
    print("SEARCH TARGET DI MFT")
    print("=" * 90)

    mask = (
        mft["FileName"]
        .astype(str)
        .str.lower()
        .eq(TARGET.lower())
    )

    target = mft.loc[
        mask
    ].copy()

    print(
        f"\nTarget:"
    )
    print(TARGET)

    print(
        f"\nJumlah record target: "
        f"{len(target)}"
    )

    if target.empty:

        print(
            "\nTARGET TIDAK DITEMUKAN."
        )

        return target

    columns = [
        "EntryNumber",
        "SequenceNumber",
        "InUse",
        "ParentEntryNumber",
        "ParentSequenceNumber",
        "ParentPath",
        "FileName",
        "Extension",
        "FileSize",
        "SI<FN",
        "SiFlags",
        "NameType",
        "Created0x10",
        "Created0x30",
        "LastModified0x10",
        "LastModified0x30",
        "LastRecordChange0x10",
        "LastRecordChange0x30",
        "LastAccess0x10",
        "LastAccess0x30",
        "UpdateSequenceNumber",
        "LogfileSequenceNumber",
        "SourceFile"
    ]

    columns = [
        col
        for col in columns
        if col in target.columns
    ]

    print("\nTarget record:")

    print(
        target[columns].to_string(
            index=False
        )
    )

    target[
        columns
    ].to_csv(
        OUTPUT_TARGET,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nOutput:"
    )
    print(OUTPUT_TARGET)

    return target


# ============================================================
# LOAD USN
# ============================================================

def load_usn():

    print("\n")
    print("=" * 90)
    print("LOAD USN")
    print("=" * 90)

    usn = pd.read_csv(
        USN_CSV,
        low_memory=False
    )

    target_usn = usn[
        usn["Name"]
        .astype(str)
        .str.lower()
        .eq(TARGET.lower())
    ].copy()

    print(
        f"\nTotal USN rows: "
        f"{len(usn):,}"
    )

    print(
        f"Target USN rows: "
        f"{len(target_usn)}"
    )

    return target_usn


# ============================================================
# MATCH USN DAN MFT
# ============================================================

def compare_usn_mft(
    target_usn,
    target_mft
):

    print("\n")
    print("=" * 90)
    print("USN ↔ MFT TARGET COMPARISON")
    print("=" * 90)

    if target_usn.empty:

        print(
            "\nTidak ada target pada USN."
        )

        return

    if target_mft.empty:

        print(
            "\nTidak ada target pada MFT."
        )

        return

    key = [
        "EntryNumber",
        "SequenceNumber"
    ]

    mft_columns = [
        "Created0x10",
        "Created0x30",
        "LastModified0x10",
        "LastModified0x30",
        "LastRecordChange0x10",
        "LastRecordChange0x30",
        "LastAccess0x10",
        "LastAccess0x30",
        "SI<FN",
        "SiFlags",
        "NameType",
        "InUse",
        "FileSize"
    ]

    mft_columns = [
        col
        for col in mft_columns
        if col in target_mft.columns
    ]

    mft_small = target_mft[
        key + mft_columns
    ].copy()

    # Tambahkan prefix supaya jelas
    mft_small = mft_small.rename(
        columns={
            col: f"MFT_{col}"
            for col in mft_columns
        }
    )

    # Pastikan key numerik
    for col in key:

        target_usn[col] = pd.to_numeric(
            target_usn[col],
            errors="coerce"
        )

        mft_small[col] = pd.to_numeric(
            mft_small[col],
            errors="coerce"
        )

    result = target_usn.merge(
        mft_small,
        on=key,
        how="left"
    )

    # Urutkan berdasarkan USN
    if "UpdateSequenceNumber" in result.columns:

        result = result.sort_values(
            "UpdateSequenceNumber"
        )

    print(
        f"\nJumlah hasil comparison: "
        f"{len(result)}"
    )

    display_columns = [
        "Name",
        "EntryNumber",
        "SequenceNumber",
        "UpdateSequenceNumber",
        "UpdateTimestamp",
        "UpdateReasons"
    ]

    display_columns += [
        col
        for col in result.columns
        if col.startswith("MFT_")
    ]

    display_columns = [
        col
        for col in display_columns
        if col in result.columns
    ]

    print("\nHasil:")

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_COMPARE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nOutput:"
    )
    print(OUTPUT_COMPARE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SCRIPT 19 - ANALISIS TARGET MFT")
    print("=" * 90)

    mft = load_mft()

    target_mft = find_target(
        mft
    )

    target_usn = load_usn()

    compare_usn_mft(
        target_usn,
        target_mft
    )

    print("\n")
    print("=" * 90)
    print("SCRIPT 19 SELESAI")
    print("=" * 90)


if __name__ == "__main__":
    main()