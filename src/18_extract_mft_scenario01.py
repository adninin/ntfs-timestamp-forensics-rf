from pathlib import Path
import subprocess
import pandas as pd
import sys


# ============================================================
# KONFIGURASI
# ============================================================

MFTECMD = Path(
    r"C:\Users\atnin\Downloads\MFTECmd\MFTECmd.exe"
)

MFT_FILE = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
    r"\01_Peformance Evaluation\01\$MFT"
)

USN_CSV = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
    r"\CSV_PERFORMANCE\Scenario_01.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
    r"\output\mft_scenario_01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MFT_CSV = OUTPUT_DIR / "Scenario_01_MFT.csv"
TARGET_CONTEXT_CSV = (
    OUTPUT_DIR / "Scenario_01_target_mft.csv"
)
MATCHED_CSV = (
    OUTPUT_DIR / "Scenario_01_usn_mft_match.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET_NAME = (
    "NewFileTime_SI_C_Manipulation.dll"
)


# ============================================================
# VALIDASI FILE
# ============================================================

def validate_inputs():

    print("=" * 90)
    print("VALIDASI INPUT")
    print("=" * 90)

    print(f"\nMFTECmd : {MFTECMD}")
    print(f"$MFT    : {MFT_FILE}")
    print(f"USN CSV : {USN_CSV}")

    if not MFTECMD.exists():

        print("\nERROR: MFTECmd tidak ditemukan.")

        sys.exit(1)

    if not MFT_FILE.exists():

        print("\nERROR: $MFT tidak ditemukan.")

        sys.exit(1)

    if not USN_CSV.exists():

        print("\nERROR: Scenario_01.csv tidak ditemukan.")

        sys.exit(1)

    print("\nSemua input ditemukan.")


# ============================================================
# EKSTRAKSI MFT
# ============================================================

def extract_mft():

    print("\n")
    print("=" * 90)
    print("EKSTRAKSI $MFT SCENARIO 01")
    print("=" * 90)

    # Kalau CSV sudah ada, jangan ekstrak ulang.
    if MFT_CSV.exists():

        print(
            "\nCSV MFT sudah ada:"
        )
        print(MFT_CSV)

        print(
            "\nEkstraksi dilewati."
        )

        return

    command = [
        str(MFTECMD),
        "-f",
        str(MFT_FILE),
        "--csv",
        str(OUTPUT_DIR),
        "--csvf",
        MFT_CSV.name
    ]

    print("\nCommand:")
    print(
        " ".join(
            f'"{x}"'
            if " " in str(x)
            else str(x)
            for x in command
        )
    )

    print("\nMenjalankan MFTECmd...\n")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:

        print(
            "\nSTDERR:"
        )
        print(result.stderr)

    if result.returncode != 0:

        print(
            f"\nMFTECmd gagal. "
            f"Return code: {result.returncode}"
        )

        sys.exit(
            result.returncode
        )

    if not MFT_CSV.exists():

        print(
            "\nERROR: CSV MFT tidak ditemukan "
            "setelah ekstraksi."
        )

        sys.exit(1)

    print(
        "\nEkstraksi MFT berhasil."
    )

    print(
        f"Output: {MFT_CSV}"
    )


# ============================================================
# INSPEKSI KOLOM MFT
# ============================================================

def inspect_mft():

    print("\n")
    print("=" * 90)
    print("INSPEKSI CSV $MFT")
    print("=" * 90)

    df = pd.read_csv(
        MFT_CSV,
        low_memory=False
    )

    print(
        f"\nJumlah baris : {len(df):,}"
    )

    print(
        f"Jumlah kolom: {len(df.columns)}"
    )

    print("\nKolom:")

    for i, col in enumerate(
        df.columns,
        start=1
    ):
        print(
            f"{i:3}. {col}"
        )

    print("\nPreview:")

    print(
        df.head(5).to_string(
            index=False
        )
    )

    return df


# ============================================================
# CARI TARGET DI MFT
# ============================================================

def search_target(df):

    print("\n")
    print("=" * 90)
    print("SEARCH TARGET DI $MFT")
    print("=" * 90)

    # Cari kolom nama
    name_columns = [
        col
        for col in df.columns
        if col.lower() == "name"
    ]

    if not name_columns:

        print(
            "\nERROR: kolom Name tidak ditemukan."
        )

        return pd.DataFrame()

    name_col = name_columns[0]

    mask = (
        df[name_col]
        .astype(str)
        .str.lower()
        .eq(TARGET_NAME.lower())
    )

    target_df = df.loc[
        mask
    ].copy()

    print(
        f"\nTarget: {TARGET_NAME}"
    )

    print(
        f"Ditemukan: {len(target_df)} record"
    )

    if len(target_df) > 0:

        print("\nRecord target:")

        print(
            target_df.to_string(
                index=False
            )
        )

    else:

        print(
            "\nTarget tidak ditemukan "
            "di CSV $MFT."
        )

    target_df.to_csv(
        TARGET_CONTEXT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nOutput target:"
    )
    print(TARGET_CONTEXT_CSV)

    return target_df


# ============================================================
# CARI KOLOM TIMESTAMP
# ============================================================

def inspect_timestamp_columns(df):

    print("\n")
    print("=" * 90)
    print("KOLOM TIMESTAMP")
    print("=" * 90)

    timestamp_keywords = [
        "created",
        "modified",
        "accessed",
        "entry",
        "timestamp",
        "si",
        "fn"
    ]

    timestamp_columns = []

    for col in df.columns:

        col_lower = col.lower()

        if any(
            keyword in col_lower
            for keyword in timestamp_keywords
        ):
            timestamp_columns.append(col)

    print(
        f"\nKandidat kolom timestamp: "
        f"{len(timestamp_columns)}"
    )

    for col in timestamp_columns:

        print(
            f"- {col}"
        )

    return timestamp_columns


# ============================================================
# LOAD USN
# ============================================================

def load_usn():

    print("\n")
    print("=" * 90)
    print("LOAD USN SCENARIO 01")
    print("=" * 90)

    usn = pd.read_csv(
        USN_CSV,
        low_memory=False
    )

    print(
        f"\nUSN rows: {len(usn):,}"
    )

    print("\nKolom USN:")

    for col in usn.columns:

        print(
            f"- {col}"
        )

    return usn


# ============================================================
# MATCH ENTRY + SEQUENCE
# ============================================================

def match_usn_mft(
    usn,
    mft
):

    print("\n")
    print("=" * 90)
    print("MATCH USN ↔ MFT")
    print("=" * 90)

    required_usn = [
        "EntryNumber",
        "SequenceNumber"
    ]

    missing_usn = [
        col
        for col in required_usn
        if col not in usn.columns
    ]

    if missing_usn:

        print(
            "\nKolom USN yang hilang:"
        )

        print(
            missing_usn
        )

        return pd.DataFrame()

    required_mft = [
        "EntryNumber",
        "SequenceNumber"
    ]

    missing_mft = [
        col
        for col in required_mft
        if col not in mft.columns
    ]

    if missing_mft:

        print(
            "\nKolom MFT yang hilang:"
        )

        print(
            missing_mft
        )

        return pd.DataFrame()

    # Pastikan tipe numerik
    for col in required_usn:

        usn[col] = pd.to_numeric(
            usn[col],
            errors="coerce"
        )

    for col in required_mft:

        mft[col] = pd.to_numeric(
            mft[col],
            errors="coerce"
        )

    # Cari target pada USN
    target_usn = usn[
        usn["Name"]
        .astype(str)
        .str.lower()
        .eq(TARGET_NAME.lower())
    ].copy()

    print(
        f"\nTarget rows di USN: "
        f"{len(target_usn)}"
    )

    if target_usn.empty:

        print(
            "Target tidak ditemukan di USN."
        )

        return pd.DataFrame()

    # Ambil record MFT berdasarkan
    # EntryNumber + SequenceNumber
    mft_key = [
        "EntryNumber",
        "SequenceNumber"
    ]

    target_mft = mft[
        mft_key
        + [
            col
            for col in mft.columns
            if col not in mft_key
        ]
    ].copy()

    # Rename kolom MFT agar jelas
    rename_mft = {
        col: f"MFT_{col}"
        for col in target_mft.columns
        if col not in mft_key
    }

    target_mft = target_mft.rename(
        columns=rename_mft
    )

    matched = target_usn.merge(
        target_mft,
        on=mft_key,
        how="left"
    )

    print(
        f"\nHasil match: "
        f"{len(matched)} rows"
    )

    # Statistik match
    mft_columns = [
        col
        for col in matched.columns
        if col.startswith("MFT_")
    ]

    if mft_columns:

        matched_count = (
            matched[mft_columns]
            .notna()
            .any(axis=1)
            .sum()
        )

        print(
            f"Rows dengan data MFT: "
            f"{matched_count}"
        )

        print(
            f"Rows tanpa data MFT: "
            f"{len(matched) - matched_count}"
        )

    matched.to_csv(
        MATCHED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nOutput match:"
    )
    print(MATCHED_CSV)

    return matched


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SCRIPT 18 - MFT SCENARIO 01")
    print("=" * 90)

    # 1
    validate_inputs()

    # 2
    extract_mft()

    # 3
    mft = inspect_mft()

    # 4
    inspect_timestamp_columns(
        mft
    )

    # 5
    search_target(
        mft
    )

    # 6
    usn = load_usn()

    # 7
    match_usn_mft(
        usn,
        mft
    )

    print("\n")
    print("=" * 90)
    print("SCRIPT 18 SELESAI")
    print("=" * 90)


if __name__ == "__main__":
    main()