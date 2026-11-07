from pathlib import Path
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(r"C:\Users\atnin\Downloads\Dataset_test\CSV_PERFORMANCE")
OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch\output\target_search"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET FILE DARI 12 SKENARIO
# ============================================================

TARGETS = {
    "01": "NewFileTime_SI_C_Manipulation.dll",
    "02": "NewFileTime_SI_M_Manipulation.dll",
    "03": "NewFileTime_SI_MAC_Manipulation.dll",

    "04": "PowerShell_SI_C_Manipulation.dll",
    "05": "PowerShell_SI_M_Manipulation.dll",
    "06": "PowerShell_SI_MAC_Manipulation.dll",

    "07": "nTimestomp_SI_C_Manipulation.dll",
    "08": "nTimestomp_SI_M_Manipulation.dll",
    "09": "nTimestomp_SI_MACE_Manipulation.dll",

    "10": "FileSystemTunneling_SI_C_Change.dll",

    "11": "SetMACE_SI_MACE_Copy_Manipulation.dll",
    "12": "SetMACE_SI_FN_MACE_Manipulation.dll",
}


# ============================================================
# FUNGSI
# ============================================================

def search_target(scenario, target_name):
    """
    Mencari target file berdasarkan kolom Name.
    Belum melakukan labeling.
    """

    csv_path = BASE_DIR / f"Scenario_{scenario}.csv"

    print("\n" + "=" * 80)
    print(f"SCENARIO {scenario}")
    print(f"File   : {csv_path.name}")
    print(f"Target : {target_name}")
    print("=" * 80)

    if not csv_path.exists():
        print("ERROR: CSV tidak ditemukan.")
        return {
            "scenario": scenario,
            "target_name": target_name,
            "csv_exists": False,
            "found": False,
            "row_count": 0,
            "unique_file_id": 0,
            "unique_entry_number": 0,
        }

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Total rows CSV : {len(df):,}")
    print(f"Kolom          : {len(df.columns)}")

    # --------------------------------------------------------
    # Pastikan kolom tersedia
    # --------------------------------------------------------

    required_columns = [
        "Name",
        "EntryNumber",
        "SequenceNumber",
        "UpdateSequenceNumber",
        "UpdateTimestamp",
        "UpdateReasons",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"ERROR: Kolom tidak ditemukan: {missing}")

        return {
            "scenario": scenario,
            "target_name": target_name,
            "csv_exists": True,
            "found": False,
            "row_count": 0,
            "unique_file_id": 0,
            "unique_entry_number": 0,
        }

    # --------------------------------------------------------
    # Bersihkan Name
    # --------------------------------------------------------

    df["Name"] = df["Name"].fillna("").astype(str).str.strip()

    target_lower = target_name.lower()

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    exact_mask = (
        df["Name"]
        .str.lower()
        .eq(target_lower)
    )

    exact_df = df.loc[exact_mask].copy()

    # --------------------------------------------------------
    # SUBSTRING MATCH
    #
    # Untuk memastikan kalau ada variasi path/nama.
    # --------------------------------------------------------

    substring_mask = (
        df["Name"]
        .str.lower()
        .str.contains(target_lower, regex=False, na=False)
    )

    substring_df = df.loc[substring_mask].copy()

    # --------------------------------------------------------
    # FILE ID
    # --------------------------------------------------------

    for result_df in [exact_df, substring_df]:

        if len(result_df) > 0:

            result_df["file_id"] = (
                result_df["EntryNumber"].astype(str)
                + "-"
                + result_df["SequenceNumber"].astype(str)
            )

    # --------------------------------------------------------
    # TAMPILKAN HASIL
    # --------------------------------------------------------

    print(f"\nExact match     : {len(exact_df):,} rows")
    print(f"Substring match : {len(substring_df):,} rows")

    if len(exact_df) > 0:

        print(
            f"Unique file_id  : "
            f"{exact_df['file_id'].nunique():,}"
        )

        print(
            f"Unique EntryNo  : "
            f"{exact_df['EntryNumber'].nunique():,}"
        )

        print("\nContoh record exact match:")

        display_columns = [
            "Name",
            "EntryNumber",
            "SequenceNumber",
            "UpdateSequenceNumber",
            "UpdateTimestamp",
            "UpdateReasons",
            "FileAttributes",
        ]

        display_columns = [
            c for c in display_columns
            if c in exact_df.columns
        ]

        print(
            exact_df[display_columns]
            .head(10)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Simpan hasil exact match
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"Scenario_{scenario}_target_exact.csv"
        )

        exact_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nDisimpan:")
        print(output_file)

    else:

        print("\n>>> TARGET TIDAK DITEMUKAN DENGAN EXACT MATCH.")

    # --------------------------------------------------------
    # Kalau substring match lebih luas dari exact match
    # --------------------------------------------------------

    if (
        len(substring_df) > 0
        and len(substring_df) != len(exact_df)
    ):

        print(
            "\nAda hasil substring tambahan "
            "di luar exact match."
        )

        print(
            substring_df[
                ["Name", "EntryNumber", "SequenceNumber"]
            ]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if len(exact_df) > 0:

        unique_file_id = exact_df["file_id"].nunique()
        unique_entry = exact_df["EntryNumber"].nunique()

    else:

        unique_file_id = 0
        unique_entry = 0

    return {
        "scenario": scenario,
        "target_name": target_name,
        "csv_exists": True,
        "found": len(exact_df) > 0,
        "row_count": len(exact_df),
        "substring_row_count": len(substring_df),
        "unique_file_id": unique_file_id,
        "unique_entry_number": unique_entry,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TARGET FILE SEARCH")
    print("=" * 80)

    print("\nBase directory:")
    print(BASE_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    results = []

    # --------------------------------------------------------
    # Jalankan untuk 12 skenario
    # --------------------------------------------------------

    for scenario, target_name in TARGETS.items():

        result = search_target(
            scenario,
            target_name
        )

        results.append(result)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame(results)

    summary_file = (
        OUTPUT_DIR / "target_search_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 100)
    print("SUMMARY 12 SKENARIO")
    print("=" * 100)

    print(
        summary_df.to_string(index=False)
    )

    print("\n")

    found_count = summary_df["found"].sum()
    total_count = len(summary_df)

    print(
        f"Target ditemukan : "
        f"{found_count}/{total_count}"
    )

    print(
        f"Target tidak ditemukan : "
        f"{total_count - found_count}/{total_count}"
    )

    print("\nSummary disimpan:")
    print(summary_file)

    print("\nSELESAI.")


if __name__ == "__main__":
    main()