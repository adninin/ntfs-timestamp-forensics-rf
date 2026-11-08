from pathlib import Path
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
)

OUTPUT_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch\output\mft_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGET FILE
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
# CARI FILE $MFT
# ============================================================

def find_mft_files():

    print("=" * 90)
    print("SEARCHING FOR $MFT")
    print("=" * 90)

    print(f"\nDataset directory:")
    print(DATASET_DIR)

    # Cari nama file yang mengandung MFT
    candidates = []

    for path in DATASET_DIR.rglob("*"):

        if not path.is_file():
            continue

        name_lower = path.name.lower()

        if (
            "$mft" in name_lower
            or "mft" in name_lower
        ):
            candidates.append(path)

    print(
        f"\nJumlah kandidat file MFT: "
        f"{len(candidates)}"
    )

    if candidates:

        print("\nDaftar kandidat:")

        for path in candidates:

            try:
                size_mb = (
                    path.stat().st_size
                    / (1024 * 1024)
                )
            except Exception:
                size_mb = 0

            print(
                f"- {path}"
                f" | {size_mb:.2f} MB"
            )

    else:

        print(
            "\nTidak ditemukan file dengan nama "
            "yang mengandung '$MFT' atau 'MFT'."
        )

    return candidates


# ============================================================
# CARI TARGET PADA DATASET MENTAH
# ============================================================

def search_target_files():

    print("\n")
    print("=" * 90)
    print("SEARCHING TARGET FILES IN RAW DATASET")
    print("=" * 90)

    results = []

    for scenario, target in TARGETS.items():

        print(
            f"\nScenario {scenario}: "
            f"{target}"
        )

        matches = []

        for path in DATASET_DIR.rglob("*"):

            if not path.is_file():
                continue

            # Cari nama file secara case-insensitive
            if target.lower() in path.name.lower():

                matches.append(path)

        if matches:

            print(
                f"  ditemukan: {len(matches)}"
            )

            for match in matches:
                print(
                    f"    {match}"
                )

        else:

            print(
                "  tidak ditemukan sebagai "
                "file fisik dataset."
            )

        results.append({
            "scenario": scenario,
            "target_name": target,
            "raw_file_match_count": len(matches),
            "raw_file_matches": " | ".join(
                str(x)
                for x in matches
            ),
        })

    return pd.DataFrame(results)


# ============================================================
# CARI FOLDER DAN FILE DENGAN STRUKTUR DATASET
# ============================================================

def inspect_dataset_structure():

    print("\n")
    print("=" * 90)
    print("DATASET STRUCTURE")
    print("=" * 90)

    # Hanya tampilkan sampai kedalaman tertentu
    # supaya output tidak terlalu besar.

    rows = []

    for path in DATASET_DIR.rglob("*"):

        try:
            relative = path.relative_to(
                DATASET_DIR
            )
        except ValueError:
            continue

        depth = len(relative.parts)

        if depth > 4:
            continue

        rows.append({
            "type": (
                "DIR"
                if path.is_dir()
                else "FILE"
            ),
            "relative_path": str(relative),
        })

    structure_df = pd.DataFrame(
        rows
    )

    structure_file = (
        OUTPUT_DIR
        / "dataset_structure.csv"
    )

    structure_df.to_csv(
        structure_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nStruktur dataset disimpan:"
    )
    print(structure_file)

    print(
        "\nContoh struktur:"
    )

    print(
        structure_df.head(100)
        .to_string(index=False)
    )

    return structure_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("MFT AVAILABILITY CHECK")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Cari MFT
    # --------------------------------------------------------

    mft_files = find_mft_files()

    # --------------------------------------------------------
    # 2. Cari target file fisik
    # --------------------------------------------------------

    target_df = search_target_files()

    target_output = (
        OUTPUT_DIR
        / "target_raw_file_search.csv"
    )

    target_df.to_csv(
        target_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 3. Struktur dataset
    # --------------------------------------------------------

    structure_df = (
        inspect_dataset_structure()
    )

    # --------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------

    summary = {
        "mft_candidate_count": len(
            mft_files
        ),
        "target_raw_file_matches": int(
            (
                target_df[
                    "raw_file_match_count"
                ] > 0
            ).sum()
        ),
        "total_target": len(
            TARGETS
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_file = (
        OUTPUT_DIR
        / "mft_availability_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSELESAI.")


if __name__ == "__main__":
    main()