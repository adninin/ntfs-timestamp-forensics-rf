import subprocess
from pathlib import Path
import sys


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

DATASET_DIR = Path(
    r"C:\Users\atnin\Downloads\Dataset_test"
)

MFTECMD = Path(
    r"C:\Users\atnin\Downloads\MFTECmd\MFTECmd.exe"
)

OUTPUT_BASE = (
    BASE_DIR
    / "output"
    / "mft_scenarios"
)

OUTPUT_BASE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SCRIPT 28 - BATCH EXTRACT $MFT SCENARIO 01-12")
print("=" * 70)

print(
    f"\nMFTECmd : {MFTECMD}"
)

print(
    f"Dataset : {DATASET_DIR}"
)

print(
    f"Output  : {OUTPUT_BASE}"
)


# ============================================================
# VALIDATE MFTECMD
# ============================================================

if not MFTECMD.exists():

    raise FileNotFoundError(
        f"MFTECmd tidak ditemukan:\n{MFTECMD}"
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{DATASET_DIR}"
    )


# ============================================================
# SCENARIO LOOP
# ============================================================

results = []


for scenario_number in range(1, 13):

    scenario_name = (
        f"Scenario_{scenario_number:02d}"
    )

    print("\n" + "=" * 70)
    print(
        f"[{scenario_number}/12] {scenario_name}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Find performance folder
    # --------------------------------------------------------

    scenario_folder = (
        DATASET_DIR
        / "01_Peformance Evaluation"
        / f"{scenario_number:02d}"
    )


    if not scenario_folder.exists():

        print(
            f"[WARNING] Folder tidak ditemukan:"
        )

        print(
            scenario_folder
        )

        results.append({
            "scenario": scenario_name,
            "status": "folder_not_found"
        })

        continue


    # --------------------------------------------------------
    # Find $MFT
    # --------------------------------------------------------

    mft_candidates = list(
        scenario_folder.rglob("$MFT")
    )


    if len(mft_candidates) == 0:

        print(
            "[WARNING] $MFT tidak ditemukan."
        )

        results.append({
            "scenario": scenario_name,
            "status": "mft_not_found"
        })

        continue


    if len(mft_candidates) > 1:

        print(
            "[WARNING] Ditemukan lebih dari satu $MFT:"
        )

        for candidate in mft_candidates:
            print(
                f"  - {candidate}"
            )

        print(
            "Menggunakan kandidat pertama."
        )


    mft_path = mft_candidates[0]


    print(
        f"$MFT: {mft_path}"
    )

    try:

        size_mb = (
            mft_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"Size : {size_mb:.2f} MB"
        )

    except Exception:

        size_mb = None


    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_BASE
        / scenario_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Skip if output already exists
    # --------------------------------------------------------

    existing_csv = (
        output_dir
        / f"{scenario_name}_MFT.csv"
    )


    if existing_csv.exists():

        print(
            "[SKIP] Output sudah ada:"
        )

        print(
            existing_csv
        )

        results.append({
            "scenario": scenario_name,
            "status": "already_exists",
            "mft_size_mb": size_mb,
            "output": str(existing_csv)
        })

        continue


    # --------------------------------------------------------
    # MFTECmd command
    # --------------------------------------------------------

    command = [
        str(MFTECMD),
        "-f",
        str(mft_path),
        "--csv",
        str(output_dir),
        "--csvf",
        f"{scenario_name}_MFT.csv"
    ]


    print(
        "\nMenjalankan MFTECmd..."
    )


    print(
        " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )


    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    try:

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )


        # ----------------------------------------------------
        # Show stdout
        # ----------------------------------------------------

        if process.stdout:

            print(
                "\nMFTECmd output:"
            )

            print(
                process.stdout
            )


        # ----------------------------------------------------
        # Show stderr
        # ----------------------------------------------------

        if process.stderr:

            print(
                "\nMFTECmd stderr:"
            )

            print(
                process.stderr
            )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if process.returncode == 0:

            if existing_csv.exists():

                print(
                    f"\n[SUCCESS] {scenario_name}"
                )

                print(
                    f"Output: {existing_csv}"
                )

                results.append({
                    "scenario": scenario_name,
                    "status": "success",
                    "mft_size_mb": size_mb,
                    "output": str(existing_csv)
                })

            else:

                print(
                    "\n[WARNING] MFTECmd selesai "
                    "tetapi output CSV tidak ditemukan."
                )

                results.append({
                    "scenario": scenario_name,
                    "status": "output_not_found",
                    "mft_size_mb": size_mb
                })

        else:

            print(
                f"\n[FAILED] {scenario_name}"
            )

            print(
                f"Return code: "
                f"{process.returncode}"
            )

            results.append({
                "scenario": scenario_name,
                "status": "failed",
                "return_code": process.returncode,
                "mft_size_mb": size_mb
            })


    except Exception as e:

        print(
            f"\n[ERROR] {scenario_name}"
        )

        print(
            repr(e)
        )

        results.append({
            "scenario": scenario_name,
            "status": "exception",
            "error": repr(e),
            "mft_size_mb": size_mb
        })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH EXTRACTION SUMMARY")
print("=" * 70)


for result in results:

    print(
        f"{result['scenario']:15s} "
        f"{result['status']}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

import pandas as pd


summary = pd.DataFrame(
    results
)


summary_file = (
    OUTPUT_BASE
    / "batch_extraction_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


print(
    "\nSummary saved:"
)

print(
    summary_file
)


# ============================================================
# FINAL
# ============================================================

success_count = (
    summary["status"]
    .isin([
        "success",
        "already_exists"
    ])
    .sum()
)

print(
    f"\nCompleted : "
    f"{success_count}/12"
)

print("\n" + "=" * 70)
print("SCRIPT 28 SELESAI")
print("=" * 70)