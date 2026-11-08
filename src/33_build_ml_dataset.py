from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

GROUND_TRUTH_DIR = BASE_DIR / "output" / "ground_truth_all"
NEGATIVE_DIR = BASE_DIR / "output" / "negative_candidates_all"
INTEGRATED_DIR = BASE_DIR / "output" / "integrated_dataset_all"

OUTPUT_DIR = BASE_DIR / "output" / "ml_dataset_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ------------------------------------------------------------
# Jumlah negative yang diambil per skenario
#
# Karena positive hanya 1 record per skenario, kita sengaja
# membatasi jumlah negative agar dataset tidak didominasi
# background.
# ------------------------------------------------------------

N_HARD = 30
N_MODERATE = 30
N_RANDOM = 40

# Total negative = 100 / scenario
# Total positive = 1 / scenario
# Total dataset = 101 / scenario


# ============================================================
# HELPER
# ============================================================

def normalize_file_id(value):
    """
    Normalisasi file_id agar konsisten sebagai string.

    Format:
        EntryNumber-SequenceNumber

    Contoh:
        123518-548
    """
    if pd.isna(value):
        return None

    return str(value).strip()


def find_file_id_column(df):
    """
    Mencari nama kolom file_id.
    """
    candidates = [
        "file_id",
        "FileId",
        "FileID",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Kolom file_id tidak ditemukan. "
        f"Kolom tersedia: {list(df.columns)}"
    )


def sample_without_replacement(df, n, seed):
    """
    Sampling maksimal n baris tanpa replacement.
    Kalau jumlah kandidat < n, ambil semuanya.
    """
    if df.empty:
        return df.copy()

    n_actual = min(n, len(df))

    return df.sample(
        n=n_actual,
        random_state=seed,
        replace=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SCRIPT 33 - BUILD ML DATASET")
    print("=" * 80)

    print(f"BASE DIR       : {BASE_DIR}")
    print(f"GROUND TRUTH   : {GROUND_TRUTH_DIR}")
    print(f"NEGATIVE DIR   : {NEGATIVE_DIR}")
    print(f"INTEGRATED DIR : {INTEGRATED_DIR}")
    print(f"OUTPUT DIR     : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Load ground truth summary
    # --------------------------------------------------------

    summary_path = GROUND_TRUTH_DIR / "ground_truth_all_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Ground truth summary tidak ditemukan:\n{summary_path}"
        )

    gt_summary = pd.read_csv(summary_path)

    print("[1] Ground truth summary loaded")
    print(f"    Rows : {len(gt_summary)}")
    print(f"    Columns : {list(gt_summary.columns)}")
    print()

    # Kolom penting
    required_gt_cols = [
        "scenario",
        "target_filename",
        "target_file_ids",
        "status",
    ]

    missing = [
        c for c in required_gt_cols
        if c not in gt_summary.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom ground truth yang diperlukan tidak ditemukan: {missing}"
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_datasets = []
    dataset_summary = []

    # ========================================================
    # PROCESS EACH SCENARIO
    # ========================================================

    for _, gt in gt_summary.iterrows():

        scenario = int(gt["scenario"])

        print("-" * 80)
        print(f"SCENARIO {scenario:02d}")
        print("-" * 80)

        target_filename = str(gt["target_filename"]).strip()
        target_file_id = normalize_file_id(gt["target_file_ids"])

        print(f"Target filename : {target_filename}")
        print(f"Target file_id  : {target_file_id}")

        # ----------------------------------------------------
        # Validasi status
        # ----------------------------------------------------

        if str(gt["status"]).upper() != "OK":
            print(
                f"[SKIP] Ground truth status bukan OK: {gt['status']}"
            )
            continue

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        integrated_path = (
            INTEGRATED_DIR /
            f"Scenario_{scenario:02d}_integrated.csv"
        )

        hard_path = (
            NEGATIVE_DIR /
            f"Scenario_{scenario:02d}_strict_candidates.csv"
        )

        moderate_path = (
            NEGATIVE_DIR /
            f"Scenario_{scenario:02d}_extension_candidates.csv"
        )

        random_path = (
            NEGATIVE_DIR /
            f"Scenario_{scenario:02d}_all_background_candidates.csv"
        )

        # ----------------------------------------------------
        # Check paths
        # ----------------------------------------------------

        for path in [
            integrated_path,
            hard_path,
            moderate_path,
            random_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"File tidak ditemukan:\n{path}"
                )

        # ----------------------------------------------------
        # Load integrated dataset
        # ----------------------------------------------------

        print("[LOAD] Integrated dataset...")

        integrated = pd.read_csv(
            integrated_path,
            low_memory=False
        )

        file_id_col = find_file_id_column(integrated)

        integrated[file_id_col] = (
            integrated[file_id_col]
            .map(normalize_file_id)
        )

        print(f"       Rows : {len(integrated):,}")
        print(f"       Columns : {len(integrated.columns)}")

        # ----------------------------------------------------
        # Pastikan target ada
        # ----------------------------------------------------

        target_mask = (
            integrated[file_id_col] == target_file_id
        )

        target_rows = integrated.loc[target_mask].copy()

        if len(target_rows) != 1:

            print(
                f"[ERROR] Target integrated record = "
                f"{len(target_rows)}"
            )

            raise ValueError(
                f"Scenario {scenario}: target harus tepat 1 record."
            )

        # ----------------------------------------------------
        # Ambil target
        # ----------------------------------------------------

        target_rows["label"] = 1
        target_rows["negative_type"] = "positive_target"
        target_rows["scenario"] = scenario
        target_rows["target_filename"] = target_filename

        # ----------------------------------------------------
        # Load candidate lists
        # ----------------------------------------------------

        print("[LOAD] Negative candidate pools...")

        hard = pd.read_csv(
            hard_path,
            low_memory=False
        )

        moderate = pd.read_csv(
            moderate_path,
            low_memory=False
        )

        random_pool = pd.read_csv(
            random_path,
            low_memory=False
        )

        # ----------------------------------------------------
        # Normalisasi file_id candidate
        # ----------------------------------------------------

        for df in [hard, moderate, random_pool]:

            candidate_id_col = find_file_id_column(df)

            if candidate_id_col != file_id_col:
                df.rename(
                    columns={
                        candidate_id_col: file_id_col
                    },
                    inplace=True
                )

            df[file_id_col] = (
                df[file_id_col]
                .map(normalize_file_id)
            )

        # ----------------------------------------------------
        # Pastikan target tidak masuk negative
        # ----------------------------------------------------

        hard = hard[
            hard[file_id_col] != target_file_id
        ].copy()

        moderate = moderate[
            moderate[file_id_col] != target_file_id
        ].copy()

        random_pool = random_pool[
            random_pool[file_id_col] != target_file_id
        ].copy()

        # ----------------------------------------------------
        # Hindari overlap antar negative pool
        #
        # Hard -> Moderate -> Random
        # ----------------------------------------------------

        hard_ids = set(hard[file_id_col])

        moderate = moderate[
            ~moderate[file_id_col].isin(hard_ids)
        ].copy()

        used_ids = hard_ids.union(
            set(moderate[file_id_col])
        )

        random_pool = random_pool[
            ~random_pool[file_id_col].isin(used_ids)
        ].copy()

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        hard_sample = sample_without_replacement(
            hard,
            N_HARD,
            RANDOM_SEED + scenario
        )

        hard_ids_sampled = set(
            hard_sample[file_id_col]
        )

        moderate_sample = sample_without_replacement(
            moderate,
            N_MODERATE,
            RANDOM_SEED + scenario + 100
        )

        moderate_ids_sampled = set(
            moderate_sample[file_id_col]
        )

        used_sample_ids = (
            hard_ids_sampled |
            moderate_ids_sampled
        )

        random_pool = random_pool[
            ~random_pool[file_id_col].isin(used_sample_ids)
        ].copy()

        random_sample = sample_without_replacement(
            random_pool,
            N_RANDOM,
            RANDOM_SEED + scenario + 200
        )

        # ----------------------------------------------------
        # Candidate IDs
        # ----------------------------------------------------

        hard_ids_sampled = set(
            hard_sample[file_id_col]
        )

        moderate_ids_sampled = set(
            moderate_sample[file_id_col]
        )

        random_ids_sampled = set(
            random_sample[file_id_col]
        )

        # ----------------------------------------------------
        # Ambil row dari integrated dataset
        #
        # Kita TIDAK menggunakan row candidate CSV sebagai
        # fitur ML.
        #
        # Candidate CSV hanya menentukan siapa yang dipilih.
        #
        # Fitur final selalu berasal dari integrated dataset.
        # ----------------------------------------------------

        negative_specs = [
            (
                hard_ids_sampled,
                "hard"
            ),
            (
                moderate_ids_sampled,
                "moderate"
            ),
            (
                random_ids_sampled,
                "random"
            ),
        ]

        negative_parts = []

        for ids, negative_type in negative_specs:

            if not ids:
                continue

            part = integrated[
                integrated[file_id_col].isin(ids)
            ].copy()

            if part.empty:
                print(
                    f"[WARNING] Tidak ada integrated record "
                    f"untuk negative type={negative_type}"
                )
                continue

            part["label"] = 0
            part["negative_type"] = negative_type
            part["scenario"] = scenario
            part["target_filename"] = target_filename

            negative_parts.append(part)

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        scenario_parts = [target_rows]

        if negative_parts:
            scenario_parts.extend(negative_parts)

        scenario_dataset = pd.concat(
            scenario_parts,
            ignore_index=True
        )

        # ----------------------------------------------------
        # Pastikan unique file_id
        # ----------------------------------------------------

        duplicate_ids = (
            scenario_dataset[file_id_col]
            .duplicated()
            .sum()
        )

        if duplicate_ids > 0:

            print(
                f"[ERROR] Ditemukan {duplicate_ids} "
                f"duplicate file_id pada dataset final."
            )

            raise ValueError(
                f"Scenario {scenario}: duplicate file_id."
            )

        # ----------------------------------------------------
        # Jangan masukkan similarity score
        # ----------------------------------------------------

        forbidden_columns = [
            "forensic_similarity_score",
            "same_extension",
            "exact_size",
            "size_within_5pct",
            "size_within_10pct",
            "same_name_type",
            "same_si_fn",
            "extension_exact_size",
            "extension_size_5pct",
            "extension_size_10pct",
            "extension_size5_si_fn",
            "extension_size5_name_type",
        ]

        forbidden_present = [
            c
            for c in forbidden_columns
            if c in scenario_dataset.columns
        ]

        if forbidden_present:

            print(
                "[REMOVE] Candidate-ranking columns "
                "tidak digunakan sebagai ML features:"
            )

            for col in forbidden_present:
                print(f"         - {col}")

            scenario_dataset = scenario_dataset.drop(
                columns=forbidden_present
            )

        # ----------------------------------------------------
        # Dataset summary
        # ----------------------------------------------------

        positive_count = int(
            (scenario_dataset["label"] == 1).sum()
        )

        negative_count = int(
            (scenario_dataset["label"] == 0).sum()
        )

        hard_count = int(
            (
                scenario_dataset["negative_type"]
                == "hard"
            ).sum()
        )

        moderate_count = int(
            (
                scenario_dataset["negative_type"]
                == "moderate"
            ).sum()
        )

        random_count = int(
            (
                scenario_dataset["negative_type"]
                == "random"
            ).sum()
        )

        print()
        print("[RESULT]")
        print(f"       Positive target : {positive_count}")
        print(f"       Hard negative   : {hard_count}")
        print(f"       Moderate        : {moderate_count}")
        print(f"       Random           : {random_count}")
        print(f"       Total            : {len(scenario_dataset)}")

        # ----------------------------------------------------
        # Save scenario dataset
        # ----------------------------------------------------

        scenario_output = (
            OUTPUT_DIR /
            f"Scenario_{scenario:02d}_ml_dataset.csv"
        )

        scenario_dataset.to_csv(
            scenario_output,
            index=False
        )

        print(
            f"[SAVE] {scenario_output}"
        )

        # ----------------------------------------------------
        # Store summary
        # ----------------------------------------------------

        dataset_summary.append({
            "scenario": scenario,
            "target_filename": target_filename,
            "target_file_id": target_file_id,
            "positive_count": positive_count,
            "hard_negative": hard_count,
            "moderate_negative": moderate_count,
            "random_negative": random_count,
            "total_records": len(scenario_dataset),
            "feature_columns": len(scenario_dataset.columns),
            "status": "OK",
        })

        all_datasets.append(
            scenario_dataset
        )

    # ========================================================
    # COMBINE ALL SCENARIOS
    # ========================================================

    print()
    print("=" * 80)
    print("COMBINING ALL SCENARIOS")
    print("=" * 80)

    if not all_datasets:
        raise RuntimeError(
            "Tidak ada dataset scenario yang berhasil dibuat."
        )

    combined = pd.concat(
        all_datasets,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Global duplicate check
    # --------------------------------------------------------

    duplicate_global = (
        combined
        .duplicated(
            subset=["scenario", "file_id"]
        )
        .sum()
    )

    print(
        f"Global duplicate (scenario,file_id): "
        f"{duplicate_global}"
    )

    if duplicate_global > 0:
        raise ValueError(
            "Ditemukan duplicate global."
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print("GLOBAL LABEL DISTRIBUTION")
    print(
        combined["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("GLOBAL NEGATIVE TYPE")
    print(
        combined["negative_type"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Save combined dataset
    # --------------------------------------------------------

    combined_path = (
        OUTPUT_DIR /
        "all_scenarios_ml_dataset.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print()
    print(f"[SAVE] Combined dataset:")
    print(f"       {combined_path}")
    print(f"       Rows    : {len(combined):,}")
    print(f"       Columns : {len(combined.columns)}")

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        dataset_summary
    )

    summary_path = (
        OUTPUT_DIR /
        "ml_dataset_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print()
    print(f"[SAVE] Summary:")
    print(f"       {summary_path}")

    # --------------------------------------------------------
    # Feature inventory
    # --------------------------------------------------------

    metadata_columns = [
        "scenario",
        "file_id",
        "label",
        "negative_type",
        "target_filename",
    ]

    feature_columns = [
        c
        for c in combined.columns
        if c not in metadata_columns
    ]

    feature_inventory = pd.DataFrame({
        "column": feature_columns,
        "dtype": [
            str(combined[c].dtype)
            for c in feature_columns
        ],
        "missing_count": [
            int(combined[c].isna().sum())
            for c in feature_columns
        ],
        "unique_count": [
            int(combined[c].nunique(dropna=True))
            for c in feature_columns
        ],
    })

    feature_inventory_path = (
        OUTPUT_DIR /
        "feature_inventory.csv"
    )

    feature_inventory.to_csv(
        feature_inventory_path,
        index=False
    )

    print(
        f"[SAVE] Feature inventory:"
        f"\n       {feature_inventory_path}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("SCRIPT 33 SELESAI")
    print("=" * 80)

    print()
    print("Output utama:")
    print(f"1. {combined_path}")
    print(f"2. {summary_path}")
    print(f"3. {feature_inventory_path}")
    print()
    print(
        "Dataset siap untuk tahap audit feature dan "
        "Random Forest setelah distribusi dicek."
    )


if __name__ == "__main__":
    main()