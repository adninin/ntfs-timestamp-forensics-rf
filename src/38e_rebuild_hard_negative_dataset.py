from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

INPUT_DIR = (
    BASE_DIR
    / "output"
    / "hard_negative_usn_integrated"
)

OUTPUT_DIR = INPUT_DIR

OUTPUT_FILE = OUTPUT_DIR / "hard_negative_usn_integrated_all_v2.csv"


# ============================================================
# TARGET IDS
# ============================================================

TARGET_IDS = {
    1: "123518-548",
    2: "49190-30",
    3: "49180-14",
    4: "50026-3",
    5: "43647-14",
    6: "50018-4",
    7: "10505-9",
    8: "29719-11",
    9: "49190-13",
    10: "723-8",
    11: "14310-31",
    12: "12761-9",
}


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("REBUILD HARD NEGATIVE DATASET")
print("=" * 80)

scenario_dfs = []

for scenario in range(1, 13):

    input_file = (
        INPUT_DIR
        / f"Scenario_{scenario:02d}_hard_negative.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{input_file}"
        )

    print()
    print(f"[S{scenario:02d}] Reading:")
    print(input_file)

    df = pd.read_csv(input_file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # --------------------------------------------------------
    # Tambahkan scenario
    # --------------------------------------------------------

    df["scenario"] = scenario

    # --------------------------------------------------------
    # Validasi jumlah
    # --------------------------------------------------------

    if len(df) != 101:
        raise ValueError(
            f"S{scenario:02d}: expected 101 rows "
            f"(1 target + 100 negatives), "
            f"got {len(df)}"
        )

    # --------------------------------------------------------
    # Validasi label
    # --------------------------------------------------------

    label_counts = df["label"].value_counts().to_dict()

    if label_counts.get(1, 0) != 1:
        raise ValueError(
            f"S{scenario:02d}: expected exactly 1 positive, "
            f"got {label_counts.get(1, 0)}"
        )

    if label_counts.get(0, 0) != 100:
        raise ValueError(
            f"S{scenario:02d}: expected exactly 100 negatives, "
            f"got {label_counts.get(0, 0)}"
        )

    # --------------------------------------------------------
    # Validasi target ID
    # --------------------------------------------------------

    target_rows = df[df["label"] == 1]

    target_ids_found = target_rows["file_id"].astype(str).tolist()

    expected_target = TARGET_IDS[scenario]

    if target_ids_found != [expected_target]:
        raise ValueError(
            f"S{scenario:02d}: target mismatch.\n"
            f"Expected: {expected_target}\n"
            f"Found: {target_ids_found}"
        )

    # --------------------------------------------------------
    # Validasi hard negative
    # --------------------------------------------------------

    negatives = df[df["label"] == 0]

    if not (negatives["negative_type"] == "hard_usn_integrated").all():
        raise ValueError(
            f"S{scenario:02d}: terdapat negative_type "
            f"yang bukan hard_usn_integrated."
        )

    if not (negatives["has_usn"] == 1).all():
        raise ValueError(
            f"S{scenario:02d}: terdapat hard negative "
            f"tanpa has_usn=1."
        )

    if not (negatives["usn_event_count"] > 0).all():
        raise ValueError(
            f"S{scenario:02d}: terdapat hard negative "
            f"dengan usn_event_count <= 0."
        )

    # --------------------------------------------------------
    # Validasi scenario + file_id di dalam scenario
    # --------------------------------------------------------

    dup = df.duplicated(
        subset=["scenario", "file_id"]
    ).sum()

    if dup != 0:
        raise ValueError(
            f"S{scenario:02d}: ditemukan {dup} "
            f"duplicate (scenario, file_id)."
        )

    scenario_dfs.append(df)

    print("Status: OK")


# ============================================================
# COMBINE
# ============================================================

print()
print("=" * 80)
print("COMBINING")
print("=" * 80)

combined = pd.concat(
    scenario_dfs,
    ignore_index=True
)

print(f"Combined shape: {combined.shape}")


# ============================================================
# GLOBAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("GLOBAL VALIDATION")
print("=" * 80)

# ------------------------------------------------------------
# 1. Shape
# ------------------------------------------------------------

expected_rows = 12 * 101

if len(combined) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, "
        f"got {len(combined)}"
    )

print(f"Rows: {len(combined):,}")
print(f"Columns: {len(combined.columns):,}")


# ------------------------------------------------------------
# 2. Scenario distribution
# ------------------------------------------------------------

scenario_counts = (
    combined["scenario"]
    .value_counts()
    .sort_index()
)

print()
print("Rows per scenario:")
print(scenario_counts)

if not (scenario_counts == 101).all():
    raise ValueError(
        "Tidak semua scenario memiliki 101 rows."
    )


# ------------------------------------------------------------
# 3. Label distribution
# ------------------------------------------------------------

print()
print("Labels:")
print(combined["label"].value_counts().sort_index())

if combined["label"].value_counts().get(1, 0) != 12:
    raise ValueError("Jumlah positive bukan 12.")

if combined["label"].value_counts().get(0, 0) != 1200:
    raise ValueError("Jumlah negative bukan 1200.")


# ------------------------------------------------------------
# 4. Duplicate file_id global
# ------------------------------------------------------------

global_dup_file_id = combined["file_id"].duplicated().sum()

print()
print(f"Duplicate file_id GLOBAL: {global_dup_file_id:,}")

# Duplicate file_id antar scenario boleh terjadi.
# Yang wajib unik adalah scenario + file_id.


# ------------------------------------------------------------
# 5. Duplicate scenario + file_id
# ------------------------------------------------------------

dup_scenario_file = combined.duplicated(
    subset=["scenario", "file_id"]
).sum()

print(
    f"Duplicate (scenario, file_id): "
    f"{dup_scenario_file:,}"
)

if dup_scenario_file != 0:
    raise ValueError(
        "Masih ada duplicate (scenario, file_id)."
    )


# ------------------------------------------------------------
# 6. Negative USN validation
# ------------------------------------------------------------

negatives = combined[
    combined["label"] == 0
].copy()

print()
print(
    "Negative has_usn=1:",
    int((negatives["has_usn"] == 1).sum())
)

print(
    "Negative event_count > 0:",
    int((negatives["usn_event_count"] > 0).sum())
)

if not (negatives["has_usn"] == 1).all():
    raise ValueError(
        "Ada negative yang has_usn != 1."
    )

if not (negatives["usn_event_count"] > 0).all():
    raise ValueError(
        "Ada negative dengan event_count <= 0."
    )


# ------------------------------------------------------------
# 7. Target validation
# ------------------------------------------------------------

targets = combined[
    combined["label"] == 1
].copy()

print()
print("Targets:")

print(
    targets[
        [
            "scenario",
            "file_id",
            "label",
            "negative_type",
            "usn_event_count",
            "usn_basic_info_count",
            "usn_close_count",
            "usn_file_create_count",
            "usn_data_extend_count",
            "hard_negative_distance",
        ]
    ]
    .sort_values("scenario")
    .to_string(index=False)
)


# ------------------------------------------------------------
# 8. Negative type
# ------------------------------------------------------------

print()
print("Negative types:")
print(
    combined["negative_type"]
    .value_counts()
)


# ------------------------------------------------------------
# 9. Per-scenario label summary
# ------------------------------------------------------------

print()
print("Per-scenario label summary:")

summary = (
    combined
    .groupby(["scenario", "label"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

print(summary)


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 80)
print("SAVING")
print("=" * 80)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("=" * 80)
print("REBUILD COMPLETE")
print("=" * 80)