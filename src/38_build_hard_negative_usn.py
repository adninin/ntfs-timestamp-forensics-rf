from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

INPUT = (
    ROOT
    / "output"
    / "ml_dataset_all"
    / "all_scenarios_ml_dataset.csv"
)

OUTDIR = ROOT / "output" / "hard_negative_usn"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

N_NEGATIVE_PER_SCENARIO = 100

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("SCRIPT 38 - BUILD USN HARD NEGATIVE DATASET")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"[INFO] Dataset shape: {df.shape}")
print(df["label"].value_counts())

# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

required = [
    "scenario",
    "file_id",
    "label",
    "negative_type",
    "has_usn",
    "usn_event_count",
    "usn_basic_info_count",
    "usn_close_count",
    "usn_data_extend_count",
    "usn_file_create_count",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# TARGET / BACKGROUND
# ============================================================

targets = df[df["label"] == 1].copy()

background = df[df["label"] == 0].copy()

print(f"[INFO] Targets   : {len(targets)}")
print(f"[INFO] Background: {len(background)}")

# ============================================================
# USN-RICH BACKGROUND
# ============================================================

# Hard negative harus benar-benar memiliki USN evidence.
#
# Kita sengaja tidak hanya menggunakan has_usn.
# Event count juga harus > 0.

usn_rich = background[
    (background["has_usn"] == 1)
    & (background["usn_event_count"] > 0)
].copy()

print(
    f"[INFO] USN-rich background candidates: "
    f"{len(usn_rich)}"
)

if len(usn_rich) == 0:
    raise RuntimeError(
        "Tidak ada background dengan USN evidence."
    )

# ============================================================
# HARD-NEGATIVE DISTANCE
# ============================================================

MATCH_FEATURES = [
    "usn_event_count",
    "usn_basic_info_count",
    "usn_close_count",
    "usn_data_extend_count",
    "usn_file_create_count",
]


def prepare_numeric(frame, columns):
    result = frame.copy()

    for col in columns:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    return result


targets = prepare_numeric(
    targets,
    MATCH_FEATURES
)

usn_rich = prepare_numeric(
    usn_rich,
    MATCH_FEATURES
)

# ============================================================
# NORMALIZATION
# ============================================================

# Feature scale berbeda-beda.
# Gunakan robust range dari seluruh USN-rich candidates
# + target.

combined = pd.concat(
    [
        targets[MATCH_FEATURES],
        usn_rich[MATCH_FEATURES],
    ],
    ignore_index=True
)

scales = {}

for col in MATCH_FEATURES:

    values = combined[col].dropna()

    if len(values) == 0:
        scales[col] = 1.0
        continue

    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    iqr = q75 - q25

    if not np.isfinite(iqr) or iqr == 0:
        std = values.std()

        if not np.isfinite(std) or std == 0:
            scales[col] = 1.0
        else:
            scales[col] = float(std)

    else:
        scales[col] = float(iqr)

print("\n[INFO] Matching scales:")
for k, v in scales.items():
    print(f"  {k}: {v:.6f}")

# ============================================================
# DISTANCE FUNCTION
# ============================================================

def hard_negative_distance(target_row, candidate_row):

    distances = []

    for col in MATCH_FEATURES:

        a = target_row[col]
        b = candidate_row[col]

        if pd.isna(a):
            a = 0.0

        if pd.isna(b):
            b = 0.0

        scale = scales[col]

        d = abs(float(a) - float(b)) / scale

        distances.append(d)

    return float(np.mean(distances))


# ============================================================
# SELECT HARD NEGATIVES
# ============================================================

selected_rows = []
selection_log = []

rng = np.random.default_rng(RANDOM_STATE)

scenarios = sorted(
    targets["scenario"].unique()
)

for scenario in scenarios:

    target_rows = targets[
        targets["scenario"] == scenario
    ]

    if len(target_rows) != 1:
        raise RuntimeError(
            f"Scenario {scenario} memiliki "
            f"{len(target_rows)} target; expected 1."
        )

    target = target_rows.iloc[0]

    # --------------------------------------------------------
    # Candidate background.
    #
    # Hindari file_id target yang sama.
    # --------------------------------------------------------

    candidates = usn_rich[
        usn_rich["scenario"] == scenario
    ].copy()

    if len(candidates) < N_NEGATIVE_PER_SCENARIO:

        print(
            f"[WARNING] Scenario {scenario}: "
            f"hanya {len(candidates)} USN-rich candidates."
        )

    # --------------------------------------------------------
    # Calculate distance
    # --------------------------------------------------------

    candidates["hard_negative_distance"] = (
        candidates.apply(
            lambda row:
                hard_negative_distance(
                    target,
                    row
                ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        [
            "hard_negative_distance",
            "file_id",
        ],
        ascending=[
            True,
            True,
        ]
    )

    # --------------------------------------------------------
    # Ambil kandidat terdekat.
    #
    # Untuk menghindari hanya mengambil satu pola yang
    # terlalu identik, ambil pool lebih besar lalu sample
    # deterministic.
    # --------------------------------------------------------

    pool_size = min(
        len(candidates),
        max(
            N_NEGATIVE_PER_SCENARIO * 3,
            N_NEGATIVE_PER_SCENARIO
        )
    )

    pool = candidates.head(pool_size).copy()

    if len(pool) <= N_NEGATIVE_PER_SCENARIO:

        selected = pool.copy()

    else:

        # Ambil 100 terdekat.
        selected = pool.head(
            N_NEGATIVE_PER_SCENARIO
        ).copy()

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    selected = selected[
        selected["file_id"] != target["file_id"]
    ].copy()

    if len(selected) < N_NEGATIVE_PER_SCENARIO:

        raise RuntimeError(
            f"Scenario {scenario}: hanya "
            f"{len(selected)} hard negatives."
        )

    selected = selected.head(
        N_NEGATIVE_PER_SCENARIO
    ).copy()

    # --------------------------------------------------------
    # Assign labels
    # --------------------------------------------------------

    selected["label"] = 0
    selected["negative_type"] = "hard_usn"

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        selection_log.append({
            "scenario": scenario,
            "target_file_id": target["file_id"],
            "negative_file_id": row["file_id"],
            "rank": rank,
            "distance": row[
                "hard_negative_distance"
            ],
            "target_usn_event_count":
                target["usn_event_count"],
            "negative_usn_event_count":
                row["usn_event_count"],
            "target_basic_info_count":
                target["usn_basic_info_count"],
            "negative_basic_info_count":
                row["usn_basic_info_count"],
            "target_close_count":
                target["usn_close_count"],
            "negative_close_count":
                row["usn_close_count"],
            "target_data_extend_count":
                target["usn_data_extend_count"],
            "negative_data_extend_count":
                row["usn_data_extend_count"],
            "target_file_create_count":
                target["usn_file_create_count"],
            "negative_file_create_count":
                row["usn_file_create_count"],
        })

    selected_rows.append(selected)

    print(
        f"Scenario {scenario:>2}: "
        f"target={target['file_id']} | "
        f"hard negatives={len(selected)} | "
        f"best distance="
        f"{selected['hard_negative_distance'].min():.4f} | "
        f"median distance="
        f"{selected['hard_negative_distance'].median():.4f}"
    )


# ============================================================
# BUILD DATASET
# ============================================================

hard_negatives = pd.concat(
    selected_rows,
    ignore_index=True
)

# Target records only
target_records = targets.copy()

target_records["label"] = 1
target_records["negative_type"] = "target"

# Keep original columns
final_columns = [
    c for c in df.columns
    if c in target_records.columns
]

hard_negatives = hard_negatives[
    final_columns
]

target_records = target_records[
    final_columns
]

final_df = pd.concat(
    [
        target_records,
        hard_negatives,
    ],
    ignore_index=True
)

# ============================================================
# SORT
# ============================================================

final_df = final_df.sort_values(
    [
        "scenario",
        "label",
        "file_id",
    ],
    ascending=[
        True,
        False,
        True,
    ]
).reset_index(drop=True)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("\nShape:")
print(final_df.shape)

print("\nLabels:")
print(final_df["label"].value_counts())

print("\nNegative type:")
print(
    final_df["negative_type"]
    .value_counts()
)

print("\nPer scenario:")
print(
    final_df.groupby(
        ["scenario", "label"]
    ).size()
)

# Duplicate check
duplicates = final_df.duplicated(
    subset=["scenario", "file_id"]
).sum()

print(
    f"\nDuplicate (scenario,file_id): "
    f"{duplicates}"
)

if duplicates > 0:
    raise RuntimeError(
        "Ditemukan duplicate (scenario,file_id)."
    )

# Check all negative has USN
neg = final_df[
    final_df["label"] == 0
]

bad_negatives = neg[
    (neg["has_usn"] != 1)
    | (neg["usn_event_count"] <= 0)
]

print(
    f"Negative tanpa USN evidence: "
    f"{len(bad_negatives)}"
)

if len(bad_negatives) > 0:
    raise RuntimeError(
        "Ada hard negative yang tidak memiliki "
        "USN evidence."
    )

# ============================================================
# SAVE
# ============================================================

dataset_path = (
    OUTDIR
    / "hard_negative_usn_dataset.csv"
)

final_df.to_csv(
    dataset_path,
    index=False
)

selection_df = pd.DataFrame(
    selection_log
)

selection_path = (
    OUTDIR
    / "hard_negative_selection_log.csv"
)

selection_df.to_csv(
    selection_path,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for scenario in scenarios:

    sub = final_df[
        final_df["scenario"] == scenario
    ]

    target = sub[
        sub["label"] == 1
    ].iloc[0]

    neg = sub[
        sub["label"] == 0
    ]

    summary_rows.append({
        "scenario": scenario,
        "n_target": int(len([target])),
        "n_negative": len(neg),

        "target_usn_event_count":
            target["usn_event_count"],

        "negative_mean_usn_event_count":
            neg["usn_event_count"].mean(),

        "target_basic_info_count":
            target["usn_basic_info_count"],

        "negative_mean_basic_info_count":
            neg["usn_basic_info_count"].mean(),

        "target_close_count":
            target["usn_close_count"],

        "negative_mean_close_count":
            neg["usn_close_count"].mean(),

        "target_data_extend_count":
            target["usn_data_extend_count"],

        "negative_mean_data_extend_count":
            neg["usn_data_extend_count"].mean(),

        "target_file_create_count":
            target["usn_file_create_count"],

        "negative_mean_file_create_count":
            neg["usn_file_create_count"].mean(),
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTDIR
    / "hard_negative_summary.csv",
    index=False
)

# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(summary_df.to_string(index=False))

print("\nSaved:")
print(dataset_path)
print(selection_path)
print(
    OUTDIR
    / "hard_negative_summary.csv"
)

print("\nDONE.")