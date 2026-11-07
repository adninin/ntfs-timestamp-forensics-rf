from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

INTEGRATED_DIR = (
    ROOT
    / "output"
    / "integrated_dataset_all"
)

GROUND_TRUTH_DIR = (
    ROOT
    / "output"
    / "ground_truth_all"
)

OUTDIR = (
    ROOT
    / "output"
    / "hard_negative_usn_integrated"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

N_HARD_NEGATIVE = 100

MATCH_FEATURES = [
    "usn_event_count",
    "usn_basic_info_count",
    "usn_close_count",
    "usn_data_extend_count",
    "usn_file_create_count",
]

# ============================================================
# HELPER
# ============================================================

def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    return None


def numeric(df, columns):

    out = df.copy()

    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce"
            )

    return out


# ============================================================
# START
# ============================================================

print("=" * 80)
print("SCRIPT 38B")
print("BUILD HARD NEGATIVES FROM FULL INTEGRATED DATASET")
print("=" * 80)


all_selected = []
all_logs = []
availability = []


# ============================================================
# PROCESS EACH SCENARIO
# ============================================================

for scenario in range(1, 13):

    integrated_path = (
        INTEGRATED_DIR
        / f"Scenario_{scenario:02d}_integrated.csv"
    )

    if not integrated_path.exists():

        raise FileNotFoundError(
            f"Missing: {integrated_path}"
        )

    print("\n" + "-" * 80)
    print(
        f"SCENARIO {scenario:02d}"
    )
    print("-" * 80)

    print(
        f"[INFO] Reading {integrated_path.name}"
    )

    df = pd.read_csv(
        integrated_path
    )

    print(
        f"[INFO] Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_path = (
        GROUND_TRUTH_DIR
        / f"Scenario_{scenario:02d}_ground_truth.csv"
    )

    if not gt_path.exists():

        # fallback naming
        possible = list(
            GROUND_TRUTH_DIR.glob(
                f"*{scenario:02d}*.csv"
            )
        )

        if not possible:
            raise FileNotFoundError(
                f"Ground truth not found for "
                f"scenario {scenario}"
            )

        gt_path = possible[0]

    gt = pd.read_csv(gt_path)

    print(
        f"[INFO] Ground truth rows: {len(gt)}"
    )

    # --------------------------------------------------------
    # Identify file_id
    # --------------------------------------------------------

    file_id_col = find_column(
        df,
        [
            "file_id",
            "FileId",
            "FileID",
        ]
    )

    if file_id_col is None:

        raise RuntimeError(
            f"file_id column not found in "
            f"scenario {scenario}"
        )

    gt_file_id_col = find_column(
        gt,
        [
            "file_id",
            "FileId",
            "FileID",
        ]
    )

    if gt_file_id_col is None:

        raise RuntimeError(
            f"file_id column not found in GT "
            f"scenario {scenario}"
        )

    target_ids = set(
        gt[gt_file_id_col]
        .astype(str)
        .str.strip()
    )

    df["_file_id_str"] = (
        df[file_id_col]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Ensure USN features exist
    # --------------------------------------------------------

    missing_features = [
        c for c in MATCH_FEATURES
        if c not in df.columns
    ]

    if missing_features:

        raise RuntimeError(
            f"Scenario {scenario}: missing "
            f"USN features {missing_features}"
        )

    df = numeric(
        df,
        MATCH_FEATURES
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = df[
        df["_file_id_str"].isin(
            target_ids
        )
    ].copy()

    if len(target) == 0:

        raise RuntimeError(
            f"Scenario {scenario}: "
            "target tidak ditemukan."
        )

    if len(target) > 1:

        print(
            f"[WARNING] Target rows = {len(target)}"
        )

    target_row = target.iloc[0]

    # --------------------------------------------------------
    # Candidate population
    #
    # Semua non-target file.
    # --------------------------------------------------------

    candidates = df[
        ~df["_file_id_str"].isin(
            target_ids
        )
    ].copy()

    total_candidates = len(candidates)

    # --------------------------------------------------------
    # USN-rich candidates
    #
    # Tidak menggunakan has_usn karena kita ingin
    # benar-benar melihat event activity.
    # --------------------------------------------------------

    usn_candidates = candidates[
        candidates["usn_event_count"].fillna(0) > 0
    ].copy()

    n_usn = len(usn_candidates)

    print(
        f"[INFO] All non-target files : "
        f"{total_candidates:,}"
    )

    print(
        f"[INFO] USN-active candidates: "
        f"{n_usn:,}"
    )

    availability.append({
        "scenario": scenario,
        "total_non_target_files":
            total_candidates,
        "usn_active_candidates":
            n_usn,
    })

    if n_usn == 0:

        print(
            "[WARNING] No USN-active "
            "non-target candidates."
        )

        continue

    # --------------------------------------------------------
    # Distance scale
    # --------------------------------------------------------

    scale_values = {}

    for feature in MATCH_FEATURES:

        values = pd.concat(
            [
                usn_candidates[feature],
                pd.Series(
                    [target_row[feature]]
                ),
            ],
            ignore_index=True
        )

        values = values.dropna()

        if len(values) == 0:

            scale_values[feature] = 1.0
            continue

        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)

        iqr = q75 - q25

        if (
            pd.isna(iqr)
            or iqr == 0
        ):

            std = values.std()

            if (
                pd.isna(std)
                or std == 0
            ):
                scale_values[feature] = 1.0

            else:
                scale_values[feature] = float(
                    std
                )

        else:

            scale_values[feature] = float(
                iqr
            )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distances = []

    for _, row in usn_candidates.iterrows():

        feature_distances = []

        for feature in MATCH_FEATURES:

            a = target_row[feature]
            b = row[feature]

            if pd.isna(a):
                a = 0.0

            if pd.isna(b):
                b = 0.0

            scale = scale_values[feature]

            distance = (
                abs(
                    float(a) - float(b)
                )
                / scale
            )

            feature_distances.append(
                distance
            )

        distances.append(
            np.mean(
                feature_distances
            )
        )

    usn_candidates[
        "hard_negative_distance"
    ] = distances

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    usn_candidates = (
        usn_candidates
        .sort_values(
            [
                "hard_negative_distance",
                "_file_id_str",
            ],
            ascending=True
        )
    )

    # --------------------------------------------------------
    # Select top N
    # --------------------------------------------------------

    n_select = min(
        N_HARD_NEGATIVE,
        len(usn_candidates)
    )

    selected = (
        usn_candidates
        .head(n_select)
        .copy()
    )

    selected["scenario"] = scenario
    selected["label"] = 0
    selected["negative_type"] = (
        "hard_usn_integrated"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_copy = target.iloc[[0]].copy()

    target_copy["scenario"] = scenario
    target_copy["label"] = 1
    target_copy["negative_type"] = (
        "target"
    )

    # --------------------------------------------------------
    # Keep original columns
    # --------------------------------------------------------

    selected = selected.drop(
        columns=[
            "_file_id_str",
        ],
        errors="ignore"
    )

    target_copy = target_copy.drop(
        columns=[
            "_file_id_str",
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Save scenario-specific candidate
    # --------------------------------------------------------

    scenario_output = pd.concat(
        [
            target_copy,
            selected,
        ],
        ignore_index=True
    )

    scenario_output = scenario_output.sort_values(
        [
            "label",
            file_id_col,
        ],
        ascending=[
            False,
            True,
        ]
    )

    scenario_output.to_csv(
        OUTDIR
        / f"Scenario_{scenario:02d}_hard_negative.csv",
        index=False
    )

    all_selected.append(
        scenario_output
    )

    # --------------------------------------------------------
    # Selection log
    # --------------------------------------------------------

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        log = {
            "scenario": scenario,
            "target_file_id":
                str(
                    target_row[
                        file_id_col
                    ]
                ),
            "negative_file_id":
                str(
                    row[file_id_col]
                ),
            "rank": rank,
            "distance":
                row[
                    "hard_negative_distance"
                ],
        }

        for feature in MATCH_FEATURES:

            log[
                f"target_{feature}"
            ] = target_row[feature]

            log[
                f"negative_{feature}"
            ] = row[feature]

        all_logs.append(log)

    print(
        f"[RESULT] Selected "
        f"{n_select} hard negatives."
    )

    print(
        "[RESULT] Best distance: "
        f"{selected['hard_negative_distance'].min():.6f}"
    )

    print(
        "[RESULT] Median distance: "
        f"{selected['hard_negative_distance'].median():.6f}"
    )


# ============================================================
# COMBINE
# ============================================================

if not all_selected:

    raise RuntimeError(
        "Tidak ada scenario yang menghasilkan "
        "hard negative."
    )

combined = pd.concat(
    all_selected,
    ignore_index=True
)

combined_path = (
    OUTDIR
    / "hard_negative_usn_integrated_all.csv"
)

combined.to_csv(
    combined_path,
    index=False
)


# ============================================================
# LOG
# ============================================================

log_df = pd.DataFrame(
    all_logs
)

log_path = (
    OUTDIR
    / "hard_negative_usn_selection_log.csv"
)

log_df.to_csv(
    log_path,
    index=False
)


# ============================================================
# AVAILABILITY
# ============================================================

availability_df = pd.DataFrame(
    availability
)

availability_path = (
    OUTDIR
    / "hard_negative_usn_availability.csv"
)

availability_df.to_csv(
    availability_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL AVAILABILITY")
print("=" * 80)

print(
    availability_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

print(
    f"Rows: {len(combined):,}"
)

print(
    combined["label"]
    .value_counts()
)

print(
    "\nPer scenario:"
)

print(
    combined.groupby(
        ["scenario", "label"]
    ).size()
)

print("\nSaved:")
print(combined_path)
print(log_path)
print(availability_path)

print("\nDONE.")