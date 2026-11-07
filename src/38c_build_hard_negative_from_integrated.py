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

OUTDIR = (
    ROOT
    / "output"
    / "hard_negative_usn_integrated"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

N_HARD_NEGATIVE = 100

MATCH_FEATURES = [
    "usn_event_count",
    "usn_basic_info_count",
    "usn_close_count",
    "usn_data_extend_count",
    "usn_file_create_count",
]

# ============================================================
# VALIDATED TARGET FILE IDs
# ============================================================

TARGET_FILE_IDS = {
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
# HELPERS
# ============================================================

def find_file_id_column(df):

    candidates = [
        "file_id",
        "FileId",
        "FileID",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "file_id column tidak ditemukan."
    )


def normalize_file_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


def numeric(df, columns):

    out = df.copy()

    for c in columns:

        out[c] = pd.to_numeric(
            out[c],
            errors="coerce"
        )

    return out


# ============================================================
# START
# ============================================================

print("=" * 80)
print("SCRIPT 38C")
print("HARD NEGATIVE FROM FULL INTEGRATED DATASET")
print("=" * 80)

all_selected = []
all_logs = []
availability = []


# ============================================================
# LOOP SCENARIOS
# ============================================================

for scenario in range(1, 13):

    print("\n" + "-" * 80)
    print(
        f"SCENARIO {scenario:02d}"
    )
    print("-" * 80)

    path = (
        INTEGRATED_DIR
        / f"Scenario_{scenario:02d}_integrated.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Tidak ditemukan: {path}"
        )

    print(
        f"[INFO] Reading {path.name}"
    )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        f"[INFO] Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # FILE ID
    # --------------------------------------------------------

    file_id_col = find_file_id_column(df)

    df["_file_id_str"] = normalize_file_id(
        df[file_id_col]
    )

    target_id = TARGET_FILE_IDS[scenario]

    print(
        f"[INFO] Expected target: {target_id}"
    )

    # --------------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------------

    target_mask = (
        df["_file_id_str"] == target_id
    )

    target = df[target_mask].copy()

    print(
        f"[INFO] Target rows found: {len(target)}"
    )

    if len(target) != 1:

        # Print nearby IDs untuk debugging
        print(
            "[DEBUG] Matching IDs:"
        )

        print(
            df.loc[
                df["_file_id_str"]
                .str.contains(
                    target_id.split("-")[0],
                    na=False
                ),
                [file_id_col]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"Scenario {scenario}: target "
            f"{target_id} tidak menghasilkan "
            f"tepat 1 row."
        )

    target_row = target.iloc[0]

    # --------------------------------------------------------
    # NUMERIC USN FEATURES
    # --------------------------------------------------------

    missing = [
        c for c in MATCH_FEATURES
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing USN features: {missing}"
        )

    df = numeric(
        df,
        MATCH_FEATURES
    )

    # --------------------------------------------------------
    # NON-TARGET POPULATION
    # --------------------------------------------------------

    candidates = df[
        ~target_mask
    ].copy()

    print(
        f"[INFO] All non-target files: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # USN ACTIVE
    # --------------------------------------------------------

    usn_candidates = candidates[
        candidates["usn_event_count"]
        .fillna(0)
        > 0
    ].copy()

    print(
        f"[INFO] USN-active candidates: "
        f"{len(usn_candidates):,}"
    )

    availability.append({
        "scenario": scenario,
        "target_file_id": target_id,
        "total_non_target":
            len(candidates),
        "usn_active_candidates":
            len(usn_candidates),
    })

    if len(usn_candidates) == 0:

        print(
            "[WARNING] Tidak ada USN-active "
            "non-target."
        )

        continue

    # ========================================================
    # DISTANCE SCALE
    # ========================================================

    scales = {}

    for feature in MATCH_FEATURES:

        values = usn_candidates[
            feature
        ].dropna()

        target_value = target_row[
            feature
        ]

        if not pd.isna(target_value):

            values = pd.concat(
                [
                    values,
                    pd.Series(
                        [target_value]
                    )
                ],
                ignore_index=True
            )

        if len(values) == 0:

            scales[feature] = 1.0
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

                scales[feature] = 1.0

            else:

                scales[feature] = float(
                    std
                )

        else:

            scales[feature] = float(
                iqr
            )

    # ========================================================
    # DISTANCE
    # ========================================================

    distances = []

    for _, row in usn_candidates.iterrows():

        ds = []

        for feature in MATCH_FEATURES:

            a = target_row[feature]
            b = row[feature]

            if pd.isna(a):
                a = 0.0

            if pd.isna(b):
                b = 0.0

            scale = scales[feature]

            ds.append(
                abs(
                    float(a) - float(b)
                ) / scale
            )

        distances.append(
            np.mean(ds)
        )

    usn_candidates[
        "hard_negative_distance"
    ] = distances

    # ========================================================
    # SORT
    # ========================================================

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

    # ========================================================
    # SELECT
    # ========================================================

    n_select = min(
        N_HARD_NEGATIVE,
        len(usn_candidates)
    )

    selected = (
        usn_candidates
        .head(n_select)
        .copy()
    )

    selected["label"] = 0
    selected["negative_type"] = (
        "hard_usn_integrated"
    )

    # ========================================================
    # TARGET
    # ========================================================

    target_copy = target.copy()

    target_copy["label"] = 1
    target_copy["negative_type"] = (
        "target"
    )

    # ========================================================
    # COMBINE
    # ========================================================

    selected = selected.drop(
        columns=[
            "_file_id_str"
        ]
    )

    target_copy = target_copy.drop(
        columns=[
            "_file_id_str"
        ]
    )

    scenario_dataset = pd.concat(
        [
            target_copy,
            selected,
        ],
        ignore_index=True
    )

    scenario_dataset.to_csv(
        OUTDIR
        / f"Scenario_{scenario:02d}_hard_negative.csv",
        index=False
    )

    all_selected.append(
        scenario_dataset
    )

    # ========================================================
    # LOG
    # ========================================================

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        record = {
            "scenario": scenario,
            "target_file_id":
                target_id,
            "negative_file_id":
                str(row[file_id_col]),
            "rank": rank,
            "distance":
                row[
                    "hard_negative_distance"
                ],
        }

        for feature in MATCH_FEATURES:

            record[
                "target_" + feature
            ] = target_row[feature]

            record[
                "negative_" + feature
            ] = row[feature]

        all_logs.append(record)

    print(
        f"[RESULT] Selected: "
        f"{n_select} hard negatives"
    )

    print(
        f"[RESULT] Best distance: "
        f"{selected['hard_negative_distance'].min():.6f}"
    )

    print(
        f"[RESULT] Median distance: "
        f"{selected['hard_negative_distance'].median():.6f}"
    )


# ============================================================
# FINAL COMBINE
# ============================================================

if not all_selected:

    raise RuntimeError(
        "Tidak ada hard negative yang berhasil."
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

log_df.to_csv(
    OUTDIR
    / "hard_negative_usn_selection_log.csv",
    index=False
)


# ============================================================
# AVAILABILITY
# ============================================================

availability_df = pd.DataFrame(
    availability
)

availability_df.to_csv(
    OUTDIR
    / "hard_negative_usn_availability.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("AVAILABILITY")
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
    f"Total rows: {len(combined):,}"
)

print(
    combined["label"]
    .value_counts()
)

print("\nPer scenario:")

print(
    combined.groupby(
        ["scenario", "label"]
    ).size()
)

print("\nSaved:")
print(combined_path)

print(
    OUTDIR
    / "hard_negative_usn_selection_log.csv"
)

print(
    OUTDIR
    / "hard_negative_usn_availability.csv"
)

print("\nDONE.")