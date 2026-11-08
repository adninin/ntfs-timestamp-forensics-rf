from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SCRIPT 32
# NEGATIVE CANDIDATE ANALYSIS
#
# Tujuan:
#   1. Membaca verified ground truth dari Script 31
#   2. Mengambil target file_id
#   3. Mengambil extension + NameType dari RAW MFT
#   4. Mengambil file_size + SI<FN dari MFT record features
#   5. Membentuk background / negative candidate pool
#   6. Mengukur kemiripan forensik kandidat terhadap target
#
# CATATAN:
#   - Target positif = experimental target identity
#   - Background candidate TIDAK otomatis berarti "normal"
#   - Similarity score hanya untuk ranking kandidat
#   - Similarity score BUKAN label ML
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

GROUND_TRUTH_SUMMARY = (
    BASE_DIR
    / "output"
    / "ground_truth_all"
    / "ground_truth_all_summary.csv"
)

RAW_MFT_DIR = (
    BASE_DIR
    / "output"
    / "mft_scenarios"
)

MFT_FEATURE_DIR = (
    BASE_DIR
    / "output"
    / "mft_record_features_all"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "negative_candidates_all"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Mencari nama kolom secara case-insensitive.

    Contoh:
        extension
        Extension
        EXTENSION

    akan dianggap sama.
    """

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in normalized:
            return normalized[key]

    return None


# ------------------------------------------------------------

def normalize_file_id(value):
    """
    Mengubah file_id menjadi string yang konsisten.

    Contoh:
        123518-548 -> '123518-548'
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


# ------------------------------------------------------------

def normalize_extension(value):
    """
    Normalisasi extension.

    Contoh:
        .DLL -> dll
        DLL  -> dll
        .dll -> dll
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if text.startswith("."):
        text = text[1:]

    return text


# ------------------------------------------------------------

def normalize_name_type(value):
    """
    Normalisasi NameType.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ------------------------------------------------------------

def safe_bool(value):
    """
    Konversi berbagai representasi boolean menjadi:
        True
        False
        None
    """

    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y"
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n"
    }:
        return False

    return None


# ------------------------------------------------------------

def numeric(value):
    """
    Konversi nilai menjadi float.
    """

    try:

        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:

        return np.nan


# ------------------------------------------------------------

def unique_join(values, normalizer=None):
    """
    Menggabungkan nilai unik menjadi string.

    Digunakan ketika satu MFT record memiliki
    beberapa filename representation.
    """

    result = set()

    for value in values:

        if pd.isna(value):
            continue

        if normalizer is not None:
            value = normalizer(value)
        else:
            value = str(value).strip()

        if value != "":
            result.add(value)

    return " | ".join(sorted(result))


# ============================================================
# START
# ============================================================

print()
print("=" * 100)
print("NEGATIVE CANDIDATE ANALYSIS - FINAL VERSION")
print("=" * 100)
print()

print("Base directory:")
print(BASE_DIR)

print()
print("Ground truth:")
print(GROUND_TRUTH_SUMMARY)

print()
print("Raw MFT directory:")
print(RAW_MFT_DIR)

print()
print("MFT feature directory:")
print(MFT_FEATURE_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()


# ============================================================
# VALIDATE GROUND TRUTH SUMMARY
# ============================================================

if not GROUND_TRUTH_SUMMARY.exists():

    raise FileNotFoundError(
        "\nGround truth summary tidak ditemukan:\n"
        f"{GROUND_TRUTH_SUMMARY}"
    )


# ============================================================
# LOAD GROUND TRUTH SUMMARY
# ============================================================

print("=" * 100)
print("LOADING VERIFIED GROUND TRUTH")
print("=" * 100)

gt = pd.read_csv(
    GROUND_TRUTH_SUMMARY,
    low_memory=False
)

print()
print(
    f"Ground truth rows: {len(gt):,}"
)

print()
print("Ground truth columns:")

for column in gt.columns:
    print(
        f"  - {column}"
    )

print()


# ============================================================
# DETECT GROUND TRUTH COLUMNS
# ============================================================

scenario_col = find_column(
    gt,
    [
        "scenario"
    ]
)

filename_col = find_column(
    gt,
    [
        "target_filename"
    ]
)

target_id_col = find_column(
    gt,
    [
        "target_file_ids",
        "target_file_id"
    ]
)

status_col = find_column(
    gt,
    [
        "status"
    ]
)


# ============================================================
# VALIDATE GROUND TRUTH COLUMNS
# ============================================================

missing_gt_columns = []

if scenario_col is None:
    missing_gt_columns.append(
        "scenario"
    )

if filename_col is None:
    missing_gt_columns.append(
        "target_filename"
    )

if target_id_col is None:
    missing_gt_columns.append(
        "target_file_ids / target_file_id"
    )


if missing_gt_columns:

    raise ValueError(
        "\nKolom ground truth yang dibutuhkan tidak ditemukan:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_gt_columns
        )
    )


# ============================================================
# DISPLAY VERIFIED MAPPING
# ============================================================

print("=" * 100)
print("VERIFIED GROUND TRUTH MAPPING")
print("=" * 100)

print()

print(
    gt[
        [
            scenario_col,
            filename_col,
            target_id_col
        ]
    ].to_string(
        index=False
    )
)

print()


# ============================================================
# PREPARE SUMMARY
# ============================================================

summary_rows = []


# ============================================================
# PROCESS EACH SCENARIO
# ============================================================

for _, gt_row in gt.iterrows():

    # --------------------------------------------------------
    # READ GROUND TRUTH
    # --------------------------------------------------------

    scenario = int(
        gt_row[scenario_col]
    )

    target_filename = str(
        gt_row[filename_col]
    ).strip()

    target_file_id = normalize_file_id(
        gt_row[target_id_col]
    )

    scenario_text = (
        f"{scenario:02d}"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        f"SCENARIO {scenario}"
    )
    print(
        f"Target : {target_filename}"
    )
    print(
        f"File ID: {target_file_id}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # VALIDATE TARGET FILE ID
    # --------------------------------------------------------

    if target_file_id == "":

        print(
            "[ERROR] Target file_id kosong."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "TARGET_FILE_ID_EMPTY"
        })

        continue

    # ========================================================
    # PATH RAW MFT
    # ========================================================

    raw_mft_file = (
        RAW_MFT_DIR
        / f"Scenario_{scenario_text}"
        / f"Scenario_{scenario_text}_MFT.csv"
    )

    # ========================================================
    # PATH MFT FEATURES
    # ========================================================

    feature_file = (
        MFT_FEATURE_DIR
        / f"Scenario_{scenario_text}_MFT_record_features.csv"
    )

    # --------------------------------------------------------
    # CHECK RAW MFT
    # --------------------------------------------------------

    if not raw_mft_file.exists():

        print()
        print(
            "[ERROR] Raw MFT tidak ditemukan:"
        )

        print(
            raw_mft_file
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "RAW_MFT_NOT_FOUND"
        })

        continue

    # --------------------------------------------------------
    # CHECK MFT FEATURES
    # --------------------------------------------------------

    if not feature_file.exists():

        print()
        print(
            "[ERROR] MFT feature tidak ditemukan:"
        )

        print(
            feature_file
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "MFT_FEATURE_NOT_FOUND"
        })

        continue

    # ========================================================
    # LOAD RAW MFT
    # ========================================================

    print()
    print(
        f"Loading raw MFT:"
    )

    print(
        raw_mft_file
    )

    raw_mft = pd.read_csv(
        raw_mft_file,
        low_memory=False
    )

    print()
    print(
        f"Raw MFT rows: "
        f"{len(raw_mft):,}"
    )

    # ========================================================
    # DETECT RAW MFT COLUMNS
    # ========================================================

    raw_file_id_col = find_column(
        raw_mft,
        [
            "file_id"
        ]
    )

    raw_entry_col = find_column(
        raw_mft,
        [
            "EntryNumber"
        ]
    )

    raw_sequence_col = find_column(
        raw_mft,
        [
            "SequenceNumber"
        ]
    )

    raw_filename_col = find_column(
        raw_mft,
        [
            "FileName",
            "Filename",
            "Name"
        ]
    )

    raw_extension_col = find_column(
        raw_mft,
        [
            "Extension",
            "extension"
        ]
    )

    raw_name_type_col = find_column(
        raw_mft,
        [
            "NameType",
            "name_type"
        ]
    )

    raw_filesize_col = find_column(
        raw_mft,
        [
            "FileSize",
            "file_size"
        ]
    )

    # ========================================================
    # DISPLAY RAW MFT SCHEMA
    # ========================================================

    print()
    print("RAW MFT columns:")

    print(
        f"  file_id    : "
        f"{raw_file_id_col}"
    )

    print(
        f"  Entry      : "
        f"{raw_entry_col}"
    )

    print(
        f"  Sequence   : "
        f"{raw_sequence_col}"
    )

    print(
        f"  FileName   : "
        f"{raw_filename_col}"
    )

    print(
        f"  Extension  : "
        f"{raw_extension_col}"
    )

    print(
        f"  NameType   : "
        f"{raw_name_type_col}"
    )

    print(
        f"  FileSize   : "
        f"{raw_filesize_col}"
    )

    # ========================================================
    # VALIDATE RAW MFT COLUMNS
    # ========================================================

    missing_raw_columns = []

    if raw_entry_col is None:
        missing_raw_columns.append(
            "EntryNumber"
        )

    if raw_sequence_col is None:
        missing_raw_columns.append(
            "SequenceNumber"
        )

    if raw_filename_col is None:
        missing_raw_columns.append(
            "FileName"
        )

    if raw_extension_col is None:
        missing_raw_columns.append(
            "Extension"
        )

    if raw_name_type_col is None:
        missing_raw_columns.append(
            "NameType"
        )


    if missing_raw_columns:

        print()
        print(
            "[ERROR] Raw MFT columns missing:"
        )

        for column in missing_raw_columns:

            print(
                f"  - {column}"
            )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "RAW_MFT_COLUMNS_MISSING"
        })

        continue


    # ========================================================
    # BUILD / NORMALIZE RAW MFT FILE ID
    # ========================================================

    if raw_file_id_col is not None:

        print()
        print(
            "[INFO] Raw MFT sudah memiliki file_id."
        )

        raw_mft[
            "_file_id_normalized"
        ] = (
            raw_mft[
                raw_file_id_col
            ]
            .apply(
                normalize_file_id
            )
        )

    else:

        print()
        print(
            "[INFO] Raw MFT tidak memiliki file_id."
        )

        print(
            "[INFO] Membentuk file_id dari:"
        )

        print(
            "       EntryNumber + SequenceNumber"
        )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        entry_numeric = pd.to_numeric(
            raw_mft[
                raw_entry_col
            ],
            errors="coerce"
        )

        sequence_numeric = pd.to_numeric(
            raw_mft[
                raw_sequence_col
            ],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Validate numeric IDs
        # ----------------------------------------------------

        invalid_ids = (
            entry_numeric.isna()
            |
            sequence_numeric.isna()
        )

        invalid_count = int(
            invalid_ids.sum()
        )

        if invalid_count > 0:

            print()
            print(
                f"[WARNING] "
                f"{invalid_count:,} raw MFT rows "
                f"memiliki Entry/Sequence tidak valid."
            )

        # ----------------------------------------------------
        # Build ID
        # ----------------------------------------------------

        raw_mft[
            "_file_id_normalized"
        ] = (
            entry_numeric
            .astype("Int64")
            .astype(str)
            +
            "-"
            +
            sequence_numeric
            .astype("Int64")
            .astype(str)
        )

        # ----------------------------------------------------
        # Remove invalid generated IDs
        # ----------------------------------------------------

        raw_mft.loc[
            invalid_ids,
            "_file_id_normalized"
        ] = ""


    # ========================================================
    # FIND TARGET RAW MFT RECORD
    # ========================================================

    target_raw = raw_mft[
        raw_mft[
            "_file_id_normalized"
        ]
        ==
        target_file_id
    ].copy()

    print()
    print(
        f"Target raw MFT rows: "
        f"{len(target_raw)}"
    )

    if len(target_raw) == 0:

        print()
        print(
            "[ERROR] Target file_id "
            "tidak ditemukan pada raw MFT."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "TARGET_NOT_FOUND_RAW_MFT"
        })

        continue


    # ========================================================
    # FIND TARGET FILENAME REPRESENTATION
    # ========================================================

    target_filename_mask = (
        target_raw[
            raw_filename_col
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        target_filename.lower()
    )

    target_filename_matches = (
        target_raw[
            target_filename_mask
        ]
        .copy()
    )

    print(
        f"Target filename matches: "
        f"{len(target_filename_matches)}"
    )


    # --------------------------------------------------------
    # Select target representation
    # --------------------------------------------------------

    if len(target_filename_matches) > 0:

        target_record = (
            target_filename_matches
            .iloc[0]
        )

    else:

        print()
        print(
            "[WARNING] Target file_id ditemukan,"
        )

        print(
            "tetapi exact target filename "
            "tidak ditemukan pada record."
        )

        print(
            "Menggunakan representasi pertama "
            "dari file_id tersebut."
        )

        target_record = (
            target_raw
            .iloc[0]
        )


    # ========================================================
    # TARGET EXTENSION / NAME TYPE
    # ========================================================

    target_extension = normalize_extension(
        target_record[
            raw_extension_col
        ]
    )

    target_name_type = normalize_name_type(
        target_record[
            raw_name_type_col
        ]
    )


    # ========================================================
    # LOAD MFT RECORD FEATURES
    # ========================================================

    print()
    print(
        f"Loading MFT record features:"
    )

    print(
        feature_file
    )

    features = pd.read_csv(
        feature_file,
        low_memory=False
    )

    print()
    print(
        f"MFT feature rows: "
        f"{len(features):,}"
    )


    # ========================================================
    # DETECT FEATURE COLUMNS
    # ========================================================

    feature_file_id_col = find_column(
        features,
        [
            "file_id"
        ]
    )

    feature_size_col = find_column(
        features,
        [
            "file_size"
        ]
    )

    feature_si_fn_col = find_column(
        features,
        [
            "si_fn_true"
        ]
    )


    # ========================================================
    # DISPLAY FEATURE COLUMNS
    # ========================================================

    print()
    print(
        "MFT feature columns:"
    )

    print(
        f"  file_id    : "
        f"{feature_file_id_col}"
    )

    print(
        f"  file_size  : "
        f"{feature_size_col}"
    )

    print(
        f"  si_fn_true : "
        f"{feature_si_fn_col}"
    )


    # ========================================================
    # VALIDATE FEATURE COLUMNS
    # ========================================================

    missing_feature_columns = []

    if feature_file_id_col is None:

        missing_feature_columns.append(
            "file_id"
        )

    if feature_size_col is None:

        missing_feature_columns.append(
            "file_size"
        )

    if feature_si_fn_col is None:

        missing_feature_columns.append(
            "si_fn_true"
        )


    if missing_feature_columns:

        print()
        print(
            "[ERROR] MFT feature columns missing:"
        )

        for column in missing_feature_columns:

            print(
                f"  - {column}"
            )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "MFT_FEATURE_COLUMNS_MISSING"
        })

        continue


    # ========================================================
    # NORMALIZE FEATURE FILE ID
    # ========================================================

    features[
        "_file_id_normalized"
    ] = (
        features[
            feature_file_id_col
        ]
        .apply(
            normalize_file_id
        )
    )


    # ========================================================
    # FIND TARGET FEATURE RECORD
    # ========================================================

    target_feature = features[
        features[
            "_file_id_normalized"
        ]
        ==
        target_file_id
    ].copy()

    print()
    print(
        f"Target feature records: "
        f"{len(target_feature)}"
    )


    if len(target_feature) == 0:

        print(
            "[ERROR] Target tidak ditemukan "
            "pada MFT record features."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "status":
                "TARGET_FEATURE_NOT_FOUND"
        })

        continue


    if len(target_feature) > 1:

        print(
            "[ERROR] Target memiliki lebih dari "
            "satu MFT feature record."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "target_feature_records":
                len(target_feature),

            "status":
                "TARGET_FEATURE_MULTIPLE"
        })

        continue


    # ========================================================
    # TARGET FEATURE ATTRIBUTES
    # ========================================================

    target_feature_record = (
        target_feature
        .iloc[0]
    )

    target_size = numeric(
        target_feature_record[
            feature_size_col
        ]
    )

    target_si_fn = safe_bool(
        target_feature_record[
            feature_si_fn_col
        ]
    )


    # ========================================================
    # PRINT VERIFIED TARGET
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VERIFIED TARGET ATTRIBUTES"
    )
    print("=" * 80)

    print(
        f"  filename  : "
        f"{target_filename}"
    )

    print(
        f"  file_id   : "
        f"{target_file_id}"
    )

    print(
        f"  extension : "
        f"{target_extension}"
    )

    print(
        f"  size      : "
        f"{target_size}"
    )

    print(
        f"  name_type : "
        f"{target_name_type}"
    )

    print(
        f"  SI<FN     : "
        f"{target_si_fn}"
    )


    # ========================================================
    # VALIDATE TARGET ATTRIBUTES
    # ========================================================

    target_incomplete = (

        target_extension == ""

        or

        target_name_type == ""

        or

        pd.isna(
            target_size
        )

        or

        target_si_fn is None
    )


    if target_incomplete:

        print()
        print(
            "[ERROR] Target attributes incomplete."
        )

        summary_rows.append({

            "scenario":
                scenario,

            "target_filename":
                target_filename,

            "target_file_id":
                target_file_id,

            "target_extension":
                target_extension,

            "target_size":
                target_size,

            "target_nametype":
                target_name_type,

            "target_si_fn":
                target_si_fn,

            "status":
                "TARGET_ATTRIBUTES_INCOMPLETE"
        })

        continue


    # ========================================================
    # BUILD RAW MFT METADATA PER FILE ID
    # ========================================================

    print()
    print(
        "Building raw MFT metadata..."
    )

    raw_meta = (

        raw_mft

        .groupby(
            "_file_id_normalized",
            as_index=False
        )

        .agg(

            filename=(
                raw_filename_col,
                lambda values:
                    unique_join(
                        values
                    )
            ),

            extension=(
                raw_extension_col,
                lambda values:
                    unique_join(
                        values,
                        normalize_extension
                    )
            ),

            name_type=(
                raw_name_type_col,
                lambda values:
                    unique_join(
                        values,
                        normalize_name_type
                    )
            )
        )
    )


    print(
        f"Raw MFT metadata records: "
        f"{len(raw_meta):,}"
    )


    # ========================================================
    # MERGE RAW METADATA WITH FEATURES
    # ========================================================

    print()
    print(
        "Merging raw MFT metadata + MFT features..."
    )

    candidates = features.merge(

        raw_meta,

        on="_file_id_normalized",

        how="left"
    )


    # ========================================================
    # REMOVE TARGET FROM BACKGROUND
    # ========================================================

    candidates = candidates[
        candidates[
            "_file_id_normalized"
        ]
        !=
        target_file_id
    ].copy()


    print(
        f"Background records: "
        f"{len(candidates):,}"
    )


    # ========================================================
    # NORMALIZE CANDIDATE ATTRIBUTES
    # ========================================================

    candidates[
        "_extension_norm"
    ] = (
        candidates[
            "extension"
        ]
        .apply(
            normalize_extension
        )
    )


    candidates[
        "_name_type_norm"
    ] = (
        candidates[
            "name_type"
        ]
        .apply(
            normalize_name_type
        )
    )


    candidates[
        "_size_numeric"
    ] = (
        candidates[
            feature_size_col
        ]
        .apply(
            numeric
        )
    )


    candidates[
        "_si_fn_bool"
    ] = (
        candidates[
            feature_si_fn_col
        ]
        .apply(
            safe_bool
        )
    )


    # ========================================================
    # SIMILARITY FEATURES
    # ========================================================

    # --------------------------------------------------------
    # Same extension
    # --------------------------------------------------------

    candidates[
        "same_extension"
    ] = (

        candidates[
            "_extension_norm"
        ]
        ==
        target_extension
    )


    # --------------------------------------------------------
    # Exact size
    # --------------------------------------------------------

    candidates[
        "exact_size"
    ] = (

        candidates[
            "_size_numeric"
        ]
        ==
        target_size
    )


    # --------------------------------------------------------
    # Relative size difference
    # --------------------------------------------------------

    if target_size > 0:

        relative_difference = (

            (

                candidates[
                    "_size_numeric"
                ]
                -
                target_size

            ).abs()

            /

            target_size
        )

    else:

        relative_difference = (

            candidates[
                "_size_numeric"
            ]
            -
            target_size

        ).abs()


    # --------------------------------------------------------
    # Size ±5%
    # --------------------------------------------------------

    candidates[
        "size_within_5pct"
    ] = (

        relative_difference
        <=
        0.05
    )


    # --------------------------------------------------------
    # Size ±10%
    # --------------------------------------------------------

    candidates[
        "size_within_10pct"
    ] = (

        relative_difference
        <=
        0.10
    )


    # --------------------------------------------------------
    # Same NameType
    # --------------------------------------------------------

    candidates[
        "same_name_type"
    ] = (

        candidates[
            "_name_type_norm"
        ]
        ==
        target_name_type
    )


    # --------------------------------------------------------
    # Same SI/FN
    # --------------------------------------------------------

    candidates[
        "same_si_fn"
    ] = (

        candidates[
            "_si_fn_bool"
        ]
        ==
        target_si_fn
    )


    # ========================================================
    # COMBINATION RULES
    # ========================================================

    # --------------------------------------------------------
    # Same extension + exact size
    # --------------------------------------------------------

    candidates[
        "extension_exact_size"
    ] = (

        candidates[
            "same_extension"
        ]

        &

        candidates[
            "exact_size"
        ]
    )


    # --------------------------------------------------------
    # Same extension + size ±5%
    # --------------------------------------------------------

    candidates[
        "extension_size_5pct"
    ] = (

        candidates[
            "same_extension"
        ]

        &

        candidates[
            "size_within_5pct"
        ]
    )


    # --------------------------------------------------------
    # Same extension + size ±10%
    # --------------------------------------------------------

    candidates[
        "extension_size_10pct"
    ] = (

        candidates[
            "same_extension"
        ]

        &

        candidates[
            "size_within_10pct"
        ]
    )


    # --------------------------------------------------------
    # Extension + size ±5% + SI/FN
    # --------------------------------------------------------

    candidates[
        "extension_size5_si_fn"
    ] = (

        candidates[
            "same_extension"
        ]

        &

        candidates[
            "size_within_5pct"
        ]

        &

        candidates[
            "same_si_fn"
        ]
    )


    # --------------------------------------------------------
    # Extension + size ±5% + NameType
    # --------------------------------------------------------

    candidates[
        "extension_size5_name_type"
    ] = (

        candidates[
            "same_extension"
        ]

        &

        candidates[
            "size_within_5pct"
        ]

        &

        candidates[
            "same_name_type"
        ]
    )


    # ========================================================
    # FORENSIC SIMILARITY SCORE
    # ========================================================
    #
    # PENTING:
    #
    # Score ini HANYA digunakan untuk meranking kandidat.
    #
    # Score ini BUKAN:
    #   - label
    #   - ground truth
    #   - prediction
    #   - fitur RF
    #
    # ========================================================

    candidates[
        "forensic_similarity_score"
    ] = (

        candidates[
            "same_extension"
        ].astype(int)

        +

        candidates[
            "size_within_5pct"
        ].astype(int)

        +

        candidates[
            "same_si_fn"
        ].astype(int)

        +

        candidates[
            "same_name_type"
        ].astype(int)
    )


    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    output_cols = [

        feature_file_id_col,

        "filename",

        "extension",

        feature_size_col,

        "name_type",

        feature_si_fn_col,

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

        "forensic_similarity_score"
    ]


    output_cols = [

        column

        for column in output_cols

        if column in candidates.columns
    ]


    # ========================================================
    # SORT ALL BACKGROUND
    # ========================================================

    all_background = (

        candidates[
            output_cols
        ]

        .sort_values(

            by=[

                "forensic_similarity_score",

                "extension_size_5pct",

                "exact_size"
            ],

            ascending=False
        )
    )


    # ========================================================
    # SAVE ALL BACKGROUND
    # ========================================================

    all_background_file = (

        OUTPUT_DIR
        /
        f"Scenario_{scenario_text}"
        "_all_background_candidates.csv"
    )

    all_background.to_csv(

        all_background_file,

        index=False
    )


    # ========================================================
    # EXTENSION CANDIDATES
    # ========================================================

    extension_candidates = (

        candidates[
            candidates[
                "same_extension"
            ]
        ]

        .copy()
    )


    extension_candidates = (

        extension_candidates

        .sort_values(

            by=[

                "forensic_similarity_score",

                "size_within_5pct",

                "exact_size"
            ],

            ascending=False
        )
    )


    extension_candidates_file = (

        OUTPUT_DIR
        /
        f"Scenario_{scenario_text}"
        "_extension_candidates.csv"
    )


    extension_candidates[
        output_cols
    ].to_csv(

        extension_candidates_file,

        index=False
    )


    # ========================================================
    # STRICT CANDIDATES
    # ========================================================

    strict_candidates = (

        candidates[

            candidates[
                "extension_size5_si_fn"
            ]

            |

            candidates[
                "extension_size5_name_type"
            ]
        ]

        .copy()
    )


    strict_candidates = (

        strict_candidates

        .sort_values(

            by=[

                "forensic_similarity_score",

                "extension_size5_si_fn",

                "extension_size5_name_type"
            ],

            ascending=False
        )
    )


    strict_candidates_file = (

        OUTPUT_DIR
        /
        f"Scenario_{scenario_text}"
        "_strict_candidates.csv"
    )


    strict_candidates[
        output_cols
    ].to_csv(

        strict_candidates_file,

        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    row = {

        "scenario":
            scenario,

        "target_filename":
            target_filename,

        "target_file_id":
            target_file_id,

        "target_extension":
            target_extension,

        "target_size":
            target_size,

        "target_nametype":
            target_name_type,

        "target_si_fn":
            target_si_fn,

        "background_records":
            len(candidates),

        "same_extension":
            int(
                candidates[
                    "same_extension"
                ].sum()
            ),

        "exact_size":
            int(
                candidates[
                    "exact_size"
                ].sum()
            ),

        "size_5pct":
            int(
                candidates[
                    "size_within_5pct"
                ].sum()
            ),

        "size_10pct":
            int(
                candidates[
                    "size_within_10pct"
                ].sum()
            ),

        "same_name_type":
            int(
                candidates[
                    "same_name_type"
                ].sum()
            ),

        "same_si_fn":
            int(
                candidates[
                    "same_si_fn"
                ].sum()
            ),

        "extension_exact_size":
            int(
                candidates[
                    "extension_exact_size"
                ].sum()
            ),

        "extension_size5":
            int(
                candidates[
                    "extension_size_5pct"
                ].sum()
            ),

        "extension_size10":
            int(
                candidates[
                    "extension_size_10pct"
                ].sum()
            ),

        "extension_size5_si_fn":
            int(
                candidates[
                    "extension_size5_si_fn"
                ].sum()
            ),

        "extension_size5_name_type":
            int(
                candidates[
                    "extension_size5_name_type"
                ].sum()
            ),

        "strict_candidates":
            len(
                strict_candidates
            ),

        "max_similarity_score":
            int(
                candidates[
                    "forensic_similarity_score"
                ].max()
            )
            if len(candidates) > 0
            else 0,

        "status":
            "OK"
    }


    summary_rows.append(
        row
    )


    # ========================================================
    # PRINT SCENARIO RESULT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CANDIDATE COUNTS"
    )
    print("=" * 80)

    print(
        f"  Background               : "
        f"{row['background_records']:,}"
    )

    print(
        f"  Same extension           : "
        f"{row['same_extension']:,}"
    )

    print(
        f"  Exact size               : "
        f"{row['exact_size']:,}"
    )

    print(
        f"  Size ±5%                 : "
        f"{row['size_5pct']:,}"
    )

    print(
        f"  Size ±10%                : "
        f"{row['size_10pct']:,}"
    )

    print(
        f"  Same NameType            : "
        f"{row['same_name_type']:,}"
    )

    print(
        f"  Same SI<FN               : "
        f"{row['same_si_fn']:,}"
    )

    print(
        f"  Extension + exact size   : "
        f"{row['extension_exact_size']:,}"
    )

    print(
        f"  Extension + size ±5%     : "
        f"{row['extension_size5']:,}"
    )

    print(
        f"  Extension + size ±10%    : "
        f"{row['extension_size10']:,}"
    )

    print(
        f"  Extension + size5 + SI   : "
        f"{row['extension_size5_si_fn']:,}"
    )

    print(
        f"  Extension + size5 + NT   : "
        f"{row['extension_size5_name_type']:,}"
    )

    print(
        f"  Strict candidates        : "
        f"{row['strict_candidates']:,}"
    )

    print(
        f"  Max similarity score     : "
        f"{row['max_similarity_score']}"
    )

    print()
    print(
        "[OK] Scenario selesai."
    )


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUTPUT_DIR
    /
    "negative_candidate_summary.csv"
)


summary_df.to_csv(

    summary_file,

    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("=" * 100)
print("NEGATIVE CANDIDATE ANALYSIS SUMMARY")
print("=" * 100)
print()

if len(summary_df) > 0:

    print(
        summary_df.to_string(
            index=False
        )
    )

print()
print("=" * 100)
print(
    "Output directory:"
)
print(
    OUTPUT_DIR
)
print("=" * 100)
print()


# ============================================================
# STATUS SUMMARY
# ============================================================

if len(summary_df) > 0:

    print(
        "STATUS SUMMARY"
    )

    print()

    print(
        summary_df[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    ok_count = int(
        (
            summary_df[
                "status"
            ]
            ==
            "OK"
        )
        .sum()
    )

    print(
        f"Scenario OK: "
        f"{ok_count}/{len(summary_df)}"
    )

    print()


# ============================================================
# END
# ============================================================

print(
    "Script 32 selesai."
)