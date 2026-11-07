
# ============================================================
# SCRIPT 42: LOCAL SHAP FIGURE FOR S03, S10, AND S12
# Standalone script -- does not require variables from Script 41
# Reads the latest CSV outputs from Script 40.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\atnin\Downloads\TimestampResearch"
)

SHAP_DIR = (
    BASE_DIR
    / "output"
    / "shap_hard_negative_loso"
)

FIG_DIR = SHAP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SHAP_FILE = (
    SHAP_DIR / "shap_global_summary.csv"
)

LOCAL_SHAP_FILE = (
    SHAP_DIR / "shap_local_all_test_samples.csv"
)

OUTPUT_FILE = (
    FIG_DIR / "Figure_4_Local_SHAP_S03_S10_S12.png"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_scenario(value):
    """Convert values such as 3, S03, or Scenario_03 to integer 3."""

    digits = "".join(
        char for char in str(value)
        if char.isdigit()
    )

    if not digits:
        raise ValueError(
            f"Cannot interpret scenario value: {value}"
        )

    return int(digits)


def get_unique_numeric_value(data, column, scenario_id):
    """Require one consistent metadata value per target."""

    values = pd.to_numeric(
        data[column],
        errors="coerce"
    ).dropna().unique()

    if len(values) != 1:
        raise ValueError(
            f"Inconsistent {column} values for "
            f"S{scenario_id:02d}: {values.tolist()}"
        )

    return values[0]


# ============================================================
# 3. CHECK AND LOAD INPUT FILES
# ============================================================

print("=" * 65)
print("SCRIPT 42 - LOCAL SHAP FIGURE")
print("=" * 65)

for input_file in [
    GLOBAL_SHAP_FILE,
    LOCAL_SHAP_FILE
]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"File not found:\n{input_file}\n\n"
            "Check the project path and make sure "
            "Script 40 has generated its CSV files."
        )

global_df = pd.read_csv(GLOBAL_SHAP_FILE)
local_df = pd.read_csv(LOCAL_SHAP_FILE)

print("Global SHAP:", global_df.shape)
print("Local SHAP :", local_df.shape)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_global = {
    "feature",
    "mean_abs_shap"
}

required_local = {
    "scenario",
    "y_true",
    "y_pred",
    "y_probability",
    "negative_type",
    "feature",
    "shap_value"
}

missing_global = (
    required_global - set(global_df.columns)
)

missing_local = (
    required_local - set(local_df.columns)
)

if missing_global:
    raise KeyError(
        f"Missing global SHAP columns: {missing_global}"
    )

if missing_local:
    raise KeyError(
        f"Missing local SHAP columns: {missing_local}"
    )


# ============================================================
# 5. PREPARE DATA
# ============================================================

global_df["mean_abs_shap"] = pd.to_numeric(
    global_df["mean_abs_shap"],
    errors="coerce"
)

local_df["_scenario_id"] = (
    local_df["scenario"].apply(normalize_scenario)
)

for column in [
    "y_true",
    "y_pred",
    "y_probability",
    "shap_value"
]:
    local_df[column] = pd.to_numeric(
        local_df[column],
        errors="coerce"
    )

global_df = global_df.dropna(
    subset=["feature", "mean_abs_shap"]
)

if global_df.empty:
    raise ValueError(
        "No valid features in global SHAP summary."
    )


# ============================================================
# 6. SELECT TOP 10 GLOBAL FEATURES
# ============================================================

top_features = (
    global_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .drop_duplicates(
        subset=["feature"]
    )
    .head(10)["feature"]
    .tolist()
)

if len(top_features) != 10:
    raise ValueError(
        f"Expected 10 top features, got {len(top_features)}."
    )

print("\nTop 10 features selected:")
for feature in top_features:
    print(" -", feature)


# ============================================================
# 7. EXTRACT TARGET SHAP VALUES FOR S03, S10, AND S12
# ============================================================

scenario_ids = [3, 10, 12]

shap_by_scenario = {}
metadata = {}

for scenario_id in scenario_ids:

    # First use the explicit target records.
    target_rows = local_df[
        (local_df["_scenario_id"] == scenario_id)
        & (local_df["y_true"] == 1)
        & (
            local_df["negative_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "target"
        )
    ].copy()

    # Fallback in case the target label has a different string.
    if target_rows.empty:
        target_rows = local_df[
            (local_df["_scenario_id"] == scenario_id)
            & (local_df["y_true"] == 1)
        ].copy()

    if target_rows.empty:
        raise ValueError(
            f"No positive target found for "
            f"S{scenario_id:02d}."
        )

    # Validate target metadata.
    y_true = int(
        get_unique_numeric_value(
            target_rows, "y_true", scenario_id
        )
    )

    y_pred = int(
        get_unique_numeric_value(
            target_rows, "y_pred", scenario_id
        )
    )

    probability = float(
        get_unique_numeric_value(
            target_rows, "y_probability", scenario_id
        )
    )

    if y_pred == y_true:
        status = "Correctly classified"
    elif y_true == 1:
        status = "False negative"
    else:
        status = "False positive"

    metadata[scenario_id] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "probability": probability,
        "status": status
    }

    # Keep the same global features for each target.
    selected = target_rows[
        target_rows["feature"].isin(top_features)
    ].copy()

    selected = selected.dropna(
        subset=["shap_value"]
    )

    # Each selected feature must have exactly one SHAP value.
    duplicated = selected["feature"].duplicated(
        keep=False
    )

    if duplicated.any():
        duplicated_features = (
            selected.loc[duplicated, "feature"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate feature rows in S{scenario_id:02d}: "
            f"{duplicated_features}"
        )

    shap_values = (
        selected
        .set_index("feature")["shap_value"]
        .reindex(top_features)
    )

    if shap_values.isna().any():
        missing_features = (
            shap_values.index[
                shap_values.isna()
            ].tolist()
        )

        raise ValueError(
            f"Missing SHAP values for S{scenario_id:02d}: "
            f"{missing_features}"
        )

    shap_by_scenario[scenario_id] = (
        shap_values.astype(float)
    )


# ============================================================
# 8. USE A COMMON FEATURE ORDER
# ============================================================

combined_shap = pd.concat(
    [
        shap_by_scenario[sid].rename(
            f"S{sid:02d}"
        )
        for sid in scenario_ids
    ],
    axis=1
)

# Rank features by the largest absolute local contribution
# across S03, S10, and S12.
max_abs = combined_shap.abs().max(axis=1)

ordered_features = (
    max_abs
    .sort_values(ascending=True)
    .index.tolist()
)


# ============================================================
# 9. CREATE THREE-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 6.5),
    sharey=True,
    sharex=True
)

y_positions = np.arange(
    len(ordered_features)
)

for panel_index, (ax, scenario_id) in enumerate(
    zip(axes, scenario_ids)
):

    feature_values = (
        shap_by_scenario[scenario_id]
        .reindex(ordered_features)
        .to_numpy()
    )

    info = metadata[scenario_id]

    ax.barh(
        y_positions,
        feature_values,
        height=0.72
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_title(
        f"S{scenario_id:02d} - {info['status']}\n"
        f"P(target)={info['probability']:.4f}; "
        f"prediction={info['y_pred']}",
        fontsize=9
    )

    ax.set_xlabel("SHAP value")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    if panel_index == 0:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            ordered_features,
            fontsize=8
        )
        ax.set_ylabel("Feature")
    else:
        ax.tick_params(
            axis="y",
            labelleft=False
        )

fig.suptitle(
    "Local SHAP Comparison of Detected and Missed Targets",
    fontsize=12
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\nFigure created successfully:")
print(OUTPUT_FILE)

print("\nTarget summary:")

for scenario_id in scenario_ids:
    info = metadata[scenario_id]

    print(
        f"S{scenario_id:02d}: "
        f"{info['status']}, "
        f"P(target)={info['probability']:.4f}, "
        f"prediction={info['y_pred']}"
    )

print("\nScript 42 completed.")
