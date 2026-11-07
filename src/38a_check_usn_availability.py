from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\atnin\Downloads\TimestampResearch")

INPUT = (
    ROOT
    / "output"
    / "ml_dataset_all"
    / "all_scenarios_ml_dataset.csv"
)

df = pd.read_csv(INPUT)

print("=" * 80)
print("USN AVAILABILITY BY SCENARIO")
print("=" * 80)

for scenario in sorted(df["scenario"].unique()):

    sub = df[df["scenario"] == scenario]

    background = sub[sub["label"] == 0]

    target = sub[sub["label"] == 1].iloc[0]

    usn_rich = background[
        (background["has_usn"] == 1)
        & (background["usn_event_count"] > 0)
    ]

    print(
        f"Scenario {scenario:>2} | "
        f"background={len(background):>3} | "
        f"USN-rich={len(usn_rich):>3} | "
        f"target_events={target['usn_event_count']:>5} | "
        f"target_close={target['usn_close_count']:>3} | "
        f"target_extend={target['usn_data_extend_count']:>3} | "
        f"target_create={target['usn_file_create_count']:>3}"
    )