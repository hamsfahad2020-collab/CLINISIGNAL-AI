"""Generate a clearly synthetic ICU time-series dataset for software prototyping.

This dataset is NOT clinical evidence, is NOT derived from patients, and must not
be used for diagnosis, treatment, or claims of clinical model performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "heart_rate_bpm", "map_mmhg", "resp_rate_min", "temperature_c",
    "spo2_pct", "wbc_10e9_l", "lactate_mmol_l", "creatinine_mg_dl",
    "platelets_10e9_l",
]


def generate_dataset(n_patients: int = 300, hours: int = 36, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for patient_num in range(1, n_patients + 1):
        patient_id = f"SYN-{patient_num:04d}"
        has_event = rng.random() < 0.30
        event_hour = int(rng.integers(18, hours - 2)) if has_event else None

        # Patient-specific baseline variation; values are illustrative only.
        base = {
            "hr": rng.normal(82, 9), "map": rng.normal(88, 8),
            "rr": rng.normal(17, 2.5), "temp": rng.normal(36.9, 0.25),
            "spo2": rng.normal(97, 1.2), "wbc": rng.normal(8.0, 1.8),
            "lactate": max(0.5, rng.normal(1.4, 0.35)),
            "creatinine": max(0.4, rng.normal(0.95, 0.2)),
            "platelets": max(100, rng.normal(235, 35)),
        }
        susceptibility = rng.normal(0, 0.12)

        for hour in range(hours):
            # Exclude observations at/after event onset: only pre-event prediction rows.
            if event_hour is not None and hour >= event_hour:
                break

            hours_to_event = (event_hour - hour) if event_hour is not None else None
            deterioration = 0.0
            if hours_to_event is not None and hours_to_event <= 12:
                deterioration = (13 - hours_to_event) / 12
            deterioration = max(0.0, deterioration + susceptibility * 0.15)

            hr = base["hr"] + 30 * deterioration + rng.normal(0, 4)
            map_v = base["map"] - 24 * deterioration + rng.normal(0, 4)
            rr = base["rr"] + 10 * deterioration + rng.normal(0, 1.8)
            temp = base["temp"] + 1.0 * deterioration + rng.normal(0, 0.18)
            spo2 = base["spo2"] - 4.5 * deterioration + rng.normal(0, 1.0)
            wbc = base["wbc"] + 5.5 * deterioration + rng.normal(0, 1.0)
            lactate = base["lactate"] + 3.0 * deterioration + rng.normal(0, 0.35)
            creatinine = base["creatinine"] + 0.65 * deterioration + rng.normal(0, 0.12)
            platelets = base["platelets"] - 85 * deterioration + rng.normal(0, 12)

            # Synthetic target: onset is within the next six hours.
            target = int(event_hour is not None and 0 < hours_to_event <= 6)
            values = [hr, map_v, rr, temp, spo2, wbc, lactate, creatinine, platelets]
            values = [float(v) for v in values]
            # Add modest, realistic-looking missingness for pipeline testing.
            for i in range(len(values)):
                if rng.random() < 0.025:
                    values[i] = np.nan

            row = {"patient_id": patient_id, "hour_from_admission": hour}
            row.update(dict(zip(FEATURES, values)))
            row["risk_event_within_6h"] = target
            row["synthetic_event_hour"] = event_hour if event_hour is not None else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=300)
    parser.add_argument("--hours", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_icu.csv"))
    args = parser.parse_args()

    df = generate_dataset(args.patients, args.hours, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows for {df['patient_id'].nunique():,} synthetic patients to {args.output}")
    print("Target distribution:")
    print(df["risk_event_within_6h"].value_counts(dropna=False).to_string())
    print("Reminder: synthetic prototype data only; not for clinical use.")


if __name__ == "__main__":
    main()
