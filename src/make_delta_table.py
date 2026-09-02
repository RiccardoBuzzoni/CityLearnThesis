"""
Builds a summary CSV table of absolute mean cooling_setpoint_delta values.

For each simulation CSV:
    1. Convert every cooling_setpoint_delta value to its absolute value.
    2. Compute the mean absolute delta for each of the three perturbation phases.
    3. Compute the overall mean absolute delta.

The final table groups runs by policy:
    - RBC: one row (single run)
    - Regressor: one row containing the mean across its three seeds
    - RL: one row containing the mean across its three seeds

A final "Standard deviation" column reports the sample standard deviation
(ddof=1) of the overall mean absolute deltas across the seeds of each
multi-seed policy. RBC has only one run, so its standard deviation is left
blank because it cannot estimate seed-to-seed variability.

Usage
-----
    python make_delta_table_absolute.py <folder_with_csvs> [-o output.csv]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DELTA_COL = "cooling_setpoint_delta"
SP_COL = "indoor_dry_bulb_temperature_cooling_set_point"

POLICY_ORDER: dict[str, int] = {
    "rbc": 0,
    "neural": 1,
    "rl": 2,
}

POLICY_DISPLAY: dict[str, str] = {
    "rbc": "RBC",
    "neural": "Regressor",
    "rl": "RL",
}

RESULT_COLUMNS = [
    "Mean delta - phase 1",
    "Mean delta - phase 2",
    "Mean delta - phase 3",
    "Mean delta - overall",
]


def _sp_amount_series(pert_col) -> list[float]:
    """
    Per-step cooling-setpoint perturbation amount declared in the
    'perturbations' meta-column (0.0 where the setpoint was not perturbed).
    """
    out = []

    for raw in pert_col:
        amt = 0.0
        s = str(raw).strip() if raw is not None else ""

        if s and s not in ("[]", "nan", "None", "null"):
            try:
                for p in json.loads(s):
                    if p.get("feat") == SP_COL:
                        amt = float(p.get("amount", 0.0))
            except Exception:
                pass

        out.append(amt)

    return out


def _phase_segments(amounts: list[float]) -> list[tuple[int, int, float]]:
    """
    Return contiguous runs of a constant cooling-setpoint perturbation amount
    as [(start_idx, end_idx_exclusive, amount), ...].
    """
    segments = []

    if not amounts:
        return segments

    start, cur = 0, amounts[0]

    for i in range(1, len(amounts)):
        if amounts[i] != cur:
            segments.append((start, i, cur))
            start, cur = i, amounts[i]

    segments.append((start, len(amounts), cur))

    return segments


def compute_row(path: Path) -> dict:
    """Compute absolute-delta means for one simulation CSV."""
    df = pd.read_csv(path)

    if DELTA_COL not in df.columns:
        raise ValueError(f"{path.name}: missing column '{DELTA_COL}'")
    if "perturbations" not in df.columns:
        raise ValueError(f"{path.name}: missing column 'perturbations'")
    if "policy" not in df.columns:
        raise ValueError(f"{path.name}: missing column 'policy'")

    # Convert every per-timestep delta to its absolute value BEFORE averaging.
    abs_delta = df[DELTA_COL].abs()

    amounts = _sp_amount_series(df["perturbations"].tolist())
    segments = _phase_segments(amounts)

    phase_means = []

    for i in range(3):
        if i < len(segments):
            start, end, _ = segments[i]
            phase_means.append(abs_delta.iloc[start:end].mean())
        else:
            phase_means.append(float("nan"))

    policy_key = str(df["policy"].iloc[0]).strip().lower()

    return {
        "_policy_key": policy_key,
        "Mean delta - phase 1": phase_means[0],
        "Mean delta - phase 2": phase_means[1],
        "Mean delta - phase 3": phase_means[2],
        "Mean delta - overall": abs_delta.mean(),
    }


def aggregate_by_policy(rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate all seed-level rows into one final row per policy.

    Phase and overall columns are averaged across the available seeds.
    Standard deviation is computed across the seed-level overall means.
    """
    seed_table = pd.DataFrame(rows)

    final_rows = []

    for policy_key, group in seed_table.groupby("_policy_key", sort=False):
        policy_key = str(policy_key)

        row: dict[str, str | float] = {
            "Run": POLICY_DISPLAY.get(policy_key, policy_key),
        }

        # Mean result across all seeds/runs belonging to this policy.
        for col in RESULT_COLUMNS:
            row[col] = group[col].mean()

        # Seed-to-seed variability of the OVERALL mean absolute delta.
        # With only one RBC run, sample std is undefined and is left blank.
        row["Standard deviation"] = (
            group["Mean delta - overall"].std(ddof=1)
            if len(group) > 1
            else float("nan")
        )

        row["_policy_key"] = policy_key
        final_rows.append(row)

    table = pd.DataFrame(final_rows)
    table["_order"] = table["_policy_key"].map(POLICY_ORDER).fillna(99)
    table = (
        table.sort_values("_order")
        .drop(columns=["_policy_key", "_order"])
        .reset_index(drop=True)
    )

    return table


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("folder", help="Folder containing the run CSVs")
    ap.add_argument(
        "-o",
        "--output",
        default="delta_summary_table.csv",
        help=(
            "Output CSV path (default: delta_summary_table.csv "
            "inside the input folder)"
        ),
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.csv"))

    if not files:
        sys.exit(f"No CSV files found in {folder}")

    rows = []

    for f in files:
        try:
            rows.append(compute_row(f))
        except Exception as e:
            print(f"  [skipped] {f.name}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No valid run CSVs could be processed.")

    table = aggregate_by_policy(rows)

    out_path = Path(args.output)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = folder / out_path

    table.to_csv(out_path, index=False, float_format="%.4f")

    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
