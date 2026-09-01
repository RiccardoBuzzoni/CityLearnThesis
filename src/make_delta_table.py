"""
Builds a summary CSV table of mean cooling_setpoint_delta per policy/seed,
split into the three schedule phases (standard / high / low) plus an
overall-mean column, for the thesis presentation.
 
Usage
-----
    python make_delta_table.py <folder_with_csvs> [-o output.csv]
 
Expects one CSV per run (as produced by gui.py), each containing at least:
    - "policy"              (constant per file: "rbc" / "neural" / "RL")
    - "perturbations"       (per-step JSON list, used to find phase
                              boundaries — see _phase_segments)
    - "cooling_setpoint_delta"   (the per-step delta already saved by gui.py)
 
The three phases are found dynamically from the "perturbations" column of
EACH file (not assumed to be fixed-length thirds), so the table stays
correct even if a run's episode length or phase split differs slightly.
"""

import argparse
import json 
import re 
import sys 
from pathlib import Path 

import pandas as pd

DELTA_COL   = "cooling_setpoint_delta"
SP_COL      = "indoor_dry_bulb_temperature_cooling_set_point"

POLICY_ORDER    = {"rbc": 0, "neural": 1, "rl":2}
POLICY_DISPLAY  = {"rbc": "RBC", "neural": "Regressor", "rl": "RL"}

def _sp_amount_series(pert_col) -> list[float]:
    """
    Per-step cooling-setpoint perturbation amount declared in the
    'perturbations' meta-column (0.0 where the setpoint wasn't perturbed),
    same logic gui.py uses to draw the phase markers on its own plots.
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
    [(start_idx, end_idx_exclusive, amount), ...] — contiguous runs of a
    constant cooling-setpoint perturbation amount.
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

def _row_label(policy: str, path: Path) -> str:
    policy_key = policy.strip().lower()
    display = POLICY_DISPLAY.get(policy_key, policy)
    m = re.search(r"seed[_-]?(\d+)", path.stem, re.IGNORECASE)
    if policy_key == 'rbc' or not m:
        return display

    return f"{display} (seed {m.group(1)})"

def _sort_key(row: dict, path: Path):
    policy_key = row["_policy_key"]
    m = re.search(r"seed[_-]?(\d+)", path.stem, re.IGNORECASE)
    seed = int(m.group(1)) if m else -1

    return (POLICY_ORDER.get(policy_key, 99), seed)

def compute_row(path: Path) -> dict:
    df = pd.read_csv(path)
 
    if DELTA_COL not in df.columns:
        raise ValueError(f"{path.name}: missing column '{DELTA_COL}'")
    if "perturbations" not in df.columns:
        raise ValueError(f"{path.name}: missing column 'perturbations'")
    if "policy" not in df.columns:
        raise ValueError(f"{path.name}: missing column 'policy'")
 
    amounts  = _sp_amount_series(df["perturbations"].tolist())
    segments = _phase_segments(amounts)
 
    thirds = []
    for i in range(3):
        if i < len(segments):
            start, end, _ = segments[i]
            thirds.append(df[DELTA_COL].iloc[start:end].mean())
        else:
            thirds.append(float("nan"))
 
    policy = str(df["policy"].iloc[0])
    return {
        "_policy_key": policy.strip().lower(),
        "Run": _row_label(policy, path),
        "Mean delta - phase 1": thirds[0],
        "Mean delta - phase 2": thirds[1],
        "Mean delta - phase 3": thirds[2],
        "Mean delta - overall": df[DELTA_COL].mean(),
    }
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="Folder containing the run CSVs")
    ap.add_argument("-o", "--output", default="delta_summary_table.csv",
                     help="Output CSV path (default: delta_summary_table.csv "
                          "inside the input folder)")
    args = ap.parse_args()
 
    folder = Path(args.folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {folder}")
 
    rows = []
    for f in files:
        try:
            rows.append((compute_row(f), f))
        except Exception as e:
            print(f"  [skipped] {f.name}: {e}", file=sys.stderr)
 
    if not rows:
        sys.exit("No valid run CSVs could be processed.")
 
    rows.sort(key=lambda rf: _sort_key(rf[0], rf[1]))
 
    table = pd.DataFrame([r for r, _ in rows]).drop(columns=["_policy_key"])
 
    out_path = Path(args.output)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = folder / out_path
    table.to_csv(out_path, index=False, float_format="%.4f")
 
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}")
 
 
if __name__ == "__main__":
    main()