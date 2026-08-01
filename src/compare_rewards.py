"""
compare_rewards.py
------------------
Standalone comparison script for the two RL training experiments produced
by train.py ('standard' reward vs 'custom' reward).

Purpose
-------
Before integrating the RL agent into evaluate.py and the GUI, this script
evaluates both actor networks on the held-out evaluation schema (the 20%
of the simulation window reserved by train.py) and prints a side-by-side
KPI table.  The better-performing model is then the one to carry forward.

This comparison is intentionally kept separate from the main evaluate.py,
which benchmarks agents of different *types* (RBC, regressor, RL).  Here
we are comparing two variants of the *same agent type* to select the best
reward formulation — a different and earlier decision.

What it produces
----------------
results/
└── reward_comparison/
    ├── district_kpis.png   ← bar chart of all district-level KPIs
    └── kpi_summary.csv     ← raw KPI values for both models

Usage
-----
    python compare_rewards.py --exp results/experiments/PPOLag_myrun

The --exp argument must point to the experiment folder created by train.py.
It expects the following structure inside that folder:

    {exp}/
    ├── seed{n}/
    │   ├── actor_net_standard.pt
    │   └── actor_net_custom.pt
    └── schemas/
        └── eval_schema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Add the project root to sys.path so that local packages (CityLearn,
# omnisafe) and src/ modules are importable without installation.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.citylearn import CityLearnEnv

# evaluate_agent is the shared episode-runner used by all agents in this
# project; reusing it keeps the evaluation logic consistent.
from evaluate import evaluate_agent
from rl_agent import OmnisafeActorAgent
from utils import get_kpis, plot_district_kpis


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standard vs custom reward RL models on the eval schema."
    )
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        help="Path to the experiment folder produced by train.py "
             "(e.g. 'results/experiments/PPOLag_run1').",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Seed subfolder to load actor networks from (default: 1).",
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help="Output directory for plots and CSV. "
             "Defaults to 'results/reward_comparison/' if not specified.",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_eval_schema(exp_dir: str) -> dict:
    """
    Load the evaluation schema saved by train.py.

    The eval schema covers the held-out (1 - frac) portion of the
    simulation window and was never seen by the agent during training.

    Parameters
    ----------
    exp_dir : str
        Root experiment directory (the one passed via --exp).

    Returns
    -------
    dict
        CityLearn schema dict ready to be passed to CityLearnEnv.
    """
    schema_path = os.path.join(exp_dir, 'schemas', 'eval_schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Eval schema not found at '{schema_path}'.\n"
            "Make sure --exp points to a folder produced by train.py."
        )
    with open(schema_path) as f:
        return json.load(f)


def load_actor(exp_dir: str, seed: int, reward_tag: str) -> str:
    """
    Resolve and validate the path to a saved TorchScript actor file.

    Parameters
    ----------
    exp_dir : str
        Root experiment directory.
    seed : int
        Seed used during training (determines the subfolder name).
    reward_tag : str
        'standard' or 'custom'.

    Returns
    -------
    str
        Absolute path to the .pt file.
    """
    path = os.path.join(exp_dir, f'seed{seed}', f'actor_net_{reward_tag}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Actor file not found: '{path}'.\n"
            f"Run train.py first to generate 'actor_net_{reward_tag}.pt'."
        )
    return path


def print_kpi_table(kpi_df: pd.DataFrame, label: str) -> None:
    """
    Print a formatted KPI summary for a single agent to stdout.

    Parameters
    ----------
    kpi_df : pd.DataFrame
        District-level KPI table returned by get_kpis().
    label : str
        Agent label shown in the header (e.g. 'standard' or 'custom').
    """
    district = kpi_df[kpi_df['level'] == 'district'][['kpi', 'value']]
    print(f"\n  [{label}]")
    for _, row in district.iterrows():
        print(f"    {row['kpi']:<30} {row['value']:.4f}")


def save_kpi_csv(envs: dict[str, CityLearnEnv], out_dir: str) -> None:
    """
    Merge district-level KPIs from both environments and save to CSV.

    Parameters
    ----------
    envs : dict[str, CityLearnEnv]
        Mapping of reward_tag → evaluated environment.
    out_dir : str
        Directory where 'kpi_summary.csv' is written.
    """
    rows = []
    for label, env in envs.items():
        kpis = get_kpis(env)
        district = kpis[kpis['level'] == 'district'].copy()
        district['model'] = label
        rows.append(district)

    summary = pd.concat(rows, ignore_index=True)
    csv_path = os.path.join(out_dir, 'kpi_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"\nKPI summary saved → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = args.out if args.out is not None else os.path.join(
        os.path.dirname(__file__), '..', 'results', 'reward_comparison'
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Load eval schema ──────────────────────────────────────────────────────
    print(f"Loading eval schema from: {args.exp}")
    eval_schema = load_eval_schema(args.exp)

    n_steps = (
        eval_schema['simulation_end_time_step']
        - eval_schema['simulation_start_time_step']
    )
    print(f"Eval window: {n_steps} time steps")

    # ── Evaluate both models ──────────────────────────────────────────────────
    # Each model gets its own fresh CityLearnEnv so episode states are
    # completely independent and do not influence each other.
    reward_tags = ['standard', 'custom']
    envs: dict[str, CityLearnEnv] = {}

    for tag in reward_tags:
        print(f"\n{'─' * 50}")
        print(f"Evaluating: actor_net_{tag}.pt")

        actor_path = load_actor(args.exp, args.seed, tag)

        # Fresh environment for this model
        env = CityLearnEnv(schema=eval_schema, central_agent=True)

        # OmnisafeActorAgent handles observation normalisation internally,
        # so the env passed here stays unwrapped (same as for RBC / regressor).
        agent = OmnisafeActorAgent(fname=actor_path, env=env)

        # Run a full episode and collect building-level data
        _ = evaluate_agent(agent, env)

        # Store the stepped environment for KPI extraction via env.evaluate()
        envs[tag] = env

    # ── KPI comparison ────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("District-level KPI comparison")
    print(f"{'=' * 50}")

    for tag, env in envs.items():
        kpis = get_kpis(env)
        print_kpi_table(kpis, label=tag)

    # ── Identify the better model ─────────────────────────────────────────────
    # Primary criterion: Discomfort (lower is better — fewer comfort violations).
    # Tiebreaker: Emissions (lower is better).
    # Both are normalised KPIs where 1.0 = no-control baseline.
    scores: dict[str, float] = {}
    for tag, env in envs.items():
        kpis = get_kpis(env)
        district = kpis[kpis['level'] == 'district']
        discomfort = district[district['kpi'] == 'Discomfort']['value'].values
        emissions  = district[district['kpi'] == 'Emissions']['value'].values
        # Weighted sum: discomfort weighted more heavily as the CMDP cost signal
        scores[tag] = (
            0.7 * float(discomfort[0]) if len(discomfort) else 0.0
          + 0.3 * float(emissions[0])  if len(emissions)  else 0.0
        )

    best = min(scores, key=lambda t: scores[t])
    print(f"\n{'=' * 50}")
    print(f"  Recommended model: '{best}' (score: {scores[best]:.4f})")
    print(f"  (Scoring: 0.7 × Discomfort + 0.3 × Emissions, lower is better)")
    print(f"{'=' * 50}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Bar chart comparing all district KPIs side by side
    plot_district_kpis(envs, base_path=out_dir)
    print(f"KPI bar chart saved → {out_dir}/district_kpis.png")

    # Raw CSV for further inspection
    save_kpi_csv(envs, out_dir)


if __name__ == '__main__':
    main()