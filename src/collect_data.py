"""
collect_data.py
───────────────
Interactive data-collection script for the CityLearn thesis project.

Usage
-----
    python collect_data.py [--data DATASET] [--custom]

The script presents a terminal menu to choose which controller to simulate:

  1. RBC   → always available; collects an AdvancedRBC episode and writes
             data/datasets/advanced_rbc_dataset.csv
             (also the training data required by linear_regressor.py)

  2. Linear Regressor → available only if the model artefacts are present in
             ../results/linear_regressor/  (feature_mean.npy, feature_std.npy,
             and one model_<target>.pth per target). Collects a *clean*
             (unperturbed) LinearRegressor episode and writes
             data/datasets/linear_regressor_dataset.csv
             This CSV is the correct baseline for the GUI when using the
             neural policy — run it after training.

If the model artefacts are missing when option 2 is selected, the script
prints a clear error and exits without touching the filesystem.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Ensure project root is on the import path ─────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils import Config, default_env_config, select_env_config
from CityLearn.citylearn.citylearn import CityLearnEnv
from main_rbc import AdvancedRBC


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR    = PROJECT_ROOT / "results" / "linear_regressor"
DATASET_DIR  = PROJECT_ROOT / "data" / "datasets"

RBC_OUTPUT_PATH = DATASET_DIR / "advanced_rbc_dataset.csv"
LR_OUTPUT_PATH  = DATASET_DIR / "linear_regressor_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(agent, env) -> pd.DataFrame:
    """
    Run a complete episode and return a DataFrame with observations and actions.

    The SOC values are read directly from the building object at the correct
    time step (the post-step SOC), matching the fix applied in gui.py.

    Parameters
    ----------
    agent : any controller with a ``predict(obs_list)`` method
    env   : a reset-ready CityLearnEnv instance

    Returns
    -------
    pd.DataFrame  with columns:
        hour, indoor_temp, cooling_setpoint, outdoor_temp,
        outdoor_temp_predicted, cooling_demand, elec_price, carbon_intensity,
        solar_generation, occupant_count, electrical_storage_soc,
        dhw_storage_soc, dhw_demand,
        cooling_device, dhw_storage, electrical_storage,
        reward
    """
    obs, _ = env.reset()
    done   = False
    rows   = []

    obs_names    = env.observation_names[0]
    action_names = env.action_names[0]

    def get_obs(name):
        return obs[0][obs_names.index(name)] if name in obs_names else None

    def get_act(name, actions):
        return actions[0][action_names.index(name)] if name in action_names else None

    while not done:
        actions = agent.predict(obs)
        building = env.buildings[0]
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        indoor_temp = building.indoor_dry_bulb_temperature[building.time_step - 1]

        # Read SOC from the building object at the current (post-step) index.
        # After step() the env has advanced, so the acted-upon SOC is at
        # building.time_step - 1  (same fix as in gui.py and the original
        # collect_rbc_data.py).
        electrical_storage_soc = building.electrical_storage.soc[building.time_step - 1]
        dhw_storage_soc        = building.dhw_storage.soc[building.time_step - 1]

        rows.append({
            # Observations
            "hour":                   get_obs("hour"),
            "indoor_temp":            indoor_temp,
            "cooling_setpoint":       get_obs("indoor_dry_bulb_temperature_cooling_set_point"),
            "outdoor_temp":           get_obs("outdoor_dry_bulb_temperature"),
            "outdoor_temp_predicted": get_obs("outdoor_dry_bulb_temperature_predicted_1"),
            "cooling_demand":         get_obs("cooling_demand"),
            "elec_price":             get_obs("electricity_pricing"),
            "carbon_intensity":       get_obs("carbon_intensity"),
            "solar_generation":       get_obs("solar_generation"),
            "occupant_count":         get_obs("occupant_count"),
            "electrical_storage_soc": electrical_storage_soc,
            "dhw_storage_soc":        dhw_storage_soc,
            "dhw_demand":             get_obs("dhw_demand"),
            # Actions
            "cooling_device":     get_act("cooling_device",     actions),
            "dhw_storage":        get_act("dhw_storage",        actions),
            "electrical_storage": get_act("electrical_storage", actions),
            # Reward
            "reward": rewards[0],
        })

        obs  = next_obs
        done = terminated or truncated

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# RBC collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rbc(env: CityLearnEnv) -> None:
    """Simulate a full AdvancedRBC episode and save the dataset."""
    print("\nCollecting RBC data …")
    agent = AdvancedRBC(env)
    df    = _simulate(agent, env)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RBC_OUTPUT_PATH, index=False)
    print(f"  ✓  {df.shape[0]} timesteps saved → {RBC_OUTPUT_PATH}")
    print("     (Use this CSV as the baseline when running with RBC policy.)")
    print("     (Run linear_regressor.py on it to train the neural model.)")


# ─────────────────────────────────────────────────────────────────────────────
# Linear-Regressor collection
# ─────────────────────────────────────────────────────────────────────────────

def _check_model_artefacts() -> tuple[bool, str]:
    """
    Return (ok, message).  ok is True only when all required model files exist.
    """
    required = [
        MODEL_DIR / "feature_mean.npy",
        MODEL_DIR / "feature_std.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        return False, (
            "Missing normalisation artefacts:\n  " +
            "\n  ".join(missing)
        )

    # Check for the three per-target model files.
    # We import TARGET_COLS lazily to avoid importing torch at the top level.
    try:
        import linear_regressor as lr_mod
        target_cols = lr_mod.TARGET_COLS
    except Exception as exc:
        return False, f"Could not import linear_regressor.py: {exc}"

    missing_models = [
        str(MODEL_DIR / f"model_{t}.pth")
        for t in target_cols
        if not (MODEL_DIR / f"model_{t}.pth").exists()
    ]
    if missing_models:
        return False, (
            "Missing model files:\n  " +
            "\n  ".join(missing_models)
        )

    return True, "All model artefacts found."


class LinearRegressorAgent:
    """
    Thin wrapper that exposes the same ``predict(obs_list)`` interface as
    AdvancedRBC so it can be dropped into ``_simulate()`` unchanged.

    The agent loads the trained LinearRegressor models from MODEL_DIR and
    performs a clean (no-perturbation) forward pass at every step.
    """

    def __init__(self, env: CityLearnEnv):
        import torch
        import linear_regressor as lr_mod

        self._torch  = torch
        self._lr     = lr_mod
        self._env    = env

        # Resolve action order from the env (same logic as gui.py)
        # Fall back to lr.TARGET_COLS if the env isn't ready.
        try:
            # AdvancedRBC stores observation/action names on the env directly.
            rbc_tmp = AdvancedRBC(env)
            self._action_order = list(rbc_tmp.action_names[0])
        except Exception:
            self._action_order = list(lr_mod.TARGET_COLS)

        self._obs_names = env.observation_names[0]

        # Load normalisation stats
        mean = np.load(str(MODEL_DIR / "feature_mean.npy"))
        std  = np.load(str(MODEL_DIR / "feature_std.npy"))
        std[std == 0] = 1.0
        self._mean = mean
        self._std  = std

        # Load models
        self._models: dict = {}
        self._features: dict = {}
        for target in lr_mod.TARGET_COLS:
            domain_feats = lr_mod.DOMAIN_FEATURES[target]
            is_storage   = target in ("dhw_storage", "electrical_storage")
            model = lr_mod.LinearRegressor(
                input_dim=len(domain_feats), is_storage=is_storage
            )
            state = torch.load(
                str(MODEL_DIR / f"model_{target}.pth"), map_location="cpu"
            )
            model.load_state_dict(state)
            model.eval()
            self._models[target]   = model
            self._features[target] = domain_feats

        print(f"  LinearRegressor loaded from: {MODEL_DIR}")

    # Map CityLearn obs names → CSV feature names (matches CL_TO_LR in gui.py)
    _CL_TO_LR = {
        "indoor_dry_bulb_temperature":               "indoor_temp",
        "indoor_dry_bulb_temperature_cooling_set_point": "cooling_setpoint",
        "outdoor_dry_bulb_temperature":              "outdoor_temp",
        "outdoor_dry_bulb_temperature_predicted_1":  "outdoor_temp_predicted",
        "cooling_demand":                            "cooling_demand",
        "electricity_pricing":                       "elec_price",
        "carbon_intensity":                          "carbon_intensity",
        "solar_generation":                          "solar_generation",
        "occupant_count":                            "occupant_count",
        "hour":                                      "hour",
        "electrical_storage_soc":                    "electrical_storage_soc",
        "dhw_storage_soc":                           "dhw_storage_soc",
        "dhw_demand":                                "dhw_demand",
    }

    def _clamp(self, target: str, value: float) -> float:
        ranges = {
            "cooling_device":     (0.0,  1.0),
            "dhw_storage":        (-1.0, 1.0),
            "electrical_storage": (-1.0, 1.0),
        }
        lo, hi = ranges.get(target, (-1.0, 1.0))
        return max(lo, min(hi, value))

    def predict(self, obs_list) -> list:
        lr    = self._lr
        torch = self._torch

        # Build obs dict: CityLearn name → value
        obs_vec  = obs_list[0]
        obs_dict = {n: float(obs_vec[i]) for i, n in enumerate(self._obs_names)}

        # Re-map to CSV feature names
        lr_obs = {v: obs_dict.get(k, 0.0) for k, v in self._CL_TO_LR.items()}

        # Normalise the full 13-feature vector
        feat_cols = list(lr.FEATURE_COLS)
        row  = [lr_obs.get(c, 0.0) for c in feat_cols]
        norm = [
            (v - float(self._mean[i])) / float(self._std[i])
            for i, v in enumerate(row)
        ]

        # Per-target inference
        predictions: dict[str, float] = {}
        for target in lr.TARGET_COLS:
            sel = [norm[feat_cols.index(f)] for f in self._features[target]]
            x   = torch.tensor(sel, dtype=torch.float32)
            with torch.no_grad():
                val = self._models[target](x).item()
            predictions[target] = self._clamp(target, val)

        # Pack in the env's action order
        packed = [[predictions.get(a, 0.0) for a in self._action_order]]
        return packed


def collect_linear_regressor(env: CityLearnEnv, exit_on_missing: bool = True) -> None:
    """Check artefacts, simulate a clean LR episode, and save the dataset.

    ``exit_on_missing`` controls what happens when the model artefacts are
    not found: by default the script exits (standalone option 2), but when
    called as part of ``collect_both`` (option 3) it should just skip the
    LinearRegressor step and let the script finish normally, since the RBC
    dataset was already collected and saved successfully.
    """
    ok, msg = _check_model_artefacts()
    if not ok:
        print(f"\n  ✗  Cannot collect LinearRegressor data.\n  {msg}")
        print(
            "\n  Train the model first:\n"
            "    1. Collect the RBC dataset:  python collect_data.py  → option 1\n"
            "    2. Train the regressor:      python linear_regressor.py\n"
            "    3. Re-run this script        → option 2"
        )
        if exit_on_missing:
            sys.exit(1)
        return

    print(f"\n{msg}")
    print("Collecting LinearRegressor (clean, unperturbed) data …")
    agent = LinearRegressorAgent(env)
    df    = _simulate(agent, env)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LR_OUTPUT_PATH, index=False)
    print(f"  ✓  {df.shape[0]} timesteps saved → {LR_OUTPUT_PATH}")
    print("     (Load this CSV as the baseline in the GUI when using the neural policy.)")


def collect_both(env: CityLearnEnv) -> None:
    """Collect the RBC dataset, then (if model artefacts are available) the
    LinearRegressor dataset too, in a single run. Each call to ``_simulate``
    resets the env internally, so the same ``env`` instance can safely be
    reused for both episodes."""
    collect_rbc(env)
    print()
    collect_linear_regressor(env, exit_on_missing=False)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal menu
# ─────────────────────────────────────────────────────────────────────────────

def _model_status_line() -> str:
    """One-liner shown in the menu to inform the user about model availability."""
    ok, _ = _check_model_artefacts()
    if ok:
        return "✓  model artefacts found in results/linear_regressor/"
    else:
        return "✗  model artefacts NOT found  (train first with linear_regressor.py)"


def terminal_menu() -> str:
    """
    Display the interactive menu and return the user's choice as a string:
    '1'  → RBC
    '2'  → LinearRegressor
    'q'  → quit
    """
    lr_status = _model_status_line()
    separator = "─" * 60

    print()
    print(separator)
    print("  collect_data.py  —  Dataset collection for CityLearnThesis")
    print(separator)
    print(f"  Model status: {lr_status}")
    print(separator)
    print("  [1]  Collect RBC dataset")
    print("         → data/datasets/advanced_rbc_dataset.csv")
    print("         (baseline for RBC policy; training data for the regressor)")
    print()
    print("  [2]  Collect LinearRegressor dataset (clean/unperturbed)")
    print("         → data/datasets/linear_regressor_dataset.csv")
    print("         (baseline for the neural policy in the GUI)")
    print()
    print("  [3]  Collect BOTH datasets (RBC, then LinearRegressor)")
    print("         → both CSVs above, in one run")
    print("         (LinearRegressor step is skipped with a warning if model")
    print("          artefacts are missing — the RBC dataset is saved either way)")
    print()
    print("  [q]  Quit")
    print(separator)

    while True:
        choice = input("  Choose [1/2/3/q]: ").strip().lower()
        if choice in ("1", "2", "3", "q"):
            return choice
        print("  Invalid choice. Please enter 1, 2, 3, or q.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = Config()
    args   = config.args

    if args.data is None:
        args.data = "citylearn_challenge_2023_phase_1"

    # Build the environment once (shared between both collection branches)
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)  # type: ignore

    print(f"\nInitialising CityLearn environment  [{args.data}] …")
    env = CityLearnEnv(schema=schema, central_agent=True)
    print(f"  Observations : {env.observation_names[0]}")
    print(f"  Actions      : {env.action_names[0]}")

    # Interactive menu
    choice = terminal_menu()

    if choice == "q":
        print("\nAborted.")
        sys.exit(0)
    elif choice == "1":
        collect_rbc(env)
    elif choice == "2":
        collect_linear_regressor(env)
    elif choice == "3":
        collect_both(env)

    print("\nDone.\n")