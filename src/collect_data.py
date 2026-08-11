"""
collect_data.py
───────────────
Interactive data-collection script for the CityLearn thesis project.

Usage
-----
    python collect_data.py [--data DATASET] [--custom]

The script presents a terminal menu to choose which controller to simulate:

  1. RBC              → Collects an AdvancedRBC episode and writes
                        data/datasets/advanced_rbc_dataset.csv
                        (also the training data required by linear_regressor.py)

  2. Linear Regressor → Collects a clean LinearRegressor episode and writes
                        data/datasets/linear_regressor_dataset.csv

  3. RL Agent         → Collects an RL policy episode (Omnisafe actor) and writes
                        data/datasets/rl_dataset.csv

  4. All Datasets     → Collects RBC, Linear Regressor, and RL datasets in sequence.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# ── Ensure project root is on the import path ─────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils import Config, default_env_config, select_env_config
from CityLearn.citylearn.citylearn import CityLearnEnv
from CityLearn.citylearn.wrappers import NormalizedObservationWrapper
from main_rbc import AdvancedRBC


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR    = PROJECT_ROOT / "results" / "linear_regressor"
RL_MODEL_DIR = PROJECT_ROOT / "results" / "rl_model"
EXP_DIR      = PROJECT_ROOT / "results" / "experiments"
DATASET_DIR  = PROJECT_ROOT / "data" / "datasets"

RBC_OUTPUT_PATH = DATASET_DIR / "advanced_rbc_dataset.csv"
LR_OUTPUT_PATH  = DATASET_DIR / "linear_regressor_dataset.csv"
RL_OUTPUT_PATH  = DATASET_DIR / "rl_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(agent, env) -> pd.DataFrame:
    """
    Run a complete episode and return a DataFrame with observations and actions.
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

        # Read SOC from building at post-step index
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


# ─────────────────────────────────────────────────────────────────────────────
# Linear-Regressor collection
# ─────────────────────────────────────────────────────────────────────────────

def _check_lr_artefacts() -> tuple[bool, str]:
    """Check if Linear Regressor artefacts are present."""
    required = [
        MODEL_DIR / "feature_mean.npy",
        MODEL_DIR / "feature_std.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        return False, "Missing normalisation artefacts:\n  " + "\n  ".join(missing)

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
        return False, "Missing model files:\n  " + "\n  ".join(missing_models)

    return True, "Linear Regressor artefacts found."


class LinearRegressorAgent:
    """Wrapper for LinearRegressor models."""

    def __init__(self, env: CityLearnEnv):
        import linear_regressor as lr_mod

        self._torch  = torch
        self._lr     = lr_mod
        self._env    = env

        try:
            rbc_tmp = AdvancedRBC(env)
            self._action_order = list(rbc_tmp.action_names[0])
        except Exception:
            self._action_order = list(lr_mod.TARGET_COLS)

        self._obs_names = env.observation_names[0]

        mean = np.load(str(MODEL_DIR / "feature_mean.npy"))
        std  = np.load(str(MODEL_DIR / "feature_std.npy"))
        std[std == 0] = 1.0
        self._mean = mean
        self._std  = std

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
        lr = self._lr

        obs_vec  = obs_list[0]
        obs_dict = {n: float(obs_vec[i]) for i, n in enumerate(self._obs_names)}
        lr_obs   = {v: obs_dict.get(k, 0.0) for k, v in self._CL_TO_LR.items()}

        feat_cols = list(lr.FEATURE_COLS)
        row  = [lr_obs.get(c, 0.0) for c in feat_cols]
        norm = [
            (v - float(self._mean[i])) / float(self._std[i])
            for i, v in enumerate(row)
        ]

        predictions: dict[str, float] = {}
        for target in lr.TARGET_COLS:
            sel = [norm[feat_cols.index(f)] for f in self._features[target]]
            x   = torch.tensor(sel, dtype=torch.float32)
            with torch.no_grad():
                val = self._models[target](x).item()
            predictions[target] = self._clamp(target, val)

        return [[predictions.get(a, 0.0) for a in self._action_order]]


def collect_linear_regressor(env: CityLearnEnv, exit_on_missing: bool = True) -> None:
    """Simulate a clean LR episode and save the dataset."""
    ok, msg = _check_lr_artefacts()
    if not ok:
        print(f"\n  ✗  Cannot collect LinearRegressor data.\n  {msg}")
        if exit_on_missing:
            sys.exit(1)
        return

    print("Collecting LinearRegressor data …")
    agent = LinearRegressorAgent(env)
    df    = _simulate(agent, env)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LR_OUTPUT_PATH, index=False)
    print(f"  ✓  {df.shape[0]} timesteps saved → {LR_OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Reinforcement Learning (RL) collection
# ─────────────────────────────────────────────────────────────────────────────

def _find_rl_model() -> Path | None:
    """Find the trained RL actor model file."""
    # 1. Primary path: results/rl_model/actor.pt
    primary_actor = RL_MODEL_DIR / "actor.pt"
    if primary_actor.exists():
        return primary_actor

    # 2. Fallback: search in results/experiments/
    if EXP_DIR.exists():
        candidates = list(EXP_DIR.glob("**/actor.pt")) + \
                     list(EXP_DIR.glob("**/actor_net_custom.pt")) + \
                     list(EXP_DIR.glob("**/actor_net_standard.pt"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]

    return None


def _check_rl_artefacts() -> tuple[bool, str, Path | None]:
    """Check if an RL model file is available."""
    model_path = _find_rl_model()
    if model_path is None:
        return False, f"No RL actor model found under {RL_MODEL_DIR} or {EXP_DIR}", None
    return True, f"RL model found: {model_path.relative_to(PROJECT_ROOT)}", model_path


class RLAgent:
    """Wrapper for the Omnisafe TorchScript RL Agent."""

    def __init__(self, model_path: Path, env: CityLearnEnv):
        self.actor = torch.jit.load(str(model_path))
        self.actor.eval()
        self._action_space = env.action_space[0]
        self._normalizer   = NormalizedObservationWrapper(env)
        print(f"  RL Agent loaded from: {model_path.relative_to(PROJECT_ROOT)}")

    def predict(self, observations: list[list[float]]) -> list[list[float]]:
        norm_observations = self._normalizer.observation(observations)
        actions = []

        for norm_obs in norm_observations:
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32)
            with torch.no_grad():
                output = self.actor(obs_t)

            mean, log_std = output.chunk(2, dim=-1)

            # DEBUG
            raw_action = torch.tanh(mean).detach().cpu().numpy()
            print("raw:", raw_action, "| rescaled:", self._rescale(raw_action))

            action = torch.tanh(mean).detach().cpu().numpy()
            action = self._rescale(action)
            actions.append(action.tolist())

        return actions

    def _rescale(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        low    = self._action_space.low
        high   = self._action_space.high
        scaled = low + (action + 1.0) / 2.0 * (high - low)
        return scaled.astype(np.float32)


def collect_rl(env: CityLearnEnv, exit_on_missing: bool = True) -> None:
    """Simulate an RL episode and save the dataset."""
    ok, msg, model_path = _check_rl_artefacts()
    if not ok or model_path is None:
        print(f"\n  ✗  Cannot collect RL data.\n  {msg}")
        if exit_on_missing:
            sys.exit(1)
        return

    print(f"\nCollecting RL data …")
    agent = RLAgent(model_path, env)
    df    = _simulate(agent, env)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RL_OUTPUT_PATH, index=False)
    print(f"  ✓  {df.shape[0]} timesteps saved → {RL_OUTPUT_PATH}")


def collect_all(env: CityLearnEnv) -> None:
    """Collect RBC, Linear Regressor, and RL datasets in sequence."""
    collect_rbc(env)
    print()
    collect_linear_regressor(env, exit_on_missing=False)
    print()
    collect_rl(env, exit_on_missing=False)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal menu
# ─────────────────────────────────────────────────────────────────────────────

def _model_status_lines() -> tuple[str, str]:
    """Return status lines for LR and RL models."""
    lr_ok, _ = _check_lr_artefacts()
    lr_status = "✓  LR artefacts found" if lr_ok else "✗  LR artefacts NOT found"

    rl_ok, _, _ = _check_rl_artefacts()
    rl_status = "✓  RL model found" if rl_ok else "✗  RL model NOT found"

    return lr_status, rl_status


def terminal_menu() -> str:
    """Display the menu choices."""
    lr_status, rl_status = _model_status_lines()
    separator = "─" * 60

    print()
    print(separator)
    print("  collect_data.py  —  Dataset collection for CityLearnThesis")
    print(separator)
    print(f"  Linear Regressor Status : {lr_status}")
    print(f"  RL Model Status         : {rl_status}")
    print(separator)
    print("  [1]  Collect RBC dataset")
    print("         → data/datasets/advanced_rbc_dataset.csv")
    print()
    print("  [2]  Collect LinearRegressor dataset")
    print("         → data/datasets/linear_regressor_dataset.csv")
    print()
    print("  [3]  Collect RL dataset")
    print("         → data/datasets/rl_dataset.csv")
    print()
    print("  [4]  Collect ALL datasets (RBC, LinearRegressor, RL)")
    print("         → all CSVs in sequence")
    print()
    print("  [q]  Quit")
    print(separator)

    while True:
        choice = input("  Choose [1/2/3/4/q]: ").strip().lower()
        if choice in ("1", "2", "3", "4", "q"):
            return choice
        print("  Invalid choice. Please enter 1, 2, 3, 4, or q.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = Config()
    args   = config.args

    if args.data is None:
        args.data = "citylearn_challenge_2023_phase_1"

    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)  # type: ignore

    print(f"\nInitialising CityLearn environment  [{args.data}] …")
    env = CityLearnEnv(schema=schema, central_agent=True)

    choice = terminal_menu()

    if choice == "q":
        print("\nAborted.")
        sys.exit(0)
    elif choice == "1":
        collect_rbc(env)
    elif choice == "2":
        collect_linear_regressor(env)
    elif choice == "3":
        collect_rl(env)
    elif choice == "4":
        collect_all(env)

    print("\nDone.\n")