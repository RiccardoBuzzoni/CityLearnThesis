"""
evaluate.py
-----------
Evaluation and comparative plotting script for the CityLearn thesis project.

This script runs full-episode simulations on a specified CityLearn environment
for up to three different agent types:
  1. AdvancedRBC        — Rule-Based Control baseline
  2. LinearRegressor    — Neural linear regressor surrogate policy
  3. OmnisafeActorAgent — Constrained Reinforcement Learning policy (PPOLag)

After running the evaluation loops, it generates comparative plots overlaying
the indoor temperature, DHW storage state of charge (SoC), and Electrical
storage SoC for all evaluated controllers.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.citylearn import CityLearnEnv
from utils import Config, default_env_config, select_env_config

# Import of AdvancedRBC agent
from main_rbc import AdvancedRBC

# Import of LinearRegressor and domain feature configuration
from linear_regressor import (
    LinearRegressor,
    FEATURE_COLS,
    TARGET_COLS,
    DOMAIN_FEATURES,
    main as train_linear_models,
)

ROOT_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "linear_regressor"

# Import of Omnisafe RL Agent wrapper
from rl_agent import OmnisafeActorAgent


# ──────────────────────────────────────────────
# LINEAR REGRESSOR AGENT
# ──────────────────────────────────────────────

class LinearRegressorAgent:
    """
    CityLearn-compatible agent that uses the three trained LinearRegressor
    models to predict actions from observations.

    Each model was trained only on the domain-relevant features for its
    target action, as defined in DOMAIN_FEATURES. Observations are
    normalised using the training-set statistics (mean, std) before
    being passed to the models.

    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment (used to read observation/action names).
    trained_models : dict[str, LinearRegressor]
        Trained PyTorch models keyed by target name.
    trained_features : dict[str, list[str]]
        Domain feature lists keyed by target name.
    mean : np.ndarray
        Per-feature training mean, shape (len(FEATURE_COLS),).
    std : np.ndarray
        Per-feature training std, shape (len(FEATURE_COLS),).
    """

    def __init__(
        self,
        env: CityLearnEnv,
        trained_models: dict,
        trained_features: dict,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.env              = env
        self.trained_models   = trained_models
        self.trained_features = trained_features
        self.mean             = mean
        self.std              = std

        # Pre-compute column indices for each target to avoid repeated lookups
        self._col_idx: dict[str, list[int]] = {
            target: [FEATURE_COLS.index(f) for f in feats]
            for target, feats in trained_features.items()
        }

    # Exact mapping used in collect_data.py to align feature names
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

    def predict(self, observations: list) -> list:
        """
        Predict actions from raw observations using the trained linear regressors.
        """
        obs_vec  = observations[0]
        obs_names = self.env.observation_names[0]
        act_names = self.env.action_names[0]

        # Create a dictionary of raw environment observations
        obs_dict = {n: float(obs_vec[i]) for i, n in enumerate(obs_names)}
        
        # Map CityLearn keys to internal Regressor feature names
        lr_obs = {v: obs_dict.get(k, 0.0) for k, v in self._CL_TO_LR.items()}

        # Build the feature vector in the exact FEATURE_COLS sequence
        row = [lr_obs.get(c, 0.0) for c in FEATURE_COLS]
        full_obs = np.array(row, dtype=np.float32)

        # Normalize using training statistics
        full_obs_norm = (full_obs - self.mean) / self.std

        # Predict actions using individual target models
        action = [0.0] * len(act_names)
        for target in TARGET_COLS:
            if target not in act_names:
                continue
            model   = self.trained_models[target]
            col_idx = self._col_idx[target]
            x       = torch.tensor(
                full_obs_norm[col_idx], dtype=torch.float32
            ).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                pred = model(x).item()

            action[act_names.index(target)] = pred

        # Clip actions to physical bounds
        CLIP_BOUNDS: dict[str, tuple[float, float]] = {
            "cooling_device":      (0.0,  1.0),
            "dhw_storage":         (-1.0, 1.0),
            "electrical_storage":  (-1.0, 1.0),
        }
        for target, (lo, hi) in CLIP_BOUNDS.items():
            if target in act_names:
                idx = act_names.index(target)
                action[idx] = float(np.clip(action[idx], lo, hi))

        return [action]


# ──────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ──────────────────────────────────────────────

def plot_cooling_device(res_rbc, res_reg=None, res_rl=None, save_path='cooling_device.png'):
    '''
    Plot indoor temperature, setpoint, and comfort band for RBC, Regressor, and RL agents.

    Args:
        res_rbc: Results dictionary from AdvancedRBC.
        res_reg: Optional results dictionary from LinearRegressorAgent.
        res_rl:  Optional results dictionary from OmnisafeActorAgent (RL).
        save_path: Path where the plot will be saved.
    '''
    time_steps   = range(res_rbc['env_h']['time_steps'])
    temp_data    = res_rbc['env_h']['temperature']
    indoor_temp  = np.array(temp_data['indoor_dry_bulb_temperature'])
    set_point    = np.array(temp_data['indoor_dry_bulb_temperature_set_point'])
    comfort_band = np.array(temp_data['comfort_band'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('Cooling Device — Indoor Temperature and Comfort Band')

    # Comfort band shading
    ax1.fill_between(time_steps, set_point + comfort_band, set_point - comfort_band,
                     color='green', alpha=0.15, label='Comfort band')

    # RBC Temperature curve
    ax1.plot(time_steps, indoor_temp,
             label='Indoor Temp (RBC)', linewidth=1.5, color='tab:blue')
    ax1.plot(time_steps, set_point,
             label='Setpoint', linestyle='--', linewidth=0.8, color='gray')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    if res_reg is not None:
        indoor_temp_reg = np.array(res_reg['env_h']['temperature']['indoor_dry_bulb_temperature'])
        ax1.plot(time_steps, indoor_temp_reg,
                 label='Indoor Temp (Regressor)', linewidth=1.5,
                 color='tab:orange', linestyle='--', alpha=0.7)

    if res_rl is not None:
        indoor_temp_rl = np.array(res_rl['env_h']['temperature']['indoor_dry_bulb_temperature'])
        ax1.plot(time_steps, indoor_temp_rl,
                 label='Indoor Temp (RL)', linewidth=1.5,
                 color='tab:purple', linestyle='-.', alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")


def plot_dhw_storage(res_rbc, res_reg=None, res_rl=None, save_path='dhw_storage.png'):
    '''
    Plot DHW storage SoC for RBC, Regressor, and RL agents.

    Args:
        res_rbc: Results dictionary from AdvancedRBC.
        res_reg: Optional results dictionary from LinearRegressorAgent.
        res_rl:  Optional results dictionary from OmnisafeActorAgent (RL).
        save_path: Path where the plot will be saved.
    '''
    time_steps  = range(res_rbc['env_h']['time_steps'])
    soc_rbc     = np.array(res_rbc['env_h']['dhw']['dhw_storage_soc'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('DHW Storage — State of Charge')

    ax1.plot(time_steps, soc_rbc,
             label='DHW SoC (RBC)', linewidth=1.5, color='tab:green')
    ax1.set_ylabel('State of Charge (kWh/kWh_capacity)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    if res_reg is not None:
        soc_reg = np.array(res_reg['env_h']['dhw']['dhw_storage_soc'])
        ax1.plot(time_steps, soc_reg,
                 label='DHW SoC (Regressor)', linewidth=1.5,
                 color='tab:red', linestyle='--', alpha=0.7)

    if res_rl is not None:
        soc_rl = np.array(res_rl['env_h']['dhw']['dhw_storage_soc'])
        ax1.plot(time_steps, soc_rl,
                 label='DHW SoC (RL)', linewidth=1.5,
                 color='tab:purple', linestyle='-.', alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")


def plot_electrical_storage(res_rbc, res_reg=None, res_rl=None, save_path='electrical_storage.png'):
    '''
    Plot electrical storage SoC for RBC, Regressor, and RL agents.

    Args:
        res_rbc: Results dictionary from AdvancedRBC.
        res_reg: Optional results dictionary from LinearRegressorAgent.
        res_rl:  Optional results dictionary from OmnisafeActorAgent (RL).
        save_path: Path where the plot will be saved.
    '''
    time_steps  = range(res_rbc['env_h']['time_steps'])
    soc_rbc     = np.array(res_rbc['env_h']['electrical_storage_soc'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('Electrical Storage — State of Charge')

    ax1.plot(time_steps, soc_rbc,
             label='Electrical SoC (RBC)', linewidth=1.5, color='tab:orange')
    ax1.set_ylabel('State of Charge (kWh/kWh_capacity)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    if res_reg is not None:
        soc_reg = np.array(res_reg['env_h']['electrical_storage_soc'])
        ax1.plot(time_steps, soc_reg,
                 label='Electrical SoC (Regressor)', linewidth=1.5,
                 color='tab:blue', linestyle='--', alpha=0.7)

    if res_rl is not None:
        soc_rl = np.array(res_rl['env_h']['electrical_storage_soc'])
        ax1.plot(time_steps, soc_rl,
                 label='Electrical SoC (RL)', linewidth=1.5,
                 color='tab:purple', linestyle='-.', alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")


def plot_discomfort(res_rbc, res_reg=None, res_rl=None, save_path='discomfort.png'):
    """Plot total, hot, and cold discomfort proportions per controller."""
    results = {'RBC': res_rbc}
    if res_reg is not None:
        results['Regressor'] = res_reg
    if res_rl is not None:
        results['RL'] = res_rl

    values = []
    for result in results.values():
        temperature = result['env_h']['temperature']
        indoor = np.asarray(temperature['indoor_dry_bulb_temperature'])
        setpoint = np.asarray(temperature['indoor_dry_bulb_temperature_set_point'])
        band = np.asarray(temperature['comfort_band'])
        hot = indoor > setpoint + band
        cold = indoor < setpoint - band
        values.append([
            float(np.mean(hot | cold)),
            float(np.mean(hot)),
            float(np.mean(cold)),
        ])

    labels = list(results)
    values = np.asarray(values)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (name, color) in enumerate([
        ('Total', 'tab:blue'), ('Hot', 'tab:red'), ('Cold', 'tab:cyan')
    ]):
        ax.bar(x + (index - 1) * width, values[:, index], width,
               label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Proportion of timesteps')
    ax.set_ylim(0, max(0.01, float(values.max()) * 1.25))
    ax.set_title('Thermal Discomfort')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='png', dpi=300)
    plt.close(fig)
    print(f"Plot saved in: {save_path}")


# ──────────────────────────────────────────────
# EVALUATION LOOP
# ──────────────────────────────────────────────

def evaluate_agent(agent_class, env):
    """
    Simulates episode and records results for the given agent in the provided environment.
    Accepts either an agent class (instantiated with env) or an already-instantiated agent object.
    """
    if isinstance(agent_class, type):
        agent = agent_class(env)
    else:
        agent = agent_class

    obs, _ = env.reset()

    # Track action values at each timestep for plotting
    act_names = env.action_names[0]
    action_history: dict[str, list] = {name: [] for name in act_names}

    done = False
    while not done:
        actions = agent.predict(obs)
        # Record actions before stepping
        for j, name in enumerate(act_names):
            action_history[name].append(actions[0][j])
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

    b = env.buildings[0]

    results = {
        'env_h': {
            'time_steps': env.time_steps - 1,

            'actions': action_history,

            'temperature': {
                'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature[:-1],
                'indoor_dry_bulb_temperature_set_point': b.indoor_dry_bulb_temperature_cooling_set_point[:-1],
                'comfort_band': b.comfort_band,
                'cooling_demand': b.cooling_demand[:-1]
            },

            'electrical_storage_soc': b.electrical_storage.soc[:-1],

            'dhw': {
                'dhw_storage_soc': b.dhw_storage.soc[:-1],
                'dhw_demand': b.dhw_demand[:-1]
            },

            'grid_and_solar': {
                'elec_price': b.pricing.electricity_pricing[:-1],
                'solar_generation': b.solar_generation[:-1]
            }
        }
    }

    cb = b.comfort_band
    if not hasattr(cb, '__len__'):
        results['env_h']['temperature']['comfort_band'] = [cb] * results['env_h']['time_steps']
    elif len(cb) > results['env_h']['time_steps']:
        results['env_h']['temperature']['comfort_band'] = cb[:-1]

    return results


def _find_rl_model() -> str | None:
    """Find the path to the RL actor model file."""
    root_dir = Path(__file__).parent.parent
    
    # 1. Primary path: results/rl_model/actor.pt
    primary_actor = root_dir / "results" / "rl_model" / "actor.pt"
    if primary_actor.exists():
        return str(primary_actor)

    # 2. Fallback: search in results/experiments/
    exp_dir = root_dir / "results" / "experiments"
    if exp_dir.exists():
        candidates = list(exp_dir.glob("**/actor.pt")) + \
                     list(exp_dir.glob("**/actor_net_custom.pt")) + \
                     list(exp_dir.glob("**/actor_net_standard.pt"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])

    return None


def main():
    conf = Config()
    args = conf.args
    
    if args.data is None:
        args.data = 'citylearn_challenge_2023_phase_1'

    # Load CityLearn schema
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data) # type: ignore

    print(f"Start evaluation with dataset: {args.data}")

    # ── 1. RBC evaluation ─────────────────────────────────────────────
    env_rbc = CityLearnEnv(schema=schema, central_agent=True)
    print("\nRunning AdvancedRBC...")
    results_rbc = evaluate_agent(AdvancedRBC, env_rbc)

    # ── 2. Linear Regressor evaluation ────────────────────────────────
    results_reg = None
    print("\nLoading Linear Regressor artefacts...")
    try:
        mean = np.load(os.path.join(OUTPUT_DIR, "feature_mean.npy"))
        std  = np.load(os.path.join(OUTPUT_DIR, "feature_std.npy"))

        trained_models = {}
        for target in TARGET_COLS:
            domain_feats = DOMAIN_FEATURES[target]
            is_storage = target in ["dhw_storage", "electrical_storage"]
            
            model = LinearRegressor(input_dim=len(domain_feats), is_storage=is_storage)
            model_path = os.path.join(OUTPUT_DIR, f"model_{target}.pth")
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                model.eval()
                trained_models[target] = model
            else:
                raise FileNotFoundError(f"Model file {model_path} not found.")

        env_reg = CityLearnEnv(schema=schema, central_agent=True)
        reg_agent = LinearRegressorAgent(
            env                 = env_reg,
            trained_models      = trained_models,
            trained_features    = DOMAIN_FEATURES,
            mean                = mean,
            std                 = std,
        )
        print("Running LinearRegressorAgent...")
        results_reg = evaluate_agent(reg_agent, env_reg)
    except Exception as exc:
        print(f"  [SKIP] Could not run LinearRegressorAgent: {exc}")

    # ── 3. RL Agent evaluation ────────────────────────────────────────
    results_rl = None
    rl_model_path = _find_rl_model()
    if rl_model_path:
        print(f"\nLoading RL model from: {rl_model_path}")
        try:
            env_rl = CityLearnEnv(schema=schema, central_agent=True)
            rl_agent = OmnisafeActorAgent(fname=rl_model_path, env=env_rl)
            print("Running OmnisafeActorAgent...")
            results_rl = evaluate_agent(rl_agent, env_rl)
        except Exception as exc:
            print(f"  [SKIP] Could not run OmnisafeActorAgent: {exc}")
    else:
        print("\n  [SKIP] No RL actor model found under results/rl_model/ or results/experiments/.")

    # ── 4. Plotting — overlaid curves ─────────────────────────────────
    print("\nGenerating evaluation plots...")
    eval_plots_dir = ROOT_DIR / "results" / "evaluation_plots"
    eval_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_cooling_device(
        results_rbc, results_reg, results_rl,
        save_path=str(eval_plots_dir / "cooling_device.png")
    )
    plot_dhw_storage(
        results_rbc, results_reg, results_rl,
        save_path=str(eval_plots_dir / "dhw_storage.png")
    )
    plot_electrical_storage(
        results_rbc, results_reg, results_rl,
        save_path=str(eval_plots_dir / "electrical_storage.png")
    )
    plot_discomfort(
        results_rbc, results_reg, results_rl,
        save_path=str(eval_plots_dir / "discomfort.png")
    )

if __name__ == '__main__':
    main()