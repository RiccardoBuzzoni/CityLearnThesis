# Omnisafe Wrapper for CityLearn Environment

## Environment Setup

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

2. Setup `omnisafe`
```console
# Create conda environment
cd omnisafe
conda env create --file conda-recipe.yaml

# Activate and setup omnisafe
conda activate omnisafe
pip install -e .

# Update `torch` and `torchvision`
pip install torch==2.8.0
pip install torchvision==0.23.0
``` 

3. Setup `citylearn`:
```console
# From main directory
cd CityLearn
pip install -e .
```

4. Additional requirements
```console
# Progress bar visualization
pip install ipywidgets

# SB3 comparison
pip install stable-baselines3==2.0.0

# LIME — Local Interpretable Model-agnostic Explanations
pip install lime==<VERSION> --no-deps

# GUI dependencies (tkinter is included in standard Python installations)
pip install matplotlib pandas
```

> **Note:** `lime` must be installed with `--no-deps` to prevent it from upgrading
> `numpy` to a version incompatible with CityLearn (`numpy<2.0.0` is required).

> **Note:** `tkinter` ships with standard Python on most platforms. On Ubuntu/WSL2 it
> may need to be installed separately: `sudo apt install python3-tk`.

---

## Usage
 
All scripts should be run from the `src/` directory.

### RBC Pipeline (run in order)
 
**Step 1 — Data collection**
```console
python collect_data.py
```
Presents an interactive menu:
- **Option 1 — RBC dataset**: runs a full AdvancedRBC simulation and saves the result to `datasets/advanced_rbc_dataset.csv`. Always available; required before training.
- **Option 2 — LinearRegressor dataset**: runs a clean (unperturbed) LinearRegressor simulation and saves the result to `datasets/linear_regressor_dataset.csv`. Only available after the model has been trained (Step 2). This CSV is used as the baseline in the GUI when running the neural policy.

**Step 2 — Linear regressor training**
```console
python linear_regressor.py
```
Trains three linear regressors (one per action) and saves the training plots and model weights to `../linear_regressor_results/`.
 
**Step 3 — LIME analysis**
```console
python lime_analysis.py
```
Runs the explainability analysis and saves the feature importance plots to `../lime_analysis_results/`.
 
**Step 4 — Evaluation and comparison**
```console
python evaluate.py
```
Runs both the AdvancedRBC and the linear regressor simulations and saves the comparison plots to `../evaluation_plots/`.

---

### GUI

```console
python gui.py
```

Launches an interactive Tkinter GUI for running and comparing simulations in real time.

**Features:**

- **Controller selection**: switch between RBC (AdvancedRBC) and Neural policy (LinearRegressor) at any time.
- **Dataset baseline**: load a pre-collected CSV (`advanced_rbc_dataset.csv` for RBC, `linear_regressor_dataset.csv` for the neural policy) to use as a clean baseline for comparison.
- **Perturbations**: add, stack, and remove perturbations on any of the three device actions (`cooling_device`, `dhw_storage`, `electrical_storage`) before or during a simulation.
- **Step / Start / Stop / Reset**: run the simulation step-by-step or continuously; stop and resume at any point.
- **Live plot**: toggle live plotting to see the baseline vs. perturbed curves update in real time as the simulation progresses.
- **Session export**: each simulation session is automatically saved to `simulations/sim_<policy>_<timestamp>.csv` for later analysis.
- **Auto-load**: on startup the GUI automatically searches for trained model weights in `../linear_regressor_results/` and loads them if found.

**Typical workflow:**

1. Collect the appropriate baseline dataset with `collect_data.py`.
2. Launch `gui.py`, select the desired controller, and load the matching dataset.
3. Add perturbations if needed, then start the simulation.
4. Use the plot panel to compare the baseline and perturbed curves across all observations and actions.

---

## Module Descriptions
 
### `main_rbc.py` — AdvancedRBC
Implements `AdvancedRBC`, a Rule-Based Controller with multi-threshold logic for controlling three building systems:
 
- **`cooling_device`** `[0, 1]`: manages cooling based on indoor temperature, setpoint, predicted outdoor temperature, electricity price, carbon intensity, solar generation and occupant presence. Distinguishes peak hours (12–17) from off-peak and night hours, and applies a configurable comfort band.
- **`dhw_storage`** `[-1, 1]`: manages the charge/discharge of the domestic hot water storage based on SOC level, time of day, DHW demand, electricity price and solar generation.
- **`electrical_storage`** `[-1, 1]`: manages the battery based on SOC, time of day, solar generation, electricity price and carbon intensity.
 
The file also includes `run_simulation()` for running the monthly simulation.
 
---

### `collect_data.py` — Dataset Collection
Interactive script that presents a terminal menu for collecting simulation datasets:

- **RBC dataset** (`datasets/advanced_rbc_dataset.csv`): always available; runs a full AdvancedRBC episode and records the 13 observations and 3 actions at each timestep. Used as training data for the linear regressor and as the baseline in the GUI when running the RBC policy.
- **LinearRegressor dataset** (`datasets/linear_regressor_dataset.csv`): available only after training. Runs a clean (unperturbed) LinearRegressor episode with the same structure. Used as the baseline in the GUI when running the neural policy.

The script checks for the presence of model artefacts before attempting to collect the regressor dataset, and exits with a clear error and remediation steps if they are missing. The simulation covers one month (719 hourly timesteps).
 
---
 
### `linear_regressor.py` — Linear Regressor
Trains three linear regressors in PyTorch, one per controller action. Each regressor is trained exclusively on the observations that AdvancedRBC actually uses for that specific decision (domain knowledge feature selection), eliminating spurious correlations. The dataset is split temporally: the first 3 weeks (504 timesteps) for training and the last week (215 timesteps) for evaluation. MAE and MSE plots on both train and test sets are produced for each model.
 
Feature subsets per target:
- `cooling_device`: 10 features (temperature, setpoint, demand, price, carbon intensity, solar generation, occupant count, hour)
- `dhw_storage`: 3 features (electricity price, solar generation, hour)
- `electrical_storage`: 2 features (solar generation, hour)
 
---
 
### `lime_analysis.py` — LIME Analysis
Applies LIME (Local Interpretable Model-agnostic Explanations) to the three trained linear regressors to explain AdvancedRBC's decisions. For each target, it selects 5 representative timesteps from the evaluation set (maximum action, minimum action, mean action, worst prediction and best prediction) and computes LIME weights on each of them. It produces a **mean feature importance** plot for each of the three actions, saved to `../lime_analysis_results/`. The explainer is built on the training set filtered to domain features only, ensuring that LIME's perturbations stay consistent with the model's input distribution.
 
---
 
### `evaluate.py` — Evaluation & Comparison
Sequentially runs the AdvancedRBC and LinearRegressorAgent simulations (an agent that uses the three trained regressors to produce actions) on the same CityLearn schema, producing three overlaid comparison plots:
 
- **`cooling_device.png`**: `cooling_device` action value, indoor temperature, setpoint and comfort band for both agents
- **`dhw_storage.png`**: `dhw_storage` action value and DHW storage SOC for both agents
- **`electrical_storage.png`**: `electrical_storage` action value and battery SOC for both agents
 
RBC curves are shown as solid lines, regressor curves as dashed lines. Plots are saved to `../evaluation_plots/`.

---

### `gui.py` — Interactive GUI
Tkinter-based graphical interface for running real-time simulations with perturbations. See the [GUI](#gui) section above for full details.
 
---