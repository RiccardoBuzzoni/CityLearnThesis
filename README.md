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
# From main direcotry
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
```

> **Note:** `lime` must be installed with `--no-deps` to prevent it from upgrading
> `numpy` to a version incompatible with CityLearn (`numpy<2.0.0` is required).

---

## Usage
 
All scripts should be run from the `scripts/` directory.

### RBC Pipeline (run in order)
 
**Step 1 — Data collection**
```console
python collect_rbc_data.py
```
Runs a full AdvancedRBC simulation and saves the dataset to `datasets/advanced_rbc_dataset.csv`.
 
**Step 2 — Linear regressor training**
```console
python linear_regressor.py
```
Trains three linear regressors (one per action) and saves the training plots to `../linear_regressor_results/`.
 
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

## Module Descriptions
 
### `main_rbc.py` — AdvancedRBC
Implements `AdvancedRBC`, a Rule-Based Controller with multi-threshold logic for controlling three building systems:
 
- **`cooling_device`** `[0, 1]`: manages cooling based on indoor temperature, setpoint, predicted outdoor temperature, electricity price, carbon intensity, solar generation and occupant presence. Distinguishes peak hours (12–17) from off-peak and night hours, and applies a configurable comfort band.
- **`dhw_storage`** `[-1, 1]`: manages the charge/discharge of the domestic hot water storage based on SOC level, time of day, DHW demand, electricity price and solar generation.
- **`electrical_storage`** `[-1, 1]`: manages the battery based on SOC, time of day, solar generation, electricity price and carbon intensity.
 
The file also includes `run_simulation()` for running the monthly simulation.
 
---

### `collect_rbc_data.py` — Dataset Collection
Runs a full AdvancedRBC simulation and collects at each timestep the 13 observations and the 3 action values produced by the controller, saving the result to a CSV file (`datasets/advanced_rbc_dataset.csv`). This dataset is used as input for linear regressor training. The simulation covers one month (719 hourly timesteps).
 
---
 
### `linear_regressor.py` — Linear Regressor
Trains three linear regressors in PyTorch, one per controller action. Each regressor is trained exclusively on the observations that AdvancedRBC actually uses for that specific decision (domain knowledge feature selection), eliminating spurious correlations. The dataset is split temporally: the first 3 weeks (504 timesteps) for training and the last week (215 timesteps) for evaluation. MAE and MSE plots on both train and test sets are produced for each model.
 
Feature subsets per target:
- `cooling_device`: 10 features (temperature, setpoint, demand, price, carbon intensity, solar generation, occupant count, hour)
- `dhw_storage`: 5 features (SOC, DHW demand, electricity price, solar generation, hour)
- `electrical_storage`: 4 features (SOC, solar generation, electricity price, hour)
 
---
 
### `lime_analysis.py` — LIME Analysis
Applies LIME (Local Interpretable Model-agnostic Explanations) to the three trained linear regressors to explain AdvancedRBC's decisions. For each target, it selects 5 representative timesteps from the evaluation set (maximum action, minimum action, mean action, worst prediction and best prediction) and computes LIME weights on each of them. It produces a **mean feature importance** plot for each of the three actions, saved to `../lime_analysis_results/`. The explainer is built on the training set filtered to domain features only, ensuring that LIME's perturbations stay consistent with the model's input distribution.
 
---
 
### `evaluate.py` — Evaluation & Comparison
Sequentially runs the AdvancedRBC and LinearRegressorAgent simulations (an agent that uses the three trained regressors to produce actions) on the same CityLearn schema, producing three overlaid comparison plots:
 
- **`cooling_device.png`**: `cooling_device` action value, indoor temperature, setpoint and comfort band for both agents
- **`dhw_storage.png`**: `dhw_storage` action value and DHW storage SOC for both agents
- **`electrical_storage.png`**: `electrical_storage` action value and battery SOC for both agents
 
RBC curves are shown as solid lines, regressor curves as dashed lines. Plots are saved to `evaluation_plots/`.
 
---