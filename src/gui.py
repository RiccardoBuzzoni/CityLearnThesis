import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, csv, json, tkinter as tk
from typing import Any
from tkinter import ttk, filedialog
from datetime import datetime

try:
    import linear_regressor as lr
except Exception:
    lr = None

try:
    from main_rbc import AdvancedRBC
    from CityLearn.citylearn.citylearn import CityLearnEnv
    from CityLearn.citylearn.wrappers import NormalizedObservationWrapper
    from utils import default_env_config, Config
    CITYLEARN_AVAILABLE = True
except Exception as e:
    CITYLEARN_AVAILABLE = False
    CITYLEARN_ERROR = str(e)
    AdvancedRBC = CityLearnEnv = default_env_config = None
    NormalizedObservationWrapper = None

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import torch

# Colours
BG_APP       = "#ECECEC"
BG_CARD      = "#FFFFFF"
BLUE         = "#007AFF"
BLUE_LIGHT   = "#E8F2FF"
RED          = "#FF3B30"
GREEN        = "#34C759"
ORANGE       = "#FF9500"
ORANGE_LIGHT = "#FFF4E0"
PURPLE       = "#AF52DE"   # perturbed curve (run 1)
GRAY         = "#E5E5EA"

# Palette used to colour each perturbed simulation run. The first run keeps
# PURPLE (the historical single-run colour); every following run cycles to the
# next colour so already-finished runs stay visible while a new one is drawn
# on top. Run 2 is the dark green requested for side-by-side comparison.
RUN_COLORS = [
    "#AF52DE",   # purple      – run 1
    "#1B5E20",   # dark green  – run 2
    "#FF2D55",   # pink/red    – run 3
    "#00838F",   # teal        – run 4
    "#8D6E63",   # brown       – run 5
    "#5856D6",   # indigo      – run 6
]


def run_color(idx):
    """Colour for the perturbed curve of run #idx (0-based), cycling if needed."""
    return RUN_COLORS[idx % len(RUN_COLORS)]

OBS_NAMES = [
    "hour",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "cooling_demand",
    "electricity_pricing",
    "carbon_intensity",
    "solar_generation",
    "occupant_count",
    "electrical_storage_soc",
    "dhw_storage_soc",
    "dhw_demand",
]
OBS_NAMES_DISPLAY = OBS_NAMES + ["comfort_band"]

OBS_DISPLAY = {
    "hour":                                          "hour",
    "indoor_dry_bulb_temperature":                   "indoor_temp",
    "indoor_dry_bulb_temperature_cooling_set_point": "cool_setpt",
    "outdoor_dry_bulb_temperature":                  "out_temp",
    "outdoor_dry_bulb_temperature_predicted_1":      "out_temp_pred1",
    "cooling_demand":                                "cool_demand",
    "electricity_pricing":                           "elec_price",
    "carbon_intensity":                              "carbon_int",
    "solar_generation":                              "solar_gen",
    "occupant_count":                                "occupants",
    "electrical_storage_soc":                        "elec_soc",
    "dhw_storage_soc":                               "dhw_soc",
    "dhw_demand":                                    "dhw_demand",
}
DISPLAY_TO_OBS      = {v: k for k, v in OBS_DISPLAY.items()}
OBS_DISPLAY_CHOICES = [OBS_DISPLAY[n] for n in OBS_NAMES]
# "reward" is plottable too (a single series, no clean/pert split) so runs can
# be compared by their reward curve.
PLOT_CHOICES        = OBS_DISPLAY_CHOICES + ["reward"]
ACTION_NAMES  = ["cooling_device", "dhw_storage", "electrical_storage"]
ACTION_RANGES = {
    "cooling_device":     (0.0,  1.0),
    "dhw_storage":        (-1.0, 1.0),
    "electrical_storage": (-1.0, 1.0),
}
OBS_PERTURB_TARGETS = [
    "outdoor_dry_bulb_temperature", "solar_generation",
    "electricity_pricing", "carbon_intensity", "occupant_count",
    "cooling_demand", "dhw_demand", "indoor_dry_bulb_temperature",
    "electrical_storage_soc", "dhw_storage_soc",
    "indoor_dry_bulb_temperature_cooling_set_point",
]
# Shorthand used by the plot helpers below (comfort-band shading, setpoint
# delta annotation, ...). Kept as one constant so every reference to the
# cooling setpoint observation stays in sync.
SP_COL = "indoor_dry_bulb_temperature_cooling_set_point"
TEMP_COL = "indoor_dry_bulb_temperature"
ACTION_PERTURB_TARGETS = {
    "cooling_device [0, 1]":      (0,  1),
    "dhw_storage [-1, 1]":        (-1, 1),
    "electrical_storage [-1, 1]": (-1, 1),
}
ACTION_PERTURB_MAP = {
    "cooling_device [0, 1]":      "cooling_device",
    "dhw_storage [-1, 1]":        "dhw_storage",
    "electrical_storage [-1, 1]": "electrical_storage",
}
PERTURB_TARGETS = OBS_PERTURB_TARGETS + list(ACTION_PERTURB_TARGETS.keys())

PERTURB_DISPLAY = ([OBS_DISPLAY[o] for o in OBS_PERTURB_TARGETS]
                   + list(ACTION_PERTURB_TARGETS.keys()))

CL_TO_CSV = {
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
CL_TO_LR = CL_TO_CSV
DEFAULT_DATASET = "citylearn_challenge_2023_phase_1"

# CSV columns: clean + perturbed obs + actions
SIM_OBS_CLEAN = [f"{n}_clean" for n in OBS_NAMES]
SIM_OBS_PERT  = [f"{n}_pert"  for n in OBS_NAMES]
SIM_ACT_CLEAN = [f"{a}_clean" for a in ACTION_NAMES]
SIM_ACT_PERT  = [f"{a}_pert"  for a in ACTION_NAMES]
SIM_ACT_DELTA = [f"{a}_delta" for a in ACTION_NAMES]
SIM_META_COLS = ["perturbations"]   

SIM_ALL_COLS = (
    ["step", "policy", "reward"] +
    SIM_OBS_CLEAN + SIM_OBS_PERT +
    SIM_ACT_CLEAN + SIM_ACT_PERT + SIM_ACT_DELTA +
    SIM_META_COLS
)

SIM_PLOT_CHOICES = OBS_NAMES + [f"{n}_clean" for n in OBS_NAMES] + [f"{n}_pert" for n in OBS_NAMES]


# ── Helpers ──────────────────────────────────────────────────────────────────
def log(widget, msg, max_lines=500):
    ts = datetime.now().strftime("%H:%M:%S")
    # The console panel can be disabled (see the navbar layout); fall back to
    # stdout so log calls scattered through the app never crash.
    if widget is None:
        print(f"{ts} {msg}")
        return
    widget.config(state="normal")
    widget.insert("end", f"{ts} {msg}\n")
    # Cap the console history: an unbounded Text widget makes every later
    # insert/scroll progressively slower over a long session.
    n_lines = int(widget.index("end-1c").split(".")[0])  # = content lines + 1
    if n_lines - 1 > max_lines:
        widget.delete("1.0", f"{n_lines - max_lines}.0")
    widget.see("end")
    widget.config(state="disabled")
    # NOTE: no forced update_idletasks() here – Tk repaints the console on the
    # next idle cycle (between loop steps), avoiding a synchronous flush per log.

def btn(parent, text, cmd, bg, fg="white", width=12):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     activebackground=bg, relief="flat", bd=0,
                     padx=10, pady=9, width=width, cursor="hand2",
                     font=("Helvetica", 11, "bold"))

def clamp(target, v):
    v = float(v)
    lo, hi = ACTION_RANGES.get(target, (-1.0, 1.0))
    return max(lo, min(hi, v))

def diff_color(d):
    return GREEN if d < 0.05 else ORANGE if d < 0.20 else RED

def lbl(parent, text, fg="black", bold=False, size=11, anchor="w", **kw):
    return tk.Label(parent, text=text, bg=BG_CARD, fg=fg,
                    font=("Helvetica", size, "bold" if bold else "normal"),
                    anchor="w", **kw)

def section(parent, title):
    f = tk.LabelFrame(parent, text=title, bg=BG_CARD,
                      font=("Helvetica", 10, "bold"), fg="#444")
    f.pack(fill="x", pady=(0, 10))
    return f

def card(parent, title, col, row=0, padx=(0, 6), colspan=1):
    outer = tk.Frame(parent, bg=BG_CARD,
                     highlightbackground="#D0D0D0", highlightthickness=1)
    outer.grid(row=row, column=col, columnspan=colspan,
               sticky="nsew", padx=padx)
    
    # Force the outer frame to distribute its vertical space to its children
    outer.rowconfigure(1, weight=1)
    outer.columnconfigure(0, weight=1)

    if title:
        lbl(outer, title, bold=True, size=13).pack(fill="x", padx=14, pady=(14, 6))
    inner = tk.Frame(outer, bg=BG_CARD)
    inner.pack(fill="both", expand=True, padx=14, pady=(0, 14))
    return outer, inner


# ── Neural Policy ─────────────────────────────────────────────────────────────
class NeuralPolicy:
    def __init__(self):
        self.ready    = False
        self.models   = {}
        self.features = {}
        self.mean: np.ndarray = np.array([])
        self.std:  np.ndarray = np.array([])

    def load(self, directory: str):
        if lr is None:
            return False, "Can't import linear_regressor.py"
        try:
            mean_path = os.path.join(directory, "feature_mean.npy")
            std_path  = os.path.join(directory, "feature_std.npy")
            if not os.path.exists(mean_path) or not os.path.exists(std_path):
                return False, "feature_mean.npy / feature_std.npy not found"
            self.mean = np.load(mean_path)
            self.std  = np.load(std_path)
            self.std[self.std == 0] = 1.0
            models = {}; features = {}
            for target in lr.TARGET_COLS:
                pth_path = os.path.join(directory, f"model_{target}.pth")
                if not os.path.exists(pth_path):
                    return False, f"Missing file: model_{target}.pth"
                domain_feats = lr.DOMAIN_FEATURES[target]
                is_storage   = target in ("dhw_storage", "electrical_storage")
                model = lr.LinearRegressor(input_dim=len(domain_feats),
                                           is_storage=is_storage)
                import torch
                model.load_state_dict(torch.load(pth_path, map_location="cpu"))
                model.eval()
                models[target]   = model
                features[target] = domain_feats
            self.models   = models
            self.features = features
            self.ready    = True
            return True, f"Model loaded from: {os.path.basename(directory)}"
        except Exception as e:
            return False, f"Loading error: {e}"

    def predict(self, obs):
        if not self.ready or lr is None:
            return {a: 0.0 for a in ACTION_NAMES}
        lr_obs  = {v: float(obs.get(k, 0.0)) for k, v in CL_TO_LR.items()}
        f_cols  = list(lr.FEATURE_COLS)
        row     = [lr_obs.get(c, 0.0) for c in f_cols]
        norm    = [(v - float(self.mean[i])) / (float(self.std[i]) or 1.0)
                   for i, v in enumerate(row)]
        out = {}
        for t in lr.TARGET_COLS:
            sel = [norm[f_cols.index(f)] for f in self.features[t]]
            x   = lr.torch.tensor(sel, dtype=lr.torch.float32)
            with lr.torch.no_grad():
                out[t] = clamp(t, self.models[t](x).item())
        return out

# ── RL Policy ──────────────────────────────────────────────────────────────────
class RLPolicy:
    """
    Wrapper for loading and executing a trained Reinforcement Learning (RL) policy.
    Supports Omnisafe / PyTorch TorchScript policy actors.
    """
    def __init__(self):
        self.ready: bool = False
        self.actor = None
        # Typed as Any: CityLearn ships no py.typed stubs, so pyright/pylance
        # can't resolve NormalizedObservationWrapper as an actual type (it
        # shows up as Unknown) — using it in a type expression fails even
        # when the import itself succeeds at runtime. Any tells the checker
        # to stop verifying attribute access on this one on purpose, instead
        # of fighting an unresolvable third-party type.
        self._normalizer: Any = None
        self._action_space = None

    def load(self, model_path: str, env=None) -> tuple[bool, str]:
        """
        Loads the RL TorchScript actor model from disk.
        """
        if not os.path.exists(model_path):
            return False, f"RL model file not found: {model_path}"
        try:
            import torch
            # Carica il modello usando torch.jit.load per TorchScript
            try:
                self.actor = torch.jit.load(model_path, map_location="cpu")
            except Exception:
                self.actor = torch.load(model_path, map_location="cpu")

            if hasattr(self.actor, "eval"):
                self.actor.eval()

            if env is not None:
                if NormalizedObservationWrapper is None:
                    return False, "CityLearn not available: can't build the observation normalizer."
                self._normalizer = NormalizedObservationWrapper(env)
                self._action_space = env.action_space[0]
            
            self.ready = True
            return True, f"RL model loaded from: {os.path.basename(model_path)}"
        
        except Exception as e:
            return False, f"Loading error (RL): {e}"

    def _rescale(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale dell'output della policy da [-1, 1]
        ai limiti fisici reali dell'action space di CityLearn.
        """
        if self._action_space is None:
            raise RuntimeError("RL action space is not loaded.")

        action = np.clip(action, -1.0, 1.0)

        low = self._action_space.low
        high = self._action_space.high

        scaled = low + (action + 1.0) / 2.0 * (high - low)

        return scaled.astype(np.float32)

    # Maps a perturbable observation name to (attribute-holder, array name)
    # on the CityLearn Building object. Covers everything that's a plain
    # stored time-series array; `solar_generation`, `cooling_demand` and
    # `dhw_demand` are DERIVED quantities in CityLearn (not simple stored
    # arrays) and are intentionally left out — perturbing those still works
    # for RBC/regressor (which read obs_dict directly) but is NOT reflected
    # in the RL policy's input, since its normalization always re-reads the
    # live building state (see predict()'s docstring).
    _PERTURB_ATTR_PATHS = {
        "indoor_dry_bulb_temperature": ("energy_simulation", "indoor_dry_bulb_temperature"),
        "indoor_dry_bulb_temperature_cooling_set_point": ("energy_simulation", "indoor_dry_bulb_temperature_cooling_set_point"),
        "outdoor_dry_bulb_temperature": ("weather", "outdoor_dry_bulb_temperature"),
        "outdoor_dry_bulb_temperature_predicted_1": ("weather", "outdoor_dry_bulb_temperature_predicted_1"),
        "electricity_pricing": ("pricing", "electricity_pricing"),
        "carbon_intensity": ("carbon_intensity", "carbon_intensity"),
        "occupant_count": ("energy_simulation", "occupant_count"),
        "electrical_storage_soc": ("electrical_storage", "soc"),
        "dhw_storage_soc": ("dhw_storage", "soc"),
    }

    def _normalized_values_with_overrides(self, overrides: dict) -> list:
        """
        Returns the correctly-normalized observation vector (same order as
        self._normalizer.env.observation_names[0]) reflecting `overrides`
        {obs_name: perturbed_value}.

        Needed because CityLearn's NormalizedObservationWrapper.observation()
        ignores whatever you pass it and always re-reads the live building
        state (see predict()'s docstring) — so the only way to get the
        REAL normalization pipeline to reflect a perturbed value is to
        write it into the building's own array for one instant, read the
        result back out, and immediately restore the true value. The
        restore happens in `finally` so a mid-call exception can never
        leave the live simulation state corrupted.
        """
        assert self._normalizer is not None  # predict() already guarantees this
        building = self._normalizer.env.unwrapped.buildings[0]
        t = building.time_step
        saved: dict[tuple, float] = {}
        try:
            for name, value in overrides.items():
                path = self._PERTURB_ATTR_PATHS.get(name)
                if path is None:
                    continue  # derived quantity — not directly patchable, skip
                holder_name, arr_name = path
                arr = getattr(getattr(building, holder_name), arr_name)
                saved[(holder_name, arr_name)] = arr[t]
                arr[t] = value
            norm = building.observations(normalize=True, periodic_normalization=True)
            return list(norm.values())
        finally:
            for (holder_name, arr_name), original in saved.items():
                getattr(getattr(building, holder_name), arr_name)[t] = original

    def predict(self, obs_dict: dict, overrides: dict | None = None) -> dict[str, float]:
        """
        Predicts actions using the loaded RL policy given the current observation dict.

        Usa la stessa pipeline di rl_agent.py:
            raw obs
            -> normalizzazione
            -> actor
            -> mean/log_std
            -> tanh(mean)
            -> rescale [-1, 1] verso action space CityLearn

        `overrides`: {obs_name: perturbed_value} for the observation-level
        perturbations currently active (from compute()). Needed because
        CityLearn's NormalizedObservationWrapper.observation() ignores the
        raw observation list you pass it and always re-reads the live,
        UNPERTURBED building state instead — so without this, the RL policy
        never sees any obs-level perturbation regardless of what obs_dict
        contains. See _normalized_values_with_overrides()'s docstring.
        """
        if (
            not self.ready
            or self.actor is None
            or self._normalizer is None
            or self._action_space is None
        ):
            raise RuntimeError(
                "RL model, normalizer or action space is not loaded or ready."
            )

        if overrides:
            norm_values = self._normalized_values_with_overrides(overrides)
        else:
            # Stesso ordine delle osservazioni dell'environment
            obs_names = self._normalizer.env.observation_names[0]
            raw_obs_list = [[
                float(obs_dict.get(name, 0.0))
                for name in obs_names
            ]]
            # NormalizedObservationWrapper.observation() ignores raw_obs_list
            # and re-reads the live env — fine here since nothing needs to
            # be overridden for the clean/unperturbed call.
            norm_values = self._normalizer.observation(raw_obs_list)[0]

        # 2. Tensor senza dimensione batch aggiuntiva
        obs_t = torch.as_tensor(
            norm_values,
            dtype=torch.float32
        )

        # 3. Forward dell'attore
        with torch.no_grad():
            output = self.actor(obs_t)

        # 4. Stessa logica di rl_agent.py
        mean, log_std = output.chunk(2, dim=-1)

        # Non influenza l'azione deterministica, ma mantiene
        # la stessa pipeline del wrapper originale
        log_std = torch.clamp(
            log_std,
            min=-20.0,
            max=2.0
        )

        # 5. Output dell'attore in [-1, 1]
        action = torch.tanh(mean).detach().cpu().numpy()

        # 6. Rescaling ai limiti fisici di CityLearn
        action = self._rescale(action)

        # 7. Associa le azioni al loro vero ordine nell'environment
        env_action_names = self._normalizer.env.action_names[0]

        actions = {}

        for i, name in enumerate(env_action_names):
            val = float(action[i])
            actions[name] = clamp(name, val)

        return actions

# ── Redirector ────────────────────────────────────────────────────────────────
class Redirector(io.StringIO):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.buf    = ""

    def write(self, text) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                log(self.widget, line.rstrip())
        return len(text)

    def flush(self):
        if self.buf.strip():
            log(self.widget, self.buf.rstrip())
        self.buf = ""


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.title("Policy Sandbox")
        self.root.minsize(1280, 720)
        self._configure_window_geometry()
        self.root.configure(bg=BG_APP)

        # Action-comparison labels (clean/pert/diff) and observation labels are
        # only populated when their panels are built. They stay empty while
        # those panels are disabled, so the code reading them must tolerate that.
        self.clean_lbl = {}; self.pert_lbl = {}; self.diff_lbl = {}
        self.obs_lbl   = {}
        # Console panel; None until/unless built. log() handles None gracefully.
        self.console   = None

        self.env    = None; self.rbc = None
        self.obs    = None; self.obs_dict = {}
        # Physically-accurate post-dynamics indoor_temp/SOC, kept separate
        # from obs_dict on purpose — see compute()'s docstring.
        self.obs_dict_corrected: dict = {}
        self.neural = NeuralPolicy()
        self.rl     = RLPolicy()
        self.neural_status = tk.Label(self.root) 
        self.rl_status     = tk.Label(self.root)

        self.policy_var = tk.StringVar(value="rbc")
        self.running    = False
        self.stop_req   = False
        self.step_n     = 0
        # id of the pending loop() after-callback, so Reset/Stop can cancel the
        # run loop instead of leaving it stepping a freshly-reset environment.
        self._loop_after_id = None

        # ── Multi-perturbation list: each entry is {"feat": str, "amount": float}
        self.pert_list: list[dict] = []   # active perturbations
        self.pert_feat   = tk.StringVar(value=PERTURB_DISPLAY[0])
        self.pert_amount = tk.DoubleVar(value=0.0)

        # ── Auto cooling-setpoint schedule: splits the episode into thirds
        # (standard / high / low) so the three policies get perturbed at the
        # exact same fraction of the episode without having to click "Add" by
        # hand at the right moment in each separate run.
        self.sp_schedule_var = tk.BooleanVar(value=False)
        self._sp_schedule_phase = None  # 1/2/3, or None before the first check

        # ── Baseline dataset
        self.dataset_df       = None
        self.ext_plot_obs_var = tk.StringVar(value=OBS_DISPLAY[OBS_NAMES[0]])

        # ── Simulation dataset (loaded after session closes)
        self.sim_df = None
        # True when sim_df is a just-closed session already kept in
        # _archived_runs (so the plot doesn't redraw the same curve twice).
        self._sim_df_archived = False

        # ── Archived perturbed runs kept frozen on the plot for comparison.
        # Each entry: {"cols": {step/<obs>_clean/<obs>_pert/reward: [...]},
        #              "color": str, "label": str}
        self._archived_runs: list[dict] = []

        # ── Unified plot panel
        self.unified_plot_canvas  = None
        self.unified_plot_fig     = None
        self.unified_plot_ax      = None
        self.unified_placeholder  = None
        self.unified_plot_col_var = tk.StringVar(value=OBS_DISPLAY[OBS_NAMES[0]])
        self.sim_col_combo        = None

        # ── Session / CSV recording
        self._sim_file    = None   # file handle – open for entire session
        self._sim_writer  = None
        self._sim_path    = ""
        self._session_open = False  # True = session in progress (file open)
        # True once an episode has ended (finished or crashed): the env is no
        # longer steppable, so Start must go through Reset first. A plain pause
        # (Stop) does NOT set this, so it stays resumable.
        self._needs_reset  = False

        # ── Live plot buffer (populated step-by-step during a running session)
        self._live_cols: dict[str, list] = {}
        self.live_plot_var = tk.BooleanVar(value=True)
        # Deviation view: plot each run's (perturbed − clean) instead of the
        # absolute value, so overlapping runs spread out around a flat zero line.
        self.deviation_var = tk.BooleanVar(value=False)

        self.build()
        self.init_env()
        self._auto_load_model()

    # ── Environment ──────────────────────────────────────────────────────────
    def init_env(self):
        if not CITYLEARN_AVAILABLE:
            log(self.console, f"ERROR CityLearn: {CITYLEARN_ERROR}"); return
        assert default_env_config is not None and CityLearnEnv is not None and AdvancedRBC is not None
        try:
            log(self.console, "Initialising CityLearn environment...")
            schema   = default_env_config(DEFAULT_DATASET)  # type: ignore
            self.env = CityLearnEnv(schema=schema, central_agent=True)
            self.rbc = AdvancedRBC(self.env)
            self.store(self.env.reset()[0])
            self.step_n = 0
            self._sp_schedule_phase = None
            log(self.console,
                f"Environment ready - {self.env.time_steps - 1} steps per episode.")
            self.update_obs(); self.refresh()
        except Exception as e:
            log(self.console, f"Init error: {e}")

    def reset_env(self):
        """Reset closes the current session (if any) then resets the env."""
        # Fully stop any in-flight run loop FIRST, otherwise the queued loop()
        # keeps stepping the freshly-reset env (a "ghost" run) and leaves
        # running=True, which makes the next Start a no-op ("Already running").
        self.stop_req = True
        self.running  = False
        self._cancel_loop()
        self._close_session()
        # A fresh episode is steppable again, regardless of how the last one ended.
        self._needs_reset = False
        self.stop_req     = False
        if self.env is None:
            self.init_env(); return
        try:
            self.store(self.env.reset()[0])
            self.step_n = 0
            self._sp_schedule_phase = None
            log(self.console, "Environment reset.")
            self.update_obs(); self.refresh()
        except Exception as e:
            log(self.console, f"Reset error: {e}")

    def store(self, obs, corrected_override=None):
        """corrected_override: optional {name: value} of physically-accurate
        post-dynamics values (indoor_temp/SOC) for CSV logging and the UI
        panel ONLY. self.obs_dict — what actually feeds predict() — always
        comes straight from the raw `obs` CityLearn returned, unmodified."""
        self.obs   = obs
        if self.rbc is None:
            log(self.console, "Error: RBC not initialised.")
            return
        assert self.rbc is not None
        names = self.rbc.observation_names[0]
        self.obs_dict = {n: obs[0][i] for i, n in enumerate(names)}
        self.obs_dict_corrected = dict(self.obs_dict)
        if corrected_override:
            self.obs_dict_corrected.update(corrected_override)

    def update_obs(self):
        for n, w in self.obs_lbl.items():
            if n == "comfort_band":
                val = self.rbc.comfort_band if self.rbc else 0.0
                w.config(text=f"{val:.3f}")
            else:
                w.config(text=f"{self.obs_dict_corrected.get(n, 0.0):.3f}")

    # ── Policy ────────────────────────────────────────────────────────────────
    def _env_action_order(self):
        """Action names in the order the environment expects to receive them.

        This is NOT the same as ACTION_NAMES (the env uses
        ['dhw_storage', 'electrical_storage', 'cooling_device']). Packing
        actions in the wrong order sends each value to the wrong device.
        Falls back to ACTION_NAMES if the controller isn't ready yet.
        """
        if self.rbc is not None:
            try:
                return list(self.rbc.action_names[0])
            except Exception:
                pass  # Fall through to the default return below
            
        return list(ACTION_NAMES)

    def predict(self, obs_dict, obs_overrides=None):
        policy_type = self.policy_var.get()

        if policy_type == "rbc":
            if not self.rbc or self.obs is None:
                return {a: 0.0 for a in ACTION_NAMES}
            vec  = [obs_dict.get(n, 0.0) for n in self.rbc.observation_names[0]]
            raw  = self.rbc.predict([vec])
            acts = raw[0] if raw else [0.0, 0.0, 0.0]
            return {n: clamp(n, acts[i])
                    for i, n in enumerate(self._env_action_order())}
        elif policy_type == "neural":
            return self.neural.predict(obs_dict)
        elif policy_type == "RL":
            return self.rl.predict(obs_dict, overrides=obs_overrides)

        return {a: 0.0 for a in ACTION_NAMES}

    def compute(self):
        """
        Returns (clean_actions, perturbed_actions, clean_obs, perturbed_obs).
        Applies ALL perturbations in self.pert_list.

        Two parallel observation baselines are used on purpose:
          - self.obs_dict (raw, exactly as CityLearn returned it) feeds the
            actual predict() calls below — matching collect_data.py /
            evaluate.py. Feeding the policy the physically-corrected values
            instead changes the distribution the trained LinearRegressor
            models see at rollout time relative to what they were trained
            under, and broke cooling entirely (runaway indoor temperature).
          - self.obs_dict_corrected (physically-accurate post-dynamics
            indoor_temp/SOC) is what gets returned here for CSV logging and
            plotting, so the thesis plots still show the real building
            state rather than CityLearn's pre-dynamics artefact.
        Perturbations are applied identically to both baselines, so the
        logged "perturbed" values still match what the policy actually saw
        (offset from whichever baseline that call used).
        """
        c_obs     = dict(self.obs_dict)
        p_obs     = dict(self.obs_dict)
        c_obs_log = dict(self.obs_dict_corrected)
        p_obs_log = dict(self.obs_dict_corrected)

        # Separate obs perturbations from action perturbations
        action_deltas: dict[str, float] = {}
        # obs-level perturbations, keyed by final (post-perturbation) value —
        # only used by the RL path, whose normalizer otherwise can't see them
        # (see RLPolicy.predict()'s docstring).
        obs_overrides: dict[str, float] = {}
        for p in self.pert_list:
            feat    = p["feat"]
            amount  = p["amount"]
            action_target = ACTION_PERTURB_MAP.get(feat)
            if action_target is not None:
                action_deltas[action_target] = amount
            else:
                if p.get("mode") == "assign":
                    p_obs[feat]     = amount  # Occupant count
                    p_obs_log[feat] = amount
                else:
                    p_obs[feat]     = p_obs.get(feat, 0.0) + amount
                    p_obs_log[feat] = p_obs_log.get(feat, 0.0) + amount
                obs_overrides[feat] = p_obs[feat]

        c = self.predict(c_obs)
        p = self.predict(p_obs, obs_overrides=obs_overrides)

        # Apply action-level perturbations on top of policy output
        for act_name, val in action_deltas.items():
            p[act_name] = clamp(act_name, val)

        return c, p, c_obs_log, p_obs_log

    def refresh(self):
        # Nothing to update when the action-comparison panel is disabled.
        if not self.clean_lbl:
            return
        c, p, _, _ = self.compute()
        for dev in ACTION_NAMES:
            d = abs(p[dev] - c[dev])
            self.clean_lbl[dev].config(text=f"{c[dev]:.3f}")
            self.pert_lbl[dev].config(text=f"{p[dev]:.3f}")
            self.diff_lbl[dev].config(text=f"D {d:.3f}", fg=diff_color(d))

    # ── Session / CSV Logging ─────────────────────────────────────────────────
    def _open_session(self):
        """Opens a new CSV session. No-op if session already open."""
        if self._session_open:
            return
        ts     = datetime.now().strftime("%d%m%Y_%H%M")
        p_var = self.policy_var.get()
        if p_var == "rbc":
            policy = "rbc"
        elif p_var == "RL":
            policy = "rl"
        else:
            policy = "linearregressor"

        sim_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "simulations"))
        os.makedirs(sim_dir, exist_ok=True)
        self._sim_path    = os.path.join(sim_dir, f"sim_{policy}_{ts}.csv")
        self._sim_file    = open(self._sim_path, "w", newline="", encoding="utf-8")
        self._sim_writer  = csv.DictWriter(self._sim_file, fieldnames=SIM_ALL_COLS)
        self._sim_writer.writeheader()
        self._sim_file.flush()
        self._session_open = True
        self._live_cols = {c: [] for c in SIM_ALL_COLS}
        log(self.console, f"Session started -> {self._sim_path}")
        self._update_session_label()

    def _close_session(self):
        """Closes and saves the current CSV session."""
        if not self._session_open:
            return
        if self._sim_file:
            self._sim_file.close()
            self._sim_file   = None
            self._sim_writer = None
        self._session_open = False
        # Freeze this run's perturbed curve on the plot (in its own colour) so
        # the next run can be compared against it. Runs that finish for ANY
        # reason pass through here (Reset, episode end, crash, app close), so
        # the comparison history is built up consistently.
        self._archive_current_run()
        log(self.console, f"Session saved: {self._sim_path}")
        self._load_sim_for_plot(self._sim_path, archived=True)
        self._update_session_label()

    def _archive_current_run(self):
        """Snapshot the just-finished run so its perturbed curve stays on the
        plot. No-op unless the run had perturbations and at least one step."""
        steps = self._live_cols.get("step") if self._live_cols else None
        if not steps or not self.pert_list:
            return
        # Copy only the columns the plot needs, detached from _live_cols so the
        # next session's buffer reset can't mutate this archived snapshot.
        cols = {k: list(v) for k, v in self._live_cols.items()
                if k == "step" or k == "reward" or k == "perturbations"
                or k.endswith("_clean") or k.endswith("_pert")}
        idx   = len(self._archived_runs)
        color = run_color(idx)
        self._archived_runs.append({
            "cols":  cols,
            "color": color,
            "label": f"run {idx + 1}",
        })
        log(self.console,
            f"Run {idx + 1} kept on plot for comparison [{color}].")

    def _update_session_label(self):
        if self._session_open:
            name = os.path.basename(self._sim_path)
            self.session_lbl.config(
                text=f"Session open: {name}", fg=GREEN)
        else:
            self.session_lbl.config(text="No active session.", fg="gray")

    def _write_sim_row(self, reward, c_actions, p_actions, c_obs, p_obs):
        if self._sim_writer is None or self._sim_file is None:
            return
        import json
        row = {
            "step":         self.step_n,
            "policy":       self.policy_var.get(),
            "reward":       round(float(reward), 6),
            "perturbations": json.dumps(self.pert_list),
        }
        for n in OBS_NAMES:
            row[f"{n}_clean"] = round(float(c_obs.get(n, 0.0)), 6)
            row[f"{n}_pert"]  = round(float(p_obs.get(n, 0.0)), 6)
        for a in ACTION_NAMES:
            row[f"{a}_clean"] = round(c_actions[a], 6)
            row[f"{a}_pert"]  = round(p_actions[a], 6)
            row[f"{a}_delta"] = round(abs(p_actions[a] - c_actions[a]), 6)
        self._sim_writer.writerow(row)
        self._sim_file.flush()
        # Mirror the row into the live buffer so the plot can grow step-by-step.
        if self._live_cols:
            for k in SIM_ALL_COLS:
                self._live_cols[k].append(row.get(k))

    def _apply_sp_schedule(self):
        """When the auto cooling-setpoint schedule is enabled, keeps the
        setpoint perturbation in sync with the current third of the episode:
        standard for the first third, +high for the middle third, +low for
        the last third. Runs once per step (before compute()) so every
        policy's run gets perturbed at exactly the same fraction-of-episode
        point, regardless of how each one otherwise plays out — no need to
        babysit 3 separate runs and click "Add" at matching moments."""
        if not self.sp_schedule_var.get() or not self.env:
            return
        try:
            high = float(self.sp_schedule_high_entry.get().replace(",", "."))
            low  = float(self.sp_schedule_low_entry.get().replace(",", "."))
        except ValueError:
            return  # invalid entry: leave whatever's already active alone

        total = max(1, self.env.time_steps - 1)
        third = total / 3.0
        if self.step_n < third:
            phase, amount = 1, None
        elif self.step_n < 2 * third:
            phase, amount = 2, high
        else:
            phase, amount = 3, low

        if self._sp_schedule_phase == phase:
            return  # already applied for this phase, nothing to do
        self._sp_schedule_phase = phase

        self.pert_list = [p for p in self.pert_list if p["feat"] != SP_COL]
        if amount is not None and amount != 0.0:
            self.pert_list.append({"feat": SP_COL, "amount": amount, "mode": "delta"})
            log(self.console,
                f"[auto schedule] step {self.step_n}: cooling setpoint {amount:+.2f}")
        else:
            log(self.console,
                f"[auto schedule] step {self.step_n}: cooling setpoint standard")
        self._rebuild_pert_list_ui()

    # ── Simulation ────────────────────────────────────────────────────────────
    def step(self):
        if not self.env or self.obs is None:
            log(self.console, "Environment not initialised."); return
        self._apply_sp_schedule()
        c, p, c_obs, p_obs = self.compute()
        acts        = p if self.pert_list else c
        # Pack in the ENV's action order, not ACTION_NAMES, or each value lands
        # on the wrong device (e.g. the electrical_storage action driving the
        # cooling_device → invalid cooling → CityLearn aborts the step).
        action_list = [[acts[a] for a in self._env_action_order()]]

        try:
            obs, reward, term, trunc, _ = self.env.step(action_list)
            building = self.env.buildings[0]
            if self.rbc is None:
                log(self.console, "RBC controller not initialized.")

            # Physically-accurate post-dynamics values, for CSV/plot/UI-panel
            # display ONLY (see compute()'s docstring for why `obs` itself
            # must stay untouched: it's what feeds the policy on the NEXT
            # step, and collect_data.py / evaluate.py never patch it either).
            corrected_override = {
                'electrical_storage_soc': building.electrical_storage.soc[building.time_step - 1],
                'dhw_storage_soc': building.dhw_storage.soc[building.time_step - 1],
                'indoor_dry_bulb_temperature': building.indoor_dry_bulb_temperature[building.time_step - 1],
            }

            self.store(obs, corrected_override=corrected_override)
            self.step_n += 1
            p_val = self.policy_var.get()
            ctrl  = "RL" if p_val == "RL" else ("Neural" if p_val == "neural" else "RBC")
            log(self.console, f"Step {self.step_n} [{ctrl}] reward {reward[0]:.4f}")
            for dev in ACTION_NAMES:
                d = abs(p[dev] - c[dev])
                if d > 0.05:
                    log(self.console,
                        f"  ! {dev} clean={c[dev]:.3f} "
                        f"perturbed={p[dev]:.3f} D={d:.3f}")
            self._write_sim_row(reward[0], c, p, c_obs, p_obs)
            self.update_obs(); self.refresh()
            if self.live_plot_var.get():
                self._draw_unified_plot(quiet=True)
            if term or trunc:
                log(self.console, "Episode finished. Press Reset to start a new episode.")
                self.stop_req     = True
                self.running      = False
                self._needs_reset = True
                self._close_session()
        except Exception as e:
            log(self.console, f"Step {self.step_n + 1} crashed: {e}")
            log(self.console, "Environment is in an invalid state. Press Reset "
                              "to start a new episode (Start cannot resume it).")
            self.stop_req     = True
            self.running      = False
            self._needs_reset = True
            self._close_session()

    def start(self):
        if self.running:
            log(self.console, "Already running."); return
        if self._needs_reset:
            log(self.console, "Episode ended/crashed — press Reset before starting again."); return
        if not self.env:
            log(self.console, "Environment not initialised."); return
        if self.policy_var.get() == "neural" and not self.neural.ready:
            log(self.console, "Load the neural policy first."); return
        if self.policy_var.get() == "RL" and not self.rl.ready:
            log(self.console, "Load the RL policy first or verify the model file path.")
            return
        self._cancel_loop()    # drop any stray callback before starting fresh
        self._open_session()   # opens only if not already open (pause→resume)
        self.running  = True
        self.stop_req = False
        log(self.console, "Simulation running...")
        self.loop()

    def loop(self):
        self._loop_after_id = None
        if self.stop_req:
            self.running = False
            return
        self.step()
        # step() may end the episode (term/crash); only keep looping if it
        # didn't, and remember the callback id so Reset/Stop can cancel it.
        if self.stop_req or not self.running:
            return
        self._loop_after_id = self.root.after(400, self.loop)

    def _cancel_loop(self):
        """Cancel a pending run-loop callback, if any."""
        if self._loop_after_id is not None:
            try:
                self.root.after_cancel(self._loop_after_id)
            except Exception:
                pass
            self._loop_after_id = None

    def stop(self):
        """Pause: stops the loop but keeps the session (CSV) open."""
        self.stop_req = True
        self.running  = False
        self._cancel_loop()
        # NOTE: session stays open on Stop – user can resume with Start.
        log(self.console, "Simulation paused. Press Start to resume.")

    # ── Load model ────────────────────────────────────────────────────────────
    _MODEL_SEARCH_DIR = ["../results/linear_regressor"]

    def _auto_load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Try loading Neural / LinearRegressor model
        for rel in self._MODEL_SEARCH_DIR:
            directory = os.path.normpath(os.path.join(script_dir, rel))
            mean_ok = os.path.exists(os.path.join(directory, "feature_mean.npy"))
            std_ok  = os.path.exists(os.path.join(directory, "feature_std.npy"))
            if mean_ok and std_ok:
                ok, msg = self.neural.load(directory)
                log(self.console, f"[Neural] {msg}")
                break

        # 2. Auto-load RL model (file unico sovrascritto)
        rl_path = os.path.normpath(os.path.join(script_dir, "../results/rl_model/actor.pt"))
        if os.path.exists(rl_path):
            ok, msg = self.rl.load(rl_path, env=self.env)
            log(self.console, f"[RL] {msg}")

    def load_model(self):
        directory = filedialog.askdirectory(title="Choose model folder...")
        if not directory: return
        log(self.console, f"Loading neural model from: {directory} ...")
        ok, msg = self.neural.load(directory)
        self.neural_status.config(text=f"Status: {msg}",
                                  fg=GREEN if ok else RED)
        log(self.console, msg)

    # ── Baseline dataset ──────────────────────────────────────────────────────
    def load_dataset(self):
        default_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "datasets"))
        if not os.path.isdir(default_dir):
            default_dir = os.path.dirname(os.path.abspath(__file__))
        path = filedialog.askopenfilename(
            title="Seleziona il dataset CSV (baseline)",
            initialdir=default_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            log(self.console, f"Errore caricamento dataset: {e}"); return

        # The CSV must be a COMPLETE dataset: every observation must be present,
        # either in the RBC/dataset format or in the simulation-export format.
        norm, info = self._normalize_baseline_df(df)
        if norm is None:
            log(self.console, f"Dataset non valido: {info}")
            self.dataset_status.config(
                text=f"Invalid: {os.path.basename(path)}", fg=RED)
            return

        self.dataset_df = norm
        log(self.console,
            f"Dataset caricato: {os.path.basename(path)} "
            f"({len(norm)} righe) [{info}]")
        self.dataset_status.config(
            text=f"Loaded: {os.path.basename(path)} ({info})", fg="#5856D6")
        # No "what to plot" prompt on load — the "Change obs" button (and the
        # Column selector) already pick the observation. Just draw the current one.
        self._draw_unified_plot(quiet=True)

    def _normalize_baseline_df(self, df):
        """Validate that `df` is a *complete* baseline dataset and return
        ``(normalized_df, label)`` whose columns are the internal observation
        names, or ``(None, error_message)`` if it is not usable.

        Two formats are accepted:
          * RBC / dataset export – one column per observation, named via
            ``CL_TO_CSV`` (e.g. ``indoor_temp``) or by its raw CityLearn name.
          * Simulation export – ``<obs>_clean`` / ``<obs>_pert`` for every
            observation (the user picks which series to use as baseline).
        A dataset is "complete" only if *all* observations resolve.
        """
        if len(df) == 0:
            return None, "il file non contiene righe"
        cols = set(df.columns)

        # 1) RBC / dataset format.
        rbc_src, rbc_missing = {}, []
        for n in OBS_NAMES:
            csv_name = CL_TO_CSV.get(n, n)
            if csv_name in cols:
                rbc_src[n] = csv_name
            elif n in cols:
                rbc_src[n] = n
            else:
                rbc_missing.append(n)
        if not rbc_missing:
            out = pd.DataFrame({n: df[s] for n, s in rbc_src.items()})
            if "reward" in cols:
                out["reward"] = df["reward"].values   # keep reward for plotting
            return out, "RBC dataset"

        # 2) Simulation-export format (needs both _clean and _pert for all obs).
        sim_missing = [n for n in OBS_NAMES
                       if f"{n}_clean" not in cols or f"{n}_pert" not in cols]
        if not sim_missing:
            suffix = self._ask_sim_baseline_suffix()
            if suffix is None:
                return None, "caricamento annullato"
            out = pd.DataFrame({n: df[f"{n}_{suffix}"] for n in OBS_NAMES})
            if "reward" in cols:
                out["reward"] = df["reward"].values   # keep reward for plotting
            return out, f"simulation/{suffix}"

        # Neither format is complete – report the closest match's gaps.
        miss = rbc_missing if len(rbc_missing) <= len(sim_missing) else sim_missing
        miss_disp = ", ".join(OBS_DISPLAY.get(n, n) for n in miss)
        return None, f"dataset incompleto, osservazioni mancanti: {miss_disp}"

    def _ask_sim_baseline_suffix(self):
        """Modal dialog used when a simulation CSV is loaded as baseline:
        ask whether to use the clean or the perturbed series. Returns
        ``"clean"`` / ``"pert"`` or ``None`` if the user cancels."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Serie baseline dalla simulazione")
        dlg.geometry("400x170"); dlg.resizable(False, False)
        dlg.configure(bg=BG_CARD); dlg.grab_set()
        tk.Label(dlg,
                 text="Il CSV e una simulazione (clean + perturbata).\n"
                      "Quale serie vuoi usare come baseline?",
                 bg=BG_CARD, font=("Helvetica", 12), justify="left") \
            .pack(padx=20, pady=(20, 14), anchor="w")
        choice = {"v": None}
        def pick(s): choice["v"] = s; dlg.destroy()
        row = tk.Frame(dlg, bg=BG_CARD); row.pack(pady=(0, 10))
        btn(row, "clean",     lambda: pick("clean"), ORANGE, width=11).pack(side="left", padx=6)
        btn(row, "perturbed", lambda: pick("pert"),  RED,    width=11).pack(side="left", padx=6)
        dlg.wait_window()
        return choice["v"]

    def _show_ext_obs_picker(self):
        if self.dataset_df is None: return
        available = []
        for cl_name in OBS_NAMES:
            csv_name = CL_TO_CSV.get(cl_name, cl_name)
            if csv_name in self.dataset_df.columns:
                available.append(cl_name)
            elif cl_name in self.dataset_df.columns:
                available.append(cl_name)
        if not available:
            log(self.console, "No obs column found in dataset."); return
        
        display_choices = [str(OBS_DISPLAY.get(n, n) or n or "") for n in available]

        if "reward" in self.dataset_df.columns:
            display_choices.append("reward")

        dlg = tk.Toplevel(self.root)
        dlg.title("Choose the observation to plot")
        dlg.geometry("420x180"); dlg.resizable(False, False)
        dlg.configure(bg=BG_CARD); dlg.grab_set()
        tk.Label(dlg, text="Osservazione:", bg=BG_CARD,
                 font=("Helvetica", 12)).pack(padx=20, pady=(20, 6), anchor="w")
        obs_var = tk.StringVar(value=display_choices[0])
        ttk.Combobox(dlg, textvariable=obs_var, values=display_choices,
                     state="readonly", font=("Helvetica", 11), width=40) \
            .pack(padx=20, pady=(0, 16))
        def confirm():
            disp = obs_var.get()
            self.ext_plot_obs_var.set(disp)
            self.unified_plot_col_var.set(disp)
            if self.sim_col_combo is not None:
                self.sim_col_combo.set(disp)
            dlg.destroy()
            self._draw_unified_plot()
        btn(dlg, "Plot", confirm, BLUE, width=10).pack(pady=(0, 10))

    # ── Simulation CSV → plot ─────────────────────────────────────────────────
    def _load_sim_for_plot(self, path: str, archived: bool = False):
        try:
            self.sim_df = pd.read_csv(path)
            # A just-closed session (archived=True) is already drawn from
            # _archived_runs; an externally loaded file is not, so its perturbed
            # curve must still be drawn straight from sim_df.
            self._sim_df_archived = archived
            # Keep custom display names; show only obs that actually have data.
            available = [OBS_DISPLAY[n] for n in OBS_NAMES
                         if f"{n}_clean" in self.sim_df.columns
                         or f"{n}_pert" in self.sim_df.columns
                         or n in self.sim_df.columns]
            if "reward" in self.sim_df.columns:
                available.append("reward")
            if self.sim_col_combo is not None and available:
                self.sim_col_combo["values"] = available
            self.sim_status_lbl.config(
                text=f"Loaded: {os.path.basename(path)}", fg="#5856D6")
            log(self.console,
                f"Sim plot ready - {len(self.sim_df)} steps "
                f"in {os.path.basename(path)}")
            self._draw_unified_plot()
        except Exception as e:
            log(self.console, f"Errore lettura sim file: {e}")

    def load_sim_file(self):
        default_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "simulations"))
        if not os.path.isdir(default_dir):
            default_dir = os.path.dirname(os.path.abspath(__file__))
        path = filedialog.askopenfilename(
            title="Seleziona un file di simulazione CSV",
            initialdir=default_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        # An externally loaded file is its own context; drop the in-session run
        # history so the two don't get drawn on top of each other.
        self._archived_runs.clear()
        self._load_sim_for_plot(path)

    # ── Unified plot ──────────────────────────────────────────────────────────
    def _get_sim_series(self, col):
        """Return (steps, values) for the simulation line.

        While a session is running this reads the in-memory live buffer, so the
        curve grows step-by-step over the baseline. Once the session is closed
        it falls back to the loaded sim_df (final, full curve)."""
        if self._session_open and self._live_cols.get("step"):
            if col not in self._live_cols:
                return None, None
            steps = np.asarray(self._live_cols["step"], dtype=float)
            vals  = np.asarray(self._live_cols[col],   dtype=float)
            return steps, vals
        df_s = self.sim_df
        if df_s is not None and col in df_s.columns:
            vals  = df_s[col].to_numpy(dtype=float)
            steps = (df_s["step"].to_numpy()
                     if "step" in df_s.columns else np.arange(len(vals)))
            return steps, vals
        return None, None

    def _perturbations_active(self) -> bool:
        """Whether perturbations apply to the data currently being plotted.

        Live session: the current active perturbation list. Loaded/closed sim:
        the recorded 'perturbations' meta column (non-empty on any step).
        When False, only the clean curve is drawn (clean == perturbed anyway).
        """
        if self._session_open:
            return bool(self.pert_list)
        df = self.sim_df
        if df is not None and "perturbations" in df.columns:
            s_str = df["perturbations"].fillna("").map(str).str.strip()
            return bool((~s_str.isin(["", "[]", "nan", "None", "null"])).any())
        return False

    def _get_setpoint_and_band(self):
        """Returns (steps, setpoint_values, band) for the comfort-band
        shading, or (None, None, None) if no setpoint series is available
        yet. The setpoint is exogenous — it doesn't change with the chosen
        controller — so the baseline dataset is preferred (it's the longest,
        most "complete" series) and the live/sim session is used otherwise."""
        SP = SP_COL
        band = float(getattr(self.rbc, "comfort_band", 2.0)) if self.rbc else 2.0

        df_d = self.dataset_df
        if df_d is not None and SP in df_d.columns:
            sp = df_d[SP].to_numpy(dtype=float)
            return np.arange(len(sp)), sp, band

        steps, sp = self._get_sim_series(f"{SP}_clean")
        if sp is not None and len(sp):
            return steps, sp, band

        return None, None, None

    def _plot_run_sources(self):
        """Yield {'steps', 'cols', 'color', 'label'} for every run currently
        drawn on the unified plot: the live/just-loaded run (if any) and each
        frozen archived run. 'cols' maps column name -> list of raw values,
        same shape whether it comes from the live buffer, a loaded sim_df, or
        an archived snapshot. Centralises what used to be three near-duplicate
        branches (perturbation markers, setpoint delta, comfort band)."""
        if self._session_open:
            cols  = self._live_cols
            steps = cols.get("step")
            if steps:
                yield {"steps": steps, "cols": cols,
                       "color": run_color(len(self._archived_runs)),
                       "label": f"run {len(self._archived_runs) + 1}"}
        elif self.sim_df is not None and not self._sim_df_archived:
            df    = self.sim_df
            steps = (df["step"].tolist() if "step" in df.columns
                     else list(range(len(df))))
            cols  = {c: df[c].tolist() for c in df.columns}
            yield {"steps": steps, "cols": cols, "color": PURPLE, "label": "sim"}
        for run in self._archived_runs:
            steps = run["cols"].get("step")
            if steps:
                yield {"steps": steps, "cols": run["cols"],
                       "color": run["color"], "label": run["label"]}

    @staticmethod
    def _perturbation_change_indices(pert_col):
        """Indices where the *content* of the 'perturbations' meta-column
        changes from the previous step — i.e. every point a perturbation
        started, stopped, or changed amount. Catches every phase boundary of
        a multi-phase schedule (e.g. standard -> +3 -> -3), not just the
        first "off -> on" transition."""
        if not pert_col:
            return []
        idxs = []
        prev = None
        for i, raw in enumerate(pert_col):
            s    = str(raw).strip() if raw is not None else ""
            norm = "" if s in ("[]", "nan", "None", "null", "") else s
            if i > 0 and norm != prev:
                idxs.append(i)
            prev = norm
        return idxs

    def _perturbation_markers(self):
        """(step, color) pairs marking, for every run currently drawn on the
        plot, every timestep at which the active perturbation(s) changed —
        started, stopped, or switched to a different amount."""
        markers = []
        for src in self._plot_run_sources():
            for idx in self._perturbation_change_indices(src["cols"].get("perturbations")):
                markers.append((src["steps"][idx], src["color"]))
        return markers

    @staticmethod
    def _sp_amount_series(pert_col):
        """Per-step cooling-setpoint perturbation amount actually declared in
        the 'perturbations' meta-column (0.0 for steps where the setpoint
        wasn't perturbed, regardless of what other perturbations were active).
        Used to split a run into standard/high/low style phases."""
        if not pert_col:
            return []
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

    @staticmethod
    def _phase_segments(amounts):
        """[(start_idx, end_idx_exclusive, amount), ...] — contiguous runs of
        a constant cooling-setpoint perturbation amount."""
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

    def _setpoint_perturbation_info(self):
        """For every run where the cooling setpoint was actually perturbed:
        the run's colour/label, its perturbed-setpoint series restricted to
        the steps where it was actually perturbed (drawing it for the
        untouched stretch too would just double the ordinary comfort band —
        that's the muddy-grey overlap seen before this fix), and the mean
        indoor-temp vs. active-setpoint tracking delta for EACH distinct
        phase the run went through (standard / +high / -low / ...), instead
        of one pre/post number that silently averages high and low together."""
        infos = []
        for src in self._plot_run_sources():
            cols    = src["cols"]
            amounts = self._sp_amount_series(cols.get("perturbations"))
            if not amounts or all(a == 0.0 for a in amounts):
                continue  # setpoint never perturbed in this run

            sp_clean = cols.get(f"{SP_COL}_clean")
            sp_pert  = cols.get(f"{SP_COL}_pert")
            temp     = cols.get(f"{TEMP_COL}_clean")
            steps    = src["steps"]
            if sp_clean is None or sp_pert is None or temp is None:
                continue
            n = min(len(steps), len(sp_clean), len(sp_pert), len(temp), len(amounts))
            if n == 0:
                continue
            steps_a    = np.asarray(steps[:n],    dtype=float)
            sp_clean_a = np.asarray(sp_clean[:n], dtype=float)
            sp_pert_a  = np.asarray(sp_pert[:n],  dtype=float)
            temp_a     = np.asarray(temp[:n],     dtype=float)
            amounts    = amounts[:n]

            phases = []
            for start, end, amt in self._phase_segments(amounts):
                label  = "standard" if amt == 0.0 else f"{amt:+.2f}\u00b0C"
                sp_seg = sp_clean_a[start:end] if amt == 0.0 else sp_pert_a[start:end]
                phases.append({"amount": amt, "label": label,
                               "delta": float(np.mean(temp_a[start:end] - sp_seg))})

            pert_mask = np.array([a != 0.0 for a in amounts])
            infos.append({
                "color": src["color"], "label": src["label"],
                "steps":   steps_a[pert_mask],
                "sp_pert": sp_pert_a[pert_mask],
                "band": float(getattr(self.rbc, "comfort_band", 2.0)) if self.rbc else 2.0,
                "phases": phases,
            })
        return infos

    def _draw_unified_plot(self, quiet=False):
        disp = self.unified_plot_col_var.get()
        col  = DISPLAY_TO_OBS.get(disp, disp)   # actual observation name

        baseline_series = None
        df_d = self.dataset_df
        if df_d is not None:
            csv_name = CL_TO_CSV.get(col, col)
            col_key  = (csv_name if csv_name in df_d.columns
                        else col if col in df_d.columns
                        else None)
            if col_key is not None:
                baseline_series = df_d[col_key].to_numpy(dtype=float)

        # Simulation curves for the selected observation: clean + perturbed,
        # drawn on top of the baseline so progress is visible step-by-step.
        # The perturbed curve is only shown when perturbations were applied
        # (otherwise it is identical to the clean one). Reward is special: it is
        # a single series (the applied-policy reward), with no clean/pert split.
        is_reward = (col == "reward")
        if is_reward:
            clean_steps, clean_series = self._get_sim_series("reward")
            pert_steps,  pert_series  = None, None
        else:
            clean_steps, clean_series = self._get_sim_series(f"{col}_clean")
            pert_steps,  pert_series  = self._get_sim_series(f"{col}_pert")
        show_pert = (not is_reward) and self._perturbations_active()

        if (baseline_series is None and clean_series is None
                and pert_series is None and not self._archived_runs):
            if not quiet:
                log(self.console, f"No data available for column: '{disp}'.")
            return

        if self.unified_plot_ax is None:
            return

        ax = self.unified_plot_ax
        ax.cla()
        ax.axis("on")   # may have been turned off by the empty-state message

        live   = self._session_open
        suffix = "live" if live else "sim"
        # Deviation view: plot (perturbed − clean) per run instead of absolute
        # values. The clean curve collapses to a flat zero reference, so runs
        # spread out and small effects become readable. Only meaningful for the
        # clean/perturbed obs pairs, never for the single reward series.
        deviation = bool(self.deviation_var.get()) and not is_reward

        # Comfort band shading, mirroring evaluate.py's plot_cooling_device.
        # The setpoint is exogenous (it doesn't depend on which policy is
        # running), so whichever source is available — static baseline or
        # live/sim session — is used as-is; drawn first so the temperature
        # lines sit on top of it.
        if col == "indoor_dry_bulb_temperature" and not deviation:
            band_steps, band_sp, band = self._get_setpoint_and_band()
            if (band_steps is not None
                    and band_sp is not None
                    and band is not None):
                ax.fill_between(band_steps, band_sp + band, band_sp - band,
                                color="green", alpha=0.15,
                                label="Comfort band", zorder=0)
                
                ax.plot(band_steps, band_sp, "--",
                        color="gray", linewidth=1.2,
                        label="Cooling setpoint", zorder=1)

            # Perturbed setpoint + its own comfort band, one per run where the
            # setpoint was actually perturbed, in that run's colour so it's
            # traceable back to the corresponding temperature/action curves.
            for info in self._setpoint_perturbation_info():
                ax.fill_between(info["steps"],
                                 info["sp_pert"] + info["band"],
                                 info["sp_pert"] - info["band"],
                                 color=info["color"], alpha=0.12, zorder=0)
                ax.plot(info["steps"], info["sp_pert"], "--",
                        color=info["color"], linewidth=1.2, alpha=0.9,
                        label=f"{info['label']} setpoint (perturbed)", zorder=1)

        if baseline_series is not None and not deviation:
            ax.plot(np.arange(len(baseline_series)), baseline_series,
                    color=BLUE, linewidth=1.4,
                    label=f"{disp} (baseline)", alpha=0.9)

        if deviation:
            # The clean (unperturbed) curve is the reference: Δ = 0.
            ax.axhline(0.0, color=ORANGE, linewidth=1.2, linestyle=":",
                       alpha=0.9, label="clean (Δ = 0 reference)")
        elif clean_series is not None and clean_steps is not None and len(clean_series) and len(clean_steps):
            clean_label = f"{disp} ({suffix})" if is_reward else f"{disp} (clean)"
            ax.plot(clean_steps, clean_series, color=ORANGE, linewidth=1.4,
                    label=clean_label, linestyle="--", alpha=0.9)
            if live:
                ax.plot(clean_steps[-1], clean_series[-1], "o",
                        color=ORANGE, markersize=5)

        # Already-finished perturbed runs, frozen in their own colours so the
        # current run can be compared against them. In deviation view each run
        # is drawn as (perturbed − clean) of that run.
        if not is_reward:
            pert_key  = f"{col}_pert"
            clean_key = f"{col}_clean"
            for run in self._archived_runs:
                rs = run["cols"].get("step")
                rv = run["cols"].get(pert_key)
                if not rs or not rv:
                    continue
                rsa = np.asarray(rs, dtype=float)
                rva = np.asarray(rv, dtype=float)
                if deviation:
                    rc = run["cols"].get(clean_key)
                    if not rc:
                        continue
                    rca = np.asarray(rc, dtype=float)
                    m   = min(len(rsa), len(rva), len(rca))
                    rsa, rva = rsa[:m], rva[:m] - rca[:m]
                ax.plot(rsa, rva, color=run["color"], linewidth=1.4,
                        linestyle="--", alpha=0.9, label=run["label"])

        # Current perturbed curve. Live: the run in progress, drawn in its own
        # run colour (run #N+1 after N archived runs) so it is distinct from the
        # frozen ones. Not live: only an externally loaded sim file reaches here
        # — a just-closed session is already shown via _archived_runs above.
        draw_current_pert = (show_pert and pert_series is not None
                             and pert_steps is not None
                             and len(pert_series) and len(pert_steps)
                             and (live or not self._sim_df_archived))
        # Deviation needs the paired clean series to subtract; skip if missing.
        if deviation and (clean_series is None or clean_steps is None
                          or not len(clean_series)):
            draw_current_pert = False
        if draw_current_pert:

            assert clean_series is not None

            if live:
                cur_color = run_color(len(self._archived_runs))
                cur_label = f"run {len(self._archived_runs) + 1}"
            else:
                cur_color = PURPLE
                cur_label = f"{disp} (perturbed)"
            cur_steps, cur_vals = np.asarray(pert_steps), np.asarray(pert_series)
            if deviation:
                m = min(len(cur_steps), len(cur_vals), len(clean_series))
                cur_steps = cur_steps[:m]
                cur_vals  = cur_vals[:m] - np.asarray(clean_series)[:m]
            ax.plot(cur_steps, cur_vals, color=cur_color, linewidth=1.4,
                    label=cur_label, linestyle="--", alpha=0.9)
            if live and len(cur_steps):
                ax.plot(cur_steps[-1], cur_vals[-1], "o",
                        color=cur_color, markersize=5)

        # Vertical marker(s): the timestep at which a perturbation first
        # became active, one per run currently on the plot (own run colour).
        # Only the first line carries a legend label, so overlapping runs
        # don't spam the legend with repeated entries.
        first_marker_labeled = False
        for step_val, color in self._perturbation_markers():
            ax.axvline(step_val, color=color, linestyle=":", linewidth=1.6,
                       alpha=0.85, zorder=2,
                       label=None if first_marker_labeled else "perturbation introduced")
            first_marker_labeled = True

        # Cooling-setpoint tracking-error annotation: for every run where the
        # setpoint was perturbed, the mean (indoor temp − setpoint in use)
        # for EACH distinct phase the schedule went through (standard / high
        # / low / ...) — a single pre/post number would silently average a
        # +3 phase and a -3 phase together and could easily land back near
        # the "standard" value by coincidence. Shown regardless of which
        # column is currently selected, stacked top-left, run-coloured.
        y0 = 0.98
        for info in self._setpoint_perturbation_info():
            if not info["phases"]:
                continue
            parts = [f"{ph['label']} Δ{ph['delta']:+.2f}°C" for ph in info["phases"]]
            text = f"{info['label']}: temp−setpoint  " + " · ".join(parts)
            ax.text(0.02, y0, text, transform=ax.transAxes, fontsize=9,
                    color=info["color"], va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=info["color"], alpha=0.85))
            y0 -= 0.055

        if deviation:
            ax.set_title(f"{disp} — deviation (perturbed − clean)",
                         fontsize=10, pad=10)
            ax.set_ylabel("Δ vs clean", fontsize=12, labelpad=6)
        else:
            ax.set_title(disp, fontsize=10, pad=10)
            ax.set_ylabel("")
        ax.set_xlabel("timestep", fontsize=12, labelpad=6)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_facecolor("#FAFAFA")
        for s in ax.spines.values():
            s.set_edgecolor("#DDDDDD")

        # draw_idle() coalesces redraws and lets Tk render on the next idle
        # cycle instead of blocking the event loop on every live step.
        if self.unified_plot_canvas is not None:
            self.unified_plot_canvas.draw_idle()

        if not quiet:
            parts = []
            if baseline_series is not None and not deviation: parts.append("baseline")
            if clean_series is not None:
                parts.append(suffix if is_reward else f"{suffix}_clean")
            if draw_current_pert: parts.append(f"{suffix}_pert")
            if not is_reward and self._archived_runs:
                parts.append(f"{len(self._archived_runs)} archived run(s)")
            mode = " (deviation)" if deviation else ""
            log(self.console, f"Plot aggiornato: {disp}{mode} [{', '.join(parts)}]")

    # ── Multi-perturbation management ─────────────────────────────────────────
    def _add_perturbation(self):
        # Combobox shows short display labels; map back to the internal name
        # used by the perturbation logic (action labels pass through unchanged).
        feat = DISPLAY_TO_OBS.get(self.pert_feat.get(), self.pert_feat.get())
        try:
            amount = float(self.pert_entry.get().replace(",", "."))
        except ValueError:
            log(self.console, "Invalid perturbation amount."); return
        is_assign = feat in ACTION_PERTURB_MAP or feat == "occupant_count"
        if amount == 0.0 and not is_assign:
            log(self.console, "Amount is 0 – nothing added."); return

        # For features in "assign" mode, replace instead of append
        if is_assign:
            self.pert_list = [p for p in self.pert_list if p["feat"] != feat]

        mode = "assign" if is_assign else "delta"
        self.pert_list.append({"feat": feat, "amount": amount, "mode": mode})
        log(self.console, f"Perturbation added: {feat} {'=' if mode == 'assign' else '+'}{amount:.4f}")
        self._rebuild_pert_list_ui()
        self.refresh()

    def _remove_perturbation(self, idx: int):
        if 0 <= idx < len(self.pert_list):
            removed = self.pert_list.pop(idx)
            log(self.console,
                f"Perturbation removed: {removed['feat']} {removed['amount']:+.4f}")
            self._rebuild_pert_list_ui()
            self.refresh()

    def _clear_all_perturbations(self):
        self.pert_list.clear()
        log(self.console, "All perturbations cleared.")
        self._rebuild_pert_list_ui()
        self.refresh()

    def _clear_archived_runs(self):
        """Drop the frozen run history so the plot starts fresh."""
        if not self._archived_runs:
            log(self.console, "No archived runs to clear."); return
        n = len(self._archived_runs)
        self._archived_runs.clear()
        log(self.console, f"Cleared {n} archived run(s) from the plot.")
        self._draw_unified_plot(quiet=True)

    def _rebuild_pert_list_ui(self):
        """Refreshes the scrollable list of active perturbations."""
        for w in self.pert_list_frame.winfo_children():
            w.destroy()
        if not self.pert_list:
            tk.Label(self.pert_list_frame,
                     text="No active perturbations.",
                     bg=BG_CARD, fg="gray",
                     font=("Helvetica", 10)).pack(anchor="w", padx=4, pady=2)
            self.pert_count_lbl.config(text="0 active", fg="gray")
            return
        for i, p in enumerate(self.pert_list):
            row_f = tk.Frame(self.pert_list_frame, bg=BG_CARD)
            row_f.pack(fill="x", pady=1)
            color = ORANGE if p["amount"] > 0 else BLUE
            mode = p.get("mode", "delta")
            sign = "=" if mode == "assign" else ("+" if p["amount"] >= 0 else "")
            tk.Label(row_f,
                    text=f"{OBS_DISPLAY.get(p['feat'], p['feat'])}  {sign}{p['amount']:.4f}",
                    bg=BG_CARD, fg=color,
                    font=("Helvetica", 10)).pack(side="left", padx=(4, 8))
            idx = i  # capture for lambda
            tk.Button(row_f, text="Remove", bg=BG_CARD, fg=RED,
                      relief="flat", bd=0, cursor="hand2",
                      font=("Helvetica", 10, "bold"),
                      command=lambda i=idx: self._remove_perturbation(i)) \
                .pack(side="right", padx=4)
        n = len(self.pert_list)
        self.pert_count_lbl.config(
            text=f"{n} active", fg=ORANGE if n else "gray")

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _configure_window_geometry(self):
        """Open maximised on whatever screen the app runs on, instead of a
        hardcoded 1920x1080. A fixed resolution looks fine on the machine it
        was tuned on, but on any other screen (different resolution, or a
        different Windows display-scaling %) it can open off-screen, oversized,
        or cut off by the taskbar. 'Zoomed' adapts to the real screen/work
        area; the geometry fallback below only kicks in on platforms where the
        window manager doesn't support a zoomed state."""
        try:
            self.root.state("zoomed")          # Windows, and some Linux WMs
            return
        except tk.TclError:
            pass
        try:
            self.root.attributes("-zoomed", True)   # X11 (Linux)
            return
        except tk.TclError:
            pass
        # Last-resort fallback: size to most of the screen, centred.
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = int(sw * 0.9), int(sh * 0.9)
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def build(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        body = tk.Frame(self.root, bg=BG_APP)
        body.grid(row=0, column=0, sticky="nsew", padx=12, pady=(0, 12))
        # Layout: a horizontal navbar on top (row 0, natural height), and below
        # it a compact side panel (current obs + action comparison) on the left
        # with the unified plot filling the rest.
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=0, minsize=300)   # compact side panel
        body.columnconfigure(1, weight=1)   # plot fills the rest

        self.build_navbar(body)
        self.build_side_panel(body)
        # ── Console panel disabled for this layout (kept for later — see method) ─
        # self.build_right(body)        # Console
        self.build_unified_plot(body)

    def build_navbar(self, parent):
        """Top navbar: the former left-column sections (Controller, Dataset,
        Perturbations, Simulation) laid out horizontally across the full width
        instead of stacked vertically on the left."""
        _, inner = card(parent, "", col=0, row=0, padx=(0, 0), colspan=2)

        # Four equal-width groups spread across the whole width. uniform="nav"
        # keeps them the same size so nothing stays bunched up on the left.
        for c in range(4):
            inner.columnconfigure(c, weight=1, uniform="nav")

        def group(title, col, padx):
            g = tk.LabelFrame(inner, text=title, bg=BG_CARD,
                              font=("Helvetica", 10, "bold"), fg="#444")
            g.grid(row=0, column=col, sticky="nsew", padx=padx)
            return g

        # ── Controller ────────────────────────────────────────────────────────
        ctrl = group("Controller", 0, (0, 8))
        for text, val in [("RBC (AdvancedRBC)", "rbc"),
                          ("Neural policy (LinearRegressor)", "neural"),
                          ("Reinforcement Learning Policy", "RL")]:
            tk.Radiobutton(ctrl, text=text, variable=self.policy_var,
                           value=val, command=self.refresh, bg=BG_CARD,
                           font=("Helvetica", 10)) \
                .pack(anchor="w", padx=8, pady=(6, 0))

        # ── Dataset (baseline) ────────────────────────────────────────────────
        ds = group("Dataset (baseline)", 1, 8)
        row = tk.Frame(ds, bg=BG_CARD)
        row.pack(fill="x", padx=8, pady=(10, 6))
        btn(row, "Load dataset", self.load_dataset, "#5856D6", width=12) \
            .pack(side="left")
        btn(row, "Change obs", self._show_ext_obs_picker, GRAY,
            fg="black", width=12).pack(side="left", padx=(8, 0))
        self.dataset_status = lbl(ds, "No dataset loaded", fg="gray")
        self.dataset_status.pack(fill="x", padx=8, pady=(4, 10))

        # ── Perturbations ─────────────────────────────────────────────────────
        pb = group("Perturbations", 2, 8)
        sel = tk.Frame(pb, bg=BG_CARD)
        sel.pack(fill="x", padx=8, pady=(10, 6))
        ttk.Combobox(sel, textvariable=self.pert_feat,
                     values=PERTURB_DISPLAY, state="readonly",
                     font=("Helvetica", 10), width=16) \
            .pack(side="left", fill="x", expand=True)
        lbl(sel, "Amount:", size=10).pack(side="left", padx=(8, 4))
        self.pert_entry = tk.Entry(sel, font=("Helvetica", 11), width=7,
                                   relief="solid", bd=1, justify="center")
        self.pert_entry.insert(0, "0.00")
        self.pert_entry.pack(side="left", ipady=3)

        br = tk.Frame(pb, bg=BG_CARD)
        br.pack(fill="x", padx=8, pady=(0, 6))
        btn(br, "Add", self._add_perturbation, ORANGE, width=9) \
            .pack(side="left", padx=(0, 6))
        btn(br, "Clear all", self._clear_all_perturbations,
            GRAY, fg="black", width=9).pack(side="left")

        self.pert_count_lbl = lbl(pb, "0 active", fg="gray", size=10)
        self.pert_count_lbl.pack(fill="x", padx=8, pady=(0, 4))

        list_outer = tk.Frame(pb, bg="#F4F4F4",
                              highlightbackground="#DDDDDD", highlightthickness=1)
        list_outer.pack(fill="x", padx=8, pady=(0, 10))
        self.pert_list_frame = tk.Frame(list_outer, bg=BG_CARD)
        self.pert_list_frame.pack(fill="x", padx=2, pady=2)
        self._rebuild_pert_list_ui()

        # ── Auto cooling-setpoint schedule ───────────────────────────────────
        # Splits the episode into thirds so every policy is perturbed at the
        # same fraction of the episode, without manually timing the "Add"
        # click the same way across 3 separate runs.
        sched = tk.Frame(pb, bg=BG_CARD,
                         highlightbackground="#DDDDDD", highlightthickness=1)
        sched.pack(fill="x", padx=8, pady=(0, 10))
        tk.Checkbutton(sched, text="Auto cooling-setpoint schedule (thirds)",
                       variable=self.sp_schedule_var, bg=BG_CARD,
                       font=("Helvetica", 10, "bold")) \
            .pack(anchor="w", padx=6, pady=(6, 2))
        lbl(sched, "1st third standard  ·  2nd third high  ·  3rd third low",
            fg="gray", size=9).pack(anchor="w", padx=6)
        sched_row = tk.Frame(sched, bg=BG_CARD)
        sched_row.pack(fill="x", padx=6, pady=(4, 8))
        lbl(sched_row, "High (2nd 3rd):", size=9).pack(side="left")
        self.sp_schedule_high_entry = tk.Entry(
            sched_row, font=("Helvetica", 11), width=6,
            relief="solid", bd=1, justify="center")
        self.sp_schedule_high_entry.insert(0, "3.00")
        self.sp_schedule_high_entry.pack(side="left", padx=(4, 12))
        lbl(sched_row, "Low (3rd 3rd):", size=9).pack(side="left")
        self.sp_schedule_low_entry = tk.Entry(
            sched_row, font=("Helvetica", 11), width=6,
            relief="solid", bd=1, justify="center")
        self.sp_schedule_low_entry.insert(0, "-3.00")
        self.sp_schedule_low_entry.pack(side="left", padx=(4, 0))

        # ── Simulation ────────────────────────────────────────────────────────
        sim = group("Simulation", 3, (8, 0))
        self.session_lbl = lbl(sim, "No active session.", fg="gray", size=10)
        self.session_lbl.pack(fill="x", padx=8, pady=(10, 6))
        r1 = tk.Frame(sim, bg=BG_CARD)
        r1.pack(fill="x", padx=8, pady=(0, 6))
        btn(r1, "Start", self.start, BLUE, width=9).pack(side="left", padx=(0, 6))
        btn(r1, "Stop",  self.stop,  RED,  width=9).pack(side="left")
        r2 = tk.Frame(sim, bg=BG_CARD)
        r2.pack(fill="x", padx=8, pady=(0, 10))
        btn(r2, "Step",  self.step,      GREEN, width=9).pack(side="left", padx=(0, 6))
        btn(r2, "Reset", self.reset_env, GRAY, fg="black", width=9).pack(side="left")

    # Short device names for the compact action-comparison table.
    _DEV_SHORT = {"cooling_device": "cool", "dhw_storage": "dhw",
                  "electrical_storage": "elec"}

    def build_side_panel(self, parent):
        """Compact left panel in the plot area: current observations plus the
        action comparison (devices), in small form."""
        outer, inner = card(parent, "Current state", col=0, row=1, padx=(0, 8))
        outer.grid(pady=(8, 0), sticky="nsew")

        # ── Observations (small list) ─────────────────────────────────────────
        lbl(inner, "Observations", bold=True, size=12).pack(fill="x", pady=(0, 4))
        obs_f = tk.Frame(inner, bg=BG_CARD)
        obs_f.pack(fill="x")
        obs_f.columnconfigure(1, weight=1)
        for i, name in enumerate(OBS_NAMES_DISPLAY):
            fg_name = "#5856D6" if name == "comfort_band" else "gray"
            # Show the short display name to keep the panel narrow, but key the
            # widget by the full obs name so update_obs() can find it.
            lbl(obs_f, OBS_DISPLAY.get(name, name), fg=fg_name, size=11) \
                .grid(row=i, column=0, sticky="w", padx=(0, 12), pady=3)
            w = lbl(obs_f, "---", bold=True, size=11, anchor="e")
            w.grid(row=i, column=1, sticky="e", pady=3)
            self.obs_lbl[name] = w

        # ── Action comparison (devices), compact ──────────────────────────────
        lbl(inner, "Action comparison", bold=True, size=12).pack(fill="x", pady=(14, 4))
        tbl = tk.Frame(inner, bg=BG_CARD)
        tbl.pack(fill="x")
        for c in range(4):
            tbl.columnconfigure(c, weight=1)
        for col, (h, fg) in enumerate([("dev", "gray"), ("clean", BLUE),
                                       ("pert", ORANGE), ("diff", "gray")]):
            tk.Label(tbl, text=h, bg=BG_CARD, fg=fg,
                     font=("Helvetica", 11, "bold"), anchor="center") \
                .grid(row=0, column=col, sticky="ew", padx=2, pady=(0, 4))
        for r, dev in enumerate(ACTION_NAMES, start=1):
            lbl(tbl, self._DEV_SHORT.get(dev, dev), bold=True, size=11) \
                .grid(row=r, column=0, sticky="w", padx=2, pady=2)
            cl = tk.Label(tbl, text="0.000", bg=BLUE_LIGHT, fg=BLUE,
                          font=("Helvetica", 13, "bold"), anchor="center", pady=6)
            cl.grid(row=r, column=1, sticky="ew", padx=2, pady=2)
            self.clean_lbl[dev] = cl
            pl = tk.Label(tbl, text="0.000", bg=ORANGE_LIGHT, fg=ORANGE,
                          font=("Helvetica", 13, "bold"), anchor="center", pady=6)
            pl.grid(row=r, column=2, sticky="ew", padx=2, pady=2)
            self.pert_lbl[dev] = pl
            dl = tk.Label(tbl, text="D 0.000", bg=BG_CARD, fg=GREEN,
                          font=("Helvetica", 11, "bold"), anchor="center", pady=6)
            dl.grid(row=r, column=3, sticky="ew", padx=2, pady=2)
            self.diff_lbl[dev] = dl

        # Sensitivity legend (compact).
        leg = tk.Frame(inner, bg=BG_CARD)
        leg.pack(fill="x", pady=(10, 0))
        for color, text in [(GREEN, "<0.05 robust"),
                            (ORANGE, "0.05-0.20 sens."),
                            (RED, ">0.20 fragile")]:
            lbl(leg, text, fg=color, size=10).pack(anchor="w", pady=1)

    # ── DISABLED in the navbar layout — kept for later (see build()) ──────────
    # "Action comparison" panel: clean vs perturbed action table + legend.
    # def build_center(self, parent):
    #     _, inner = card(parent, "Action comparison", col=1, row=0, padx=(6, 6))
    #     tbl = tk.Frame(inner, bg=BG_CARD)
    #     tbl.pack(fill="both", expand=True)
    #     for c in range(4): tbl.columnconfigure(c, weight=1)
    #
    #     for col, (h, bg, fg) in enumerate([
    #         ("Device",          BG_CARD,      "gray"),
    #         ("Clean input",     BLUE_LIGHT,   BLUE),
    #         ("Perturbed input", ORANGE_LIGHT, ORANGE),
    #         ("Difference",      BG_CARD,      "gray"),
    #     ]):
    #         tk.Label(tbl, text=h, bg=bg, fg=fg,
    #                  font=("Helvetica", 12, "bold"),
    #                  anchor="center", pady=10) \
    #             .grid(row=0, column=col, sticky="ew", padx=(0, 4), pady=(0, 8))
    #
    #     for row, dev in enumerate(ACTION_NAMES, start=1):
    #         lbl(tbl, dev, bold=True, size=12, padx=10) \
    #             .grid(row=row, column=0, sticky="ew",
    #                   padx=(0, 4), pady=(0, 4), ipady=5)
    #         cl = tk.Label(tbl, text="0.000", bg=BLUE_LIGHT, fg=BLUE,
    #                       font=("Helvetica", 22, "bold"),
    #                       anchor="center", pady=20)
    #         cl.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=(0, 8))
    #         self.clean_lbl[dev] = cl
    #         pl = tk.Label(tbl, text="0.000", bg=ORANGE_LIGHT, fg=ORANGE,
    #                       font=("Helvetica", 22, "bold"),
    #                       anchor="center", pady=20)
    #         pl.grid(row=row, column=2, sticky="ew", padx=(0, 4), pady=(0, 8))
    #         self.pert_lbl[dev] = pl
    #         dl = tk.Label(tbl, text="D 0.000", bg=BG_CARD, fg=GREEN,
    #                       font=("Helvetica", 16, "bold"),
    #                       anchor="center", pady=20)
    #         dl.grid(row=row, column=3, sticky="ew", padx=(0, 4), pady=(0, 8))
    #         self.diff_lbl[dev] = dl
    #
    #     leg = tk.Frame(inner, bg=BG_CARD)
    #     leg.pack(fill="x", pady=(12, 0))
    #     for color, label, desc in [
    #         (GREEN,  "D < 0.05",    "robust"),
    #         (ORANGE, "D 0.05-0.20", "sensitive"),
    #         (RED,    "D > 0.20",    "fragile"),
    #     ]:
    #         lbl(leg, label, fg=color, bold=True, size=11) \
    #             .pack(side="left", padx=(0, 4))
    #         lbl(leg, desc, fg="gray", size=11) \
    #             .pack(side="left", padx=(0, 20))

    # ── DISABLED in the navbar layout — kept for later (see build()) ──────────
    # "Current observations" panel + "Console" text widget.
    # def build_right(self, parent):
    #     outer, inner = card(parent, "Current observations", col=2, row=0, padx=(0, 0))
    #     outer.rowconfigure(1, weight=1)
    #     inner.rowconfigure(1, weight=1)
    #     inner.columnconfigure(0, weight=1)
    #
    #     obs_f = tk.Frame(inner, bg=BG_CARD)
    #     obs_f.pack(fill="x")
    #     obs_f.columnconfigure(1, weight=1)
    #     for i, name in enumerate(OBS_NAMES_DISPLAY):
    #         fg_name = "#5856D6" if name == "comfort_band" else "gray"
    #         lbl(obs_f, name, fg=fg_name, size=11) \
    #             .grid(row=i, column=0, sticky="w", padx=(0, 12), pady=2)
    #         w = lbl(obs_f, "---", bold=True, size=11, anchor="e")
    #         w.grid(row=i, column=1, sticky="e", pady=2)
    #         self.obs_lbl[name] = w
    #
    #     lbl(inner, "Console", bold=True, size=12) \
    #         .pack(fill="x", pady=(12, 4))
    #     self.console = tk.Text(inner, wrap="word", state="disabled",
    #                            font=("Courier", 11), relief="flat", bd=0,
    #                            bg="#F8F8F8", padx=8, pady=8, height=10)
    #     self.console.pack(fill="both", expand=True)
    #     log(self.console, "GUI ready.")
    #     log(self.console, "Load the neural policy before switching to it.")
    #     log(self.console, "Add perturbations and press Start to begin a session.")

    def build_unified_plot(self, parent):
        """Unified plot: baseline (blue) + sim (orange)."""
        outer, inner = card(parent,"Dataset plot (baseline vs simulation)", col=1, row=1, padx=(0, 0))
        outer.grid(pady=(8, 0), sticky="nsew")
        self.unified_plot_inner = inner

        ctrl_bar = tk.Frame(inner, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 6))

        lbl(ctrl_bar, "Column:", fg="gray", size=11) \
            .pack(side="left", padx=(0, 6))

        self.sim_col_combo = ttk.Combobox(
            ctrl_bar, textvariable=self.unified_plot_col_var,
            values=PLOT_CHOICES, state="readonly",
            font=("Helvetica", 11), width=32)
        self.sim_col_combo.pack(side="left", padx=(0, 10))
        # Defer the redraw to the next idle tick so the combobox closes
        # immediately instead of blocking on the (heavier) plot rebuild.
        self.sim_col_combo.bind("<<ComboboxSelected>>",
                                lambda e: self.root.after_idle(self._draw_unified_plot))

        btn(ctrl_bar, "Plot", self._draw_unified_plot, BLUE, width=8) \
            .pack(side="left", padx=(0, 10))
        btn(ctrl_bar, "Load sim file", self.load_sim_file,
            "#5856D6", width=14).pack(side="left", padx=(0, 10))
        btn(ctrl_bar, "Clear runs", self._clear_archived_runs,
            "#8E8E93", width=11).pack(side="left", padx=(0, 10))

        tk.Checkbutton(ctrl_bar, text="Live plot", variable=self.live_plot_var,
                       bg=BG_CARD, activebackground=BG_CARD,
                       font=("Helvetica", 10)) \
            .pack(side="left", padx=(0, 10))

        tk.Checkbutton(ctrl_bar, text="Deviation (perturbed - clean)",
                       variable=self.deviation_var,
                       command=lambda: self._draw_unified_plot(),
                       bg=BG_CARD, activebackground=BG_CARD,
                       font=("Helvetica", 10)) \
            .pack(side="left", padx=(0, 10))

        self.sim_status_lbl = lbl(ctrl_bar, "No simulation loaded yet.",
                                  fg="gray", size=10)
        self.sim_status_lbl.pack(side="left")

        # Create the figure + the ONE embedded canvas now (eagerly) and pack it,
        # instead of lazily on the first data draw. Lazy creation + the window
        # resizing afterwards is what left a stale "ghost" strip of an earlier
        # render on some platforms (Windows/HiDPI TkAgg).
        self.unified_plot_fig = Figure(figsize=(12, 7), facecolor=BG_CARD,
                                       layout="tight")
        self.unified_plot_ax  = self.unified_plot_fig.add_subplot(111)
        self.unified_plot_canvas = FigureCanvasTkAgg(
            self.unified_plot_fig, master=inner)
        cw = self.unified_plot_canvas.get_tk_widget()
        cw.configure(bg=BG_CARD, highlightthickness=0)
        cw.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        # Repaint fully (not just blit) whenever the canvas is resized, so no
        # stale render of a previous size lingers underneath. add="+" is
        # ESSENTIAL: it appends to FigureCanvasTkAgg's own <Configure> handler
        # (which resizes the figure to fill the widget) instead of replacing it
        # — replacing it would freeze the figure at its small native size.
        self._plot_resize_after = None
        cw.bind("<Configure>", self._on_plot_resize, add="+")
        self._show_plot_empty_state()
        # Once the window has settled at its final size, force one clean repaint
        # so the very first render already matches the real canvas size.
        self.root.after(300, self._redraw_plot_canvas)

    def _show_plot_empty_state(self):
        ax = self.unified_plot_ax
        if ax is None:
            return
        ax.cla()
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Load a dataset CSV (baseline) and/or a simulation CSV "
                "to display the plot here.",
                ha="center", va="center", color="gray", fontsize=11,
                transform=ax.transAxes)
        if self.unified_plot_canvas is not None:
            self.unified_plot_canvas.draw_idle()

    def _on_plot_resize(self, event):
        # Debounce a burst of resize events into a single full redraw.
        if self._plot_resize_after is not None:
            try:
                self.root.after_cancel(self._plot_resize_after)
            except Exception:
                pass
        self._plot_resize_after = self.root.after(60, self._redraw_plot_canvas)

    def _redraw_plot_canvas(self):
        self._plot_resize_after = None
        if self.unified_plot_canvas is not None:
            # Full draw() clears the backing store, removing any ghost strip.
            self.unified_plot_canvas.draw()
        
    def _on_close(self):
        if self.unified_plot_fig is not None:
            plt.close(self.unified_plot_fig)
        self._close_session()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

def _apply_real_dpi_scaling(root):
    """Make Tk's scaling follow the screen's *actual* DPI instead of a
    hardcoded value. `tk scaling 1.0` means "1 point = 1 physical pixel",
    which is only correct on an old 72dpi screen — on a normal 96dpi monitor
    it makes every font/widget render ~25% smaller than intended, and on a
    HiDPI/scaled Windows display (e.g. 125-200%) it can look tiny, which is
    almost certainly what happened on your colleague's machine. Computing the
    real DPI and feeding it to Tk keeps text/widgets a consistent, readable
    size on any screen."""
    try:
        dpi = root.winfo_fpixels("1i")   # real pixels per inch on this screen
        root.tk.call("tk", "scaling", dpi / 72.0)
    except tk.TclError:
        pass


if __name__ == "__main__":
    root = tk.Tk()
    _apply_real_dpi_scaling(root)
    App(root).mainloop()