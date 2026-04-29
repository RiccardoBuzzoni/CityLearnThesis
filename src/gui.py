import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, csv, tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime

try:
    import linear_regressor as lr
except Exception:
    lr = None

try:
    from main_rbc import AdvancedRBC
    from CityLearn.citylearn.citylearn import CityLearnEnv
    from utils import default_env_config, Config
    CITYLEARN_AVAILABLE = True
except Exception as e:
    CITYLEARN_AVAILABLE = False
    CITYLEARN_ERROR = str(e)
    AdvancedRBC = CityLearnEnv = default_env_config = None

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# Colours
BG_APP       = "#ECECEC"
BG_CARD      = "#FFFFFF"
BLUE         = "#007AFF"
BLUE_LIGHT   = "#E8F2FF"
RED          = "#FF3B30"
GREEN        = "#34C759"
ORANGE       = "#FF9500"
ORANGE_LIGHT = "#FFF4E0"
GRAY         = "#E5E5EA"

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
]
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

CL_TO_CSV = {
    "indoor_dry_bulb_temperature":               "indoor_temp",
    "indoor_dry_bulb_temperature_cooling_set_point": "cooling_set_point",
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
SIM_META_COLS = ["perturbations"]   # JSON-encoded list of {feat, amount}

SIM_ALL_COLS = (
    ["step", "policy", "reward"] +
    SIM_OBS_CLEAN + SIM_OBS_PERT +
    SIM_ACT_CLEAN + SIM_ACT_PERT + SIM_ACT_DELTA +
    SIM_META_COLS
)

SIM_PLOT_CHOICES = OBS_NAMES + [f"{n}_clean" for n in OBS_NAMES] + [f"{n}_pert" for n in OBS_NAMES]


# ── Helpers ──────────────────────────────────────────────────────────────────
def log(widget, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    widget.config(state="normal")
    widget.insert("end", f"{ts} {msg}\n")
    widget.see("end")
    widget.config(state="disabled")
    widget.update_idletasks()

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
        self.root.title("RBC vs Neural Policy")
        self.root.geometry("1900x1200")
        self.root.minsize(1600, 950)
        self.root.configure(bg=BG_APP)

        self.clean_lbl = {}; self.pert_lbl = {}; self.diff_lbl = {}
        self.obs_lbl   = {}

        self.env    = None; self.rbc = None
        self.obs    = None; self.obs_dict = {}
        self.neural = NeuralPolicy()

        self.policy_var = tk.StringVar(value="rbc")
        self.running    = False
        self.stop_req   = False
        self.step_n     = 0

        # ── Multi-perturbation list: each entry is {"feat": str, "amount": float}
        self.pert_list: list[dict] = []   # active perturbations
        self.pert_feat   = tk.StringVar(value=PERTURB_TARGETS[0])
        self.pert_amount = tk.DoubleVar(value=0.0)

        # ── Baseline dataset
        self.dataset_df       = None
        self.ext_plot_obs_var = tk.StringVar(value=OBS_NAMES[0])

        # ── Simulation dataset (loaded after session closes)
        self.sim_df = None

        # ── Unified plot panel
        self.unified_plot_canvas  = None
        self.unified_plot_fig     = None
        self.unified_plot_ax      = None
        self.unified_placeholder  = None
        self.unified_plot_col_var = tk.StringVar(value=OBS_NAMES[0])
        self.sim_col_combo        = None

        # ── Session / CSV recording
        self._sim_file    = None   # file handle – open for entire session
        self._sim_writer  = None
        self._sim_path    = ""
        self._session_open = False  # True = session in progress (file open)

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
            log(self.console,
                f"Environment ready - {self.env.time_steps - 1} steps per episode.")
            self.update_obs(); self.refresh()
        except Exception as e:
            log(self.console, f"Init error: {e}")

    def reset_env(self):
        """Reset closes the current session (if any) then resets the env."""
        self._close_session()
        if self.env is None:
            self.init_env(); return
        try:
            self.store(self.env.reset()[0])
            self.step_n = 0
            log(self.console, "Environment reset.")
            self.update_obs(); self.refresh()
        except Exception as e:
            log(self.console, f"Reset error: {e}")

    def store(self, obs):
        self.obs   = obs
        assert self.rbc is not None
        names = self.rbc.observation_names[0]
        self.obs_dict = {n: obs[0][i] for i, n in enumerate(names)}

    def update_obs(self):
        for n, w in self.obs_lbl.items():
            if n == "comfort_band":
                val = self.rbc.comfort_band if self.rbc else 0.0
                w.config(text=f"{val:.3f}")
            else:
                w.config(text=f"{self.obs_dict.get(n, 0.0):.3f}")

    # ── Policy ────────────────────────────────────────────────────────────────
    def predict(self, obs_dict):
        if self.policy_var.get() == "rbc":
            if not self.rbc or self.obs is None:
                return {a: 0.0 for a in ACTION_NAMES}
            vec  = [obs_dict.get(n, 0.0) for n in self.rbc.observation_names[0]]
            raw  = self.rbc.predict([vec])
            acts = raw[0] if raw else [0.0, 0.0, 0.0]
            return {n: clamp(n, acts[i]) for i, n in enumerate(ACTION_NAMES)}
        return self.neural.predict(obs_dict)

    def compute(self):
        """
        Returns (clean_actions, perturbed_actions, clean_obs, perturbed_obs).
        Applies ALL perturbations in self.pert_list.
        """
        c_obs = dict(self.obs_dict)
        p_obs = dict(self.obs_dict)

        # Separate obs perturbations from action perturbations
        action_deltas: dict[str, float] = {}
        for p in self.pert_list:
            feat   = p["feat"]
            amount = p["amount"]
            action_target = ACTION_PERTURB_MAP.get(feat)
            if action_target is not None:
                action_deltas[action_target] = action_deltas.get(action_target, 0.0) + amount
            else:
                p_obs[feat] = p_obs.get(feat, 0.0) + amount

        c = self.predict(c_obs)
        p = self.predict(p_obs)

        # Apply action-level perturbations on top of policy output
        for act_name, delta in action_deltas.items():
            p[act_name] = clamp(act_name, p[act_name] + delta)

        return c, p, c_obs, p_obs

    def refresh(self):
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
        policy = "rbc" if self.policy_var.get() == "rbc" else "linearregressor"
        sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "simulations")
        os.makedirs(sim_dir, exist_ok=True)
        self._sim_path    = os.path.join(sim_dir, f"sim_{policy}_{ts}.csv")
        self._sim_file    = open(self._sim_path, "w", newline="", encoding="utf-8")
        self._sim_writer  = csv.DictWriter(self._sim_file, fieldnames=SIM_ALL_COLS)
        self._sim_writer.writeheader()
        self._sim_file.flush()
        self._session_open = True
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
        log(self.console, f"Session saved: {self._sim_path}")
        self._load_sim_for_plot(self._sim_path)
        self._update_session_label()

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

    # ── Simulation ────────────────────────────────────────────────────────────
    def step(self):
        if not self.env or self.obs is None:
            log(self.console, "Environment not initialised."); return
        c, p, c_obs, p_obs = self.compute()
        acts        = p if self.pert_list else c
        action_list = [[acts[a] for a in ACTION_NAMES]]
        try:
            obs, reward, term, trunc, _ = self.env.step(action_list)
            self.store(obs)
            self.step_n += 1
            ctrl = "Neural" if self.policy_var.get() == "neural" else "RBC"
            log(self.console, f"Step {self.step_n} [{ctrl}] reward {reward[0]:.4f}")
            for dev in ACTION_NAMES:
                d = abs(p[dev] - c[dev])
                if d > 0.05:
                    log(self.console,
                        f"  ! {dev} clean={c[dev]:.3f} "
                        f"perturbed={p[dev]:.3f} D={d:.3f}")
            self._write_sim_row(reward[0], c, p, c_obs, p_obs)
            self.update_obs(); self.refresh()
            if term or trunc:
                log(self.console, "Episode finished.")
                self.stop_req = True
                self.running  = False
                self._close_session()
        except Exception as e:
            log(self.console, f"  skip step {self.step_n + 1}: {e}")
            self.step_n += 1
            self.root.update_idletasks()

    def start(self):
        if self.running:
            log(self.console, "Already running."); return
        if not self.env:
            log(self.console, "Environment not initialised."); return
        if self.policy_var.get() == "neural" and not self.neural.ready:
            log(self.console, "Load the neural policy first."); return
        self._open_session()   # opens only if not already open (pause→resume)
        self.running  = True
        self.stop_req = False
        log(self.console, "Simulation running...")
        self.loop()

    def loop(self):
        if self.stop_req:
            self.running = False
            # NOTE: session stays open on Stop – user can resume with Start
            log(self.console, "Simulation paused. Press Start to resume.")
            return
        self.step()
        self.root.after(400, self.loop)

    def stop(self):
        """Pause: stops the loop but keeps the session (CSV) open."""
        self.stop_req = True

    # ── Load model ────────────────────────────────────────────────────────────
    _MODEL_SEARCH_DIR = ["../linear_regressor_results"]

    def _auto_load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for rel in self._MODEL_SEARCH_DIR:
            directory = os.path.normpath(os.path.join(script_dir, rel))
            mean_ok = os.path.exists(os.path.join(directory, "feature_mean.npy"))
            std_ok  = os.path.exists(os.path.join(directory, "feature_std.npy"))
            log(self.console,
                f"[auto-load] trying {directory} (mean={mean_ok}, std={std_ok})")
            if mean_ok and std_ok:
                log(self.console,
                    f"Auto-loading neural model from: {directory}...")
                ok, msg = self.neural.load(directory)
                self.neural_status.config(text=f"Status: {msg}",
                                          fg=GREEN if ok else RED)
                log(self.console, msg)
                return
        log(self.console,
            "Neural model not found automatically "
            f"(searched: {', '.join(self._MODEL_SEARCH_DIR)}). "
            "Use 'Load model' to select it manually.")

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
        path = filedialog.askopenfilename(
            title="Seleziona il dataset CSV (baseline)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        try:
            self.dataset_df = pd.read_csv(path)
            log(self.console,
                f"Dataset caricato: {os.path.basename(path)} "
                f"({len(self.dataset_df)} righe, "
                f"{len(self.dataset_df.columns)} colonne)")
            self.dataset_status.config(
                text=f"Loaded: {os.path.basename(path)}", fg="#5856D6")
        except Exception as e:
            log(self.console, f"Errore caricamento dataset: {e}"); return
        self._show_ext_obs_picker()

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
            log(self.console, "Nessuna colonna obs trovata nel dataset."); return
        dlg = tk.Toplevel(self.root)
        dlg.title("Scegli l'osservazione da plottare")
        dlg.geometry("420x180"); dlg.resizable(False, False)
        dlg.configure(bg=BG_CARD); dlg.grab_set()
        tk.Label(dlg, text="Osservazione:", bg=BG_CARD,
                 font=("Helvetica", 12)).pack(padx=20, pady=(20, 6), anchor="w")
        obs_var = tk.StringVar(value=available[0])
        ttk.Combobox(dlg, textvariable=obs_var, values=available,
                     state="readonly", font=("Helvetica", 11), width=40) \
            .pack(padx=20, pady=(0, 16))
        def confirm():
            self.ext_plot_obs_var.set(obs_var.get())
            self.unified_plot_col_var.set(obs_var.get())
            if self.sim_col_combo is not None:
                self.sim_col_combo.set(obs_var.get())
            dlg.destroy()
            self._draw_unified_plot()
        btn(dlg, "Plot", confirm, BLUE, width=10).pack(pady=(0, 10))

    # ── Simulation CSV → plot ─────────────────────────────────────────────────
    def _load_sim_for_plot(self, path: str):
        try:
            self.sim_df = pd.read_csv(path)
            all_choices = list(dict.fromkeys(
                c for c in SIM_PLOT_CHOICES if c in self.sim_df.columns))
            if self.sim_col_combo is not None:
                self.sim_col_combo["values"] = all_choices
            self.sim_status_lbl.config(
                text=f"Loaded: {os.path.basename(path)}", fg="#5856D6")
            log(self.console,
                f"Sim plot ready - {len(self.sim_df)} steps "
                f"in {os.path.basename(path)}")
            self._draw_unified_plot()
        except Exception as e:
            log(self.console, f"Errore lettura sim file: {e}")

    def load_sim_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona un file di simulazione CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        self._load_sim_for_plot(path)

    # ── Unified plot ──────────────────────────────────────────────────────────
    def _draw_unified_plot(self):
        col = self.unified_plot_col_var.get()

        baseline_series = None
        df_d = self.dataset_df
        if df_d is not None:
            csv_name = CL_TO_CSV.get(col, col)
            col_key  = (csv_name if csv_name in df_d.columns
                        else col if col in df_d.columns
                        else None)
            if col_key is not None:
                baseline_series = df_d[col_key].to_numpy(dtype=float)

        sim_series = None
        df_s = self.sim_df
        if df_s is not None and col in df_s.columns:
            sim_series = df_s[col].to_numpy(dtype=float)

        if baseline_series is None and sim_series is None:
            log(self.console, f"No data available for column: '{col}'.")
            return

        if self.unified_placeholder is not None:
            self.unified_placeholder.pack_forget()
            self.unified_placeholder = None

        if self.unified_plot_fig is None:
            self.unified_plot_fig, self.unified_plot_ax = plt.subplots(
                figsize=(10, 2.8), facecolor=BG_CARD)
            self.unified_plot_fig.subplots_adjust(
                left=0.05, right=0.99, top=0.80, bottom=0.22)

        if self.unified_plot_ax is None:
            return

        ax = self.unified_plot_ax
        ax.cla()

        if baseline_series is not None:
            ax.plot(np.arange(len(baseline_series)), baseline_series,
                    color=BLUE, linewidth=1.4,
                    label=f"{col} (baseline)", alpha=0.9)

        if sim_series is not None and df_s is not None:
            t2 = (df_s["step"].to_numpy()
                  if "step" in df_s.columns
                  else np.arange(len(sim_series)))
            ax.plot(t2, sim_series, color=ORANGE, linewidth=1.4,
                    label=f"{col} (sim)", linestyle="--", alpha=0.9)

        ax.set_title(col, fontsize=10, pad=4)
        ax.set_xlabel("timestep", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_facecolor("#FAFAFA")
        for s in ax.spines.values():
            s.set_edgecolor("#DDDDDD")

        if self.unified_plot_canvas is None:
            self.unified_plot_canvas = FigureCanvasTkAgg(
                self.unified_plot_fig, master=self.unified_plot_inner)
            self.unified_plot_canvas.get_tk_widget() \
                .pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self.unified_plot_canvas.draw()

        parts = []
        if baseline_series is not None: parts.append("baseline")
        if sim_series       is not None: parts.append("sim")
        log(self.console, f"Plot aggiornato: {col} [{', '.join(parts)}]")

    # ── Multi-perturbation management ─────────────────────────────────────────
    def _add_perturbation(self):
        feat = self.pert_feat.get()
        try:
            amount = float(self.pert_entry.get().replace(",", "."))
        except ValueError:
            log(self.console, "Invalid perturbation amount."); return
        if amount == 0.0:
            log(self.console, "Amount is 0 – nothing added."); return
        self.pert_list.append({"feat": feat, "amount": amount})
        log(self.console, f"Perturbation added: {feat} {amount:+.4f}")
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
            tk.Label(row_f,
                     text=f"{p['feat']}  {p['amount']:+.4f}",
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
    def build(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        hdr = tk.Frame(self.root, bg=BG_CARD,
                       highlightbackground="#D0D0D0", highlightthickness=1)
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        lbl(hdr, "RBC vs Neural Policy", bold=True, size=18) \
            .pack(fill="x", padx=14, pady=(12, 12))

        body = tk.Frame(self.root, bg=BG_APP)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.columnconfigure(2, weight=3)

        self.build_left(body)
        self.build_center(body)
        self.build_right(body)
        self.build_unified_plot(body)

    def build_left(self, parent):
        _, inner = card(parent, "Controls", col=0, row=0)

        # Controller
        ctrl = section(inner, "Controller")
        for text, val in [("RBC (AdvancedRBC)", "rbc"),
                          ("Neural policy (LinearRegressor)", "neural")]:
            tk.Radiobutton(ctrl, text=text, variable=self.policy_var,
                           value=val, command=self.refresh, bg=BG_CARD,
                           font=("Helvetica", 11)) \
                .pack(anchor="w", padx=8,
                      pady=(8, 8) if val == "rbc" else (0, 8))

        # Neural policy
        trn = section(inner, "Neural policy")
        btn(trn, "Load model", self.load_model, BLUE, width=18) \
            .pack(anchor="w", padx=8, pady=(10, 6))
        self.neural_status = lbl(trn, "Status: not loaded", fg="gray")
        self.neural_status.pack(fill="x", padx=8, pady=(0, 10))

        # Dataset baseline
        ds = section(inner, "Dataset (baseline)")
        btn(ds, "Load dataset", self.load_dataset, "#5856D6", width=18) \
            .pack(anchor="w", padx=8, pady=(10, 6))
        self.dataset_status = lbl(ds, "No dataset loaded", fg="gray")
        self.dataset_status.pack(fill="x", padx=8, pady=(0, 4))
        btn(ds, "Change obs", self._show_ext_obs_picker, GRAY,
            fg="black", width=18) \
            .pack(anchor="w", padx=8, pady=(0, 10))

        # ── Perturbations ─────────────────────────────────────────────────────
        pb = section(inner, "Perturbations")

        # Feature selector + amount entry
        lbl(pb, "Feature:").pack(fill="x", padx=8, pady=(10, 2))
        ttk.Combobox(pb, textvariable=self.pert_feat,
                     values=PERTURB_TARGETS, state="readonly",
                     font=("Helvetica", 11)) \
            .pack(fill="x", padx=8, pady=(0, 6))

        er = tk.Frame(pb, bg=BG_CARD)
        er.pack(fill="x", padx=8, pady=(0, 6))
        lbl(er, "Amount:", size=10).pack(side="left", padx=(0, 4))
        self.pert_entry = tk.Entry(er, font=("Helvetica", 11), width=9,
                                   relief="solid", bd=1, justify="center")
        self.pert_entry.insert(0, "0.00")
        self.pert_entry.pack(side="left", ipady=4)

        br = tk.Frame(pb, bg=BG_CARD)
        br.pack(fill="x", padx=8, pady=(0, 6))
        btn(br, "Add", self._add_perturbation, ORANGE, width=10) \
            .pack(side="left", padx=(0, 6))
        btn(br, "Clear all", self._clear_all_perturbations,
            GRAY, fg="black", width=10).pack(side="left")

        # Count label
        self.pert_count_lbl = lbl(pb, "0 active", fg="gray", size=10)
        self.pert_count_lbl.pack(fill="x", padx=8, pady=(0, 4))

        # Scrollable list of active perturbations
        list_outer = tk.Frame(pb, bg="#F4F4F4",
                              highlightbackground="#DDDDDD", highlightthickness=1)
        list_outer.pack(fill="x", padx=8, pady=(0, 10))
        self.pert_list_frame = tk.Frame(list_outer, bg=BG_CARD)
        self.pert_list_frame.pack(fill="x", padx=2, pady=2)
        self._rebuild_pert_list_ui()

        # ── Simulation ────────────────────────────────────────────────────────
        sim = section(inner, "Simulation")

        # Session status
        self.session_lbl = lbl(sim, "No active session.", fg="gray", size=10)
        self.session_lbl.pack(fill="x", padx=8, pady=(6, 4))

        r1 = tk.Frame(sim, bg=BG_CARD)
        r1.pack(fill="x", padx=8, pady=(0, 6))
        btn(r1, "Start", self.start, BLUE, width=12).pack(side="left", padx=(0, 8))
        btn(r1, "Stop",  self.stop,  RED,  width=12).pack(side="left")
        r2 = tk.Frame(sim, bg=BG_CARD)
        r2.pack(fill="x", padx=8, pady=(0, 10))
        btn(r2, "Step",  self.step,      GREEN,        width=12).pack(side="left", padx=(0, 8))
        btn(r2, "Reset", self.reset_env, GRAY, fg="black", width=12).pack(side="left")

    def build_center(self, parent):
        _, inner = card(parent, "Action comparison", col=1, row=0, padx=(6, 6))
        tbl = tk.Frame(inner, bg=BG_CARD)
        tbl.pack(fill="both", expand=True)
        for c in range(4): tbl.columnconfigure(c, weight=1)

        for col, (h, bg, fg) in enumerate([
            ("Device",          BG_CARD,      "gray"),
            ("Clean input",     BLUE_LIGHT,   BLUE),
            ("Perturbed input", ORANGE_LIGHT, ORANGE),
            ("Difference",      BG_CARD,      "gray"),
        ]):
            tk.Label(tbl, text=h, bg=bg, fg=fg,
                     font=("Helvetica", 12, "bold"),
                     anchor="center", pady=10) \
                .grid(row=0, column=col, sticky="ew", padx=(0, 4), pady=(0, 8))

        for row, dev in enumerate(ACTION_NAMES, start=1):
            lbl(tbl, dev, bold=True, size=12, padx=10) \
                .grid(row=row, column=0, sticky="ew",
                      padx=(0, 4), pady=(0, 8), ipady=20)
            cl = tk.Label(tbl, text="0.000", bg=BLUE_LIGHT, fg=BLUE,
                          font=("Helvetica", 22, "bold"),
                          anchor="center", pady=20)
            cl.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.clean_lbl[dev] = cl
            pl = tk.Label(tbl, text="0.000", bg=ORANGE_LIGHT, fg=ORANGE,
                          font=("Helvetica", 22, "bold"),
                          anchor="center", pady=20)
            pl.grid(row=row, column=2, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.pert_lbl[dev] = pl
            dl = tk.Label(tbl, text="D 0.000", bg=BG_CARD, fg=GREEN,
                          font=("Helvetica", 16, "bold"),
                          anchor="center", pady=20)
            dl.grid(row=row, column=3, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.diff_lbl[dev] = dl

        leg = tk.Frame(inner, bg=BG_CARD)
        leg.pack(fill="x", pady=(12, 0))
        for color, label, desc in [
            (GREEN,  "D < 0.05",    "robust"),
            (ORANGE, "D 0.05-0.20", "sensitive"),
            (RED,    "D > 0.20",    "fragile"),
        ]:
            lbl(leg, label, fg=color, bold=True, size=11) \
                .pack(side="left", padx=(0, 4))
            lbl(leg, desc, fg="gray", size=11) \
                .pack(side="left", padx=(0, 20))

    def build_right(self, parent):
        outer, inner = card(parent, "Current observations", col=2, row=0,
                            padx=(6, 0))
        outer.rowconfigure(1, weight=1)
        inner.rowconfigure(1, weight=1)
        inner.columnconfigure(0, weight=1)

        obs_f = tk.Frame(inner, bg=BG_CARD)
        obs_f.pack(fill="x")
        obs_f.columnconfigure(1, weight=1)
        for i, name in enumerate(OBS_NAMES_DISPLAY):
            fg_name = "#5856D6" if name == "comfort_band" else "gray"
            lbl(obs_f, name, fg=fg_name, size=11) \
                .grid(row=i, column=0, sticky="w", padx=(0, 12), pady=2)
            w = lbl(obs_f, "---", bold=True, size=11, anchor="e")
            w.grid(row=i, column=1, sticky="e", pady=2)
            self.obs_lbl[name] = w

        lbl(inner, "Console", bold=True, size=12) \
            .pack(fill="x", pady=(12, 4))
        self.console = tk.Text(inner, wrap="word", state="disabled",
                               font=("Courier", 11), relief="flat", bd=0,
                               bg="#F8F8F8", padx=8, pady=8)
        self.console.pack(fill="both", expand=True)
        log(self.console, "GUI ready.")
        log(self.console, "Load the neural policy before switching to it.")
        log(self.console, "Add perturbations and press Start to begin a session.")

    def build_unified_plot(self, parent):
        """Unified plot: baseline (blue) + sim (orange)."""
        outer, inner = card(parent,
                            "Dataset plot (baseline vs simulation)",
                            col=0, row=1, padx=(0, 0), colspan=3)
        outer.grid(pady=(6, 0))
        self.unified_plot_inner = inner

        ctrl_bar = tk.Frame(inner, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 6))

        lbl(ctrl_bar, "Column:", fg="gray", size=11) \
            .pack(side="left", padx=(0, 6))

        all_choices = list(dict.fromkeys(OBS_NAMES + SIM_PLOT_CHOICES))
        self.sim_col_combo = ttk.Combobox(
            ctrl_bar, textvariable=self.unified_plot_col_var,
            values=all_choices, state="readonly",
            font=("Helvetica", 11), width=32)
        self.sim_col_combo.pack(side="left", padx=(0, 10))
        self.sim_col_combo.bind("<<ComboboxSelected>>",
                                lambda e: self._draw_unified_plot())

        btn(ctrl_bar, "Plot", self._draw_unified_plot, BLUE, width=8) \
            .pack(side="left", padx=(0, 10))
        btn(ctrl_bar, "Load sim file", self.load_sim_file,
            "#5856D6", width=14).pack(side="left", padx=(0, 10))

        self.sim_status_lbl = lbl(ctrl_bar, "No simulation loaded yet.",
                                  fg="gray", size=10)
        self.sim_status_lbl.pack(side="left")

        self.unified_placeholder = lbl(
            inner,
            "Load a dataset CSV (baseline) and/or a simulation CSV to display the plot here.",
            fg="gray", size=11, anchor="center")
        self.unified_placeholder.pack(expand=True, fill="both", pady=20)

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    App(tk.Tk()).mainloop()