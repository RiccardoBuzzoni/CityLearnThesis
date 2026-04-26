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
    AdvancedRBC = CityLearnEnv = None

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
    "indoor_dry_bulb_temperature":              "indoor_temp",
    "indoor_dry_bulb_temperature_cooling_set_point": "cooling_set_point",
    "outdoor_dry_bulb_temperature":             "outdoor_temp",
    "outdoor_dry_bulb_temperature_predicted_1": "outdoor_temp_predicted",
    "cooling_demand":                           "cooling_demand",
    "electricity_pricing":                      "elec_price",
    "carbon_intensity":                         "carbon_intensity",
    "solar_generation":                         "solar_generation",
    "occupant_count":                           "occupant_count",
    "hour":                                     "hour",
    "electrical_storage_soc":                   "electrical_storage_soc",
    "dhw_storage_soc":                          "dhw_storage_soc",
    "dhw_demand":                               "dhw_demand",
}
CL_TO_LR = CL_TO_CSV
DEFAULT_DATASET = "citylearn_challenge_2023_phase_1"

# Colonne CSV di simulazione
SIM_ACT_COLS = (
    [f"{a}_clean" for a in ACTION_NAMES] +
    [f"{a}_pert"  for a in ACTION_NAMES] +
    [f"{a}_delta" for a in ACTION_NAMES]
)
SIM_META_COLS = ["pert_feat", "pert_amount"]
SIM_ALL_COLS  = (["step", "policy", "reward"] +
                 OBS_NAMES + SIM_ACT_COLS + SIM_META_COLS)

SIM_PLOT_CHOICES = (
    OBS_NAMES +
    [f"{a}_clean" for a in ACTION_NAMES] +
    [f"{a}_pert"  for a in ACTION_NAMES] +
    [f"{a}_delta" for a in ACTION_NAMES]
)

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
                    anchor=anchor, **kw)

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
        self.mean = self.std = None

    def load(self, directory: str):
        if lr is None:
            return False, "linear_regressor.py non importabile"
        try:
            mean_path = os.path.join(directory, "feature_mean.npy")
            std_path  = os.path.join(directory, "feature_std.npy")
            if not os.path.exists(mean_path) or not os.path.exists(std_path):
                return False, "feature_mean.npy / feature_std.npy non trovati"
            self.mean = np.load(mean_path)
            self.std  = np.load(std_path)
            self.std[self.std == 0] = 1.0
            models = {}; features = {}
            for target in lr.TARGET_COLS:
                pth_path = os.path.join(directory, f"model_{target}.pth")
                if not os.path.exists(pth_path):
                    return False, f"File mancante: model_{target}.pth"
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
            return True, f"Modello caricato da: {os.path.basename(directory)}"
        except Exception as e:
            return False, f"Errore caricamento: {e}"

    def predict(self, obs):
        if not self.ready:
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

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                log(self.widget, line.rstrip())

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

        self.policy_var  = tk.StringVar(value="rbc")
        self.pert_on     = tk.BooleanVar(value=False)
        self.pert_feat   = tk.StringVar(value="outdoor_dry_bulb_temperature")
        self.pert_amount = tk.DoubleVar(value=0.0)
        self.running     = False
        self.stop_req    = False
        self.step_n      = 0

        # ── Dataset baseline (pannello sinistro)
        self.dataset_df       = None      # DataFrame baseline
        self.ext_plot_obs_var = tk.StringVar(value=OBS_NAMES[0])

        # ── Dataset simulazione (secondo dataset)
        self.sim_df = None                # DataFrame sim CSV

        # ── Pannello plot unificato
        self.unified_plot_canvas  = None
        self.unified_plot_fig     = None
        self.unified_plot_ax      = None
        self.unified_placeholder  = None
        self.unified_plot_col_var = tk.StringVar(value=OBS_NAMES[0])
        self.sim_col_combo        = None   # riferimento al widget Combobox

        # ── Registrazione simulazione
        self._sim_file   = None
        self._sim_writer = None
        self._sim_path   = ""

        self.build()
        self.init_env()

    # ── Environment ──────────────────────────────────────────────────────────
    def init_env(self):
        if not CITYLEARN_AVAILABLE:
            log(self.console, f"ERROR CityLearn: {CITYLEARN_ERROR}"); return
        try:
            log(self.console, "Initialising CityLearn environment...")
            schema   = default_env_config(DEFAULT_DATASET)
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
        names      = self.rbc.observation_names[0]
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

    def is_action_perturb(self):
        return ACTION_PERTURB_MAP.get(self.pert_feat.get())

    def compute(self):
        c_obs = dict(self.obs_dict); p_obs = dict(self.obs_dict)
        action_target = self.is_action_perturb()
        if self.pert_on.get() and action_target is None:
            f = self.pert_feat.get()
            p_obs[f] = p_obs.get(f, 0.0) + self.pert_amount.get()
        c = self.predict(c_obs); p = self.predict(p_obs)
        if self.pert_on.get() and action_target is not None:
            p[action_target] = clamp(action_target,
                                     p[action_target] + self.pert_amount.get())
        return c, p

    def refresh(self):
        c, p = self.compute()
        for dev in ACTION_NAMES:
            d = abs(p[dev] - c[dev])
            self.clean_lbl[dev].config(text=f"{c[dev]:.3f}")
            self.pert_lbl[dev].config(text=f"{p[dev]:.3f}")
            self.diff_lbl[dev].config(text=f"D {d:.3f}", fg=diff_color(d))

    # ── CSV Logging ───────────────────────────────────────────────────────────
    def _open_sim_file(self):
        ts     = datetime.now().strftime("%d%m%Y_%H%M")
        policy = "rbc" if self.policy_var.get() == "rbc" else "neural"
        sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "simulations")
        os.makedirs(sim_dir, exist_ok=True)
        self._sim_path   = os.path.join(sim_dir, f"sim_{policy}_{ts}.csv")
        self._sim_file   = open(self._sim_path, "w", newline="", encoding="utf-8")
        self._sim_writer = csv.DictWriter(self._sim_file, fieldnames=SIM_ALL_COLS)
        self._sim_writer.writeheader()
        self._sim_file.flush()
        log(self.console, f"Saving simulation to: {self._sim_path}")

    def _close_sim_file(self):
        if self._sim_file:
            self._sim_file.close()
            self._sim_file   = None
            self._sim_writer = None
            log(self.console, f"Simulation saved: {self._sim_path}")
            self._load_sim_for_plot(self._sim_path)

    def _write_sim_row(self, reward, c, p):
        if self._sim_writer is None:
            return
        row = {
            "step":        self.step_n,
            "policy":      self.policy_var.get(),
            "reward":      round(float(reward), 6),
            "pert_feat":   self.pert_feat.get() if self.pert_on.get() else "",
            "pert_amount": self.pert_amount.get() if self.pert_on.get() else 0.0,
        }
        for n in OBS_NAMES:
            row[n] = round(self.obs_dict.get(n, 0.0), 6)
        for a in ACTION_NAMES:
            row[f"{a}_clean"] = round(c[a], 6)
            row[f"{a}_pert"]  = round(p[a], 6)
            row[f"{a}_delta"] = round(abs(p[a] - c[a]), 6)
        self._sim_writer.writerow(row)
        self._sim_file.flush()

    # ── Simulation ────────────────────────────────────────────────────────────
    def step(self):
        if not self.env or self.obs is None:
            log(self.console, "Environment not initialised."); return
        c, p = self.compute()
        acts = p if self.pert_on.get() else c
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
            self._write_sim_row(reward[0], c, p)
            self.update_obs(); self.refresh()
            if term or trunc:
                log(self.console, "Episode finished.")
                self.stop_req = True
        except Exception as e:
            log(self.console, f"  skip step {self.step_n + 1}: {e}")

    def start(self):
        if self.running:
            log(self.console, "Already running."); return
        if not self.env:
            log(self.console, "Environment not initialised."); return
        if self.policy_var.get() == "neural" and not self.neural.ready:
            log(self.console, "Load the neural policy first."); return
        self._open_sim_file()
        self.running  = True
        self.stop_req = False
        log(self.console, "Simulation started.")
        self.loop()

    def loop(self):
        if self.stop_req:
            self.running = False
            self._close_sim_file()
            log(self.console, "Simulation stopped."); return
        self.step()
        self.root.after(400, self.loop)

    def stop(self):
        self.stop_req = True

    # ── Load model ────────────────────────────────────────────────────────────
    def load_model(self):
        directory = filedialog.askdirectory(
            title="Seleziona la cartella del modello (contiene i .pth e i .npy)")
        if not directory: return
        log(self.console, f"Caricamento modello da: {directory} ...")
        ok, msg = self.neural.load(directory)
        self.neural_status.config(text=f"Status: {msg}",
                                  fg=GREEN if ok else RED)
        log(self.console, msg)

    # ── Dataset baseline (pannello sinistro) ──────────────────────────────────
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
                     state="readonly", font=("Helvetica", 11), width=40)\
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
            # aggiorna i valori del combobox con le colonne disponibili
            all_choices = OBS_NAMES + SIM_PLOT_CHOICES
            available   = list(dict.fromkeys(  # dedup preserving order
                c for c in all_choices if c in self.sim_df.columns))
            if self.sim_col_combo is not None:
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
        path = filedialog.askopenfilename(
            title="Seleziona un file di simulazione CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        self._load_sim_for_plot(path)

    # ── Unified plot ──────────────────────────────────────────────────────────
    def _draw_unified_plot(self):
        col = self.unified_plot_col_var.get()

        # Recupera serie baseline dal dataset statico
        baseline_series = None
        if self.dataset_df is not None:
            cl_name  = col
            csv_name = CL_TO_CSV.get(cl_name, cl_name)
            col_key  = (csv_name if csv_name in self.dataset_df.columns
                        else cl_name if cl_name in self.dataset_df.columns
                        else None)
            if col_key is not None:
                baseline_series = self.dataset_df[col_key].values.astype(float)

        # Recupera serie simulazione
        sim_series = None
        if self.sim_df is not None and col in self.sim_df.columns:
            sim_series = self.sim_df[col].values.astype(float)

        if baseline_series is None and sim_series is None:
            log(self.console, f"Nessun dato disponibile per la colonna '{col}'.")
            return

        # Nasconde placeholder
        if self.unified_placeholder is not None:
            self.unified_placeholder.pack_forget()
            self.unified_placeholder = None

        # Crea figura al primo utilizzo
        if self.unified_plot_fig is None:
            self.unified_plot_fig, self.unified_plot_ax = plt.subplots(
                figsize=(10, 2.8), facecolor=BG_CARD)
            self.unified_plot_fig.subplots_adjust(
                left=0.05, right=0.99, top=0.80, bottom=0.22)

        ax = self.unified_plot_ax
        ax.cla()

        if baseline_series is not None:
            t = np.arange(len(baseline_series))
            ax.plot(t, baseline_series, color=BLUE, linewidth=1.4,
                    label=f"{col} (baseline)", alpha=0.9)

        if sim_series is not None:
            t2 = (self.sim_df["step"].values
                  if "step" in self.sim_df.columns
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
            self.unified_plot_canvas.get_tk_widget()\
                .pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self.unified_plot_canvas.draw()

        parts = []
        if baseline_series is not None: parts.append("baseline")
        if sim_series       is not None: parts.append("sim")
        log(self.console, f"Plot aggiornato: {col} [{', '.join(parts)}]")

    # ── Perturbation ──────────────────────────────────────────────────────────
    def entry_changed(self):
        try:
            self.pert_amount.set(float(self.entry.get().replace(",", ".")))
            self.refresh()
        except ValueError:
            self.entry.delete(0, "end"); self.entry.insert(0, "0.00")

    def apply_pert(self):
        try:
            v = float(self.entry.get().replace(",", "."))
            self.pert_amount.set(v)
        except ValueError:
            v = 0.0; self.entry.delete(0, "end"); self.entry.insert(0, "0.00")
        self.pert_on.set(True)
        log(self.console,
            f"Perturbation active: {self.pert_feat.get()} {v:+.2f}")
        self.refresh(); self.update_pert_lbl()

    def clear_pert(self):
        self.pert_on.set(False); self.pert_amount.set(0.0)
        self.entry.delete(0, "end"); self.entry.insert(0, "0.00")
        log(self.console, "Perturbation cleared.")
        self.refresh(); self.update_pert_lbl()

    def update_pert_lbl(self):
        if self.pert_on.get():
            self.pert_status.config(
                text=f"Active: {self.pert_feat.get()} "
                     f"{self.pert_amount.get():+.2f}", fg=ORANGE)
        else:
            self.pert_status.config(text="Inactive", fg="gray")

    # ── UI Build ──────────────────────────────────────────────────────────────
    def build(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        hdr = tk.Frame(self.root, bg=BG_CARD,
                       highlightbackground="#D0D0D0", highlightthickness=1)
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        lbl(hdr, "RBC vs Neural Policy", bold=True, size=18)\
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

        ctrl = section(inner, "Controller")
        for text, val in [("RBC (AdvancedRBC)", "rbc"),
                          ("Neural policy (LinearRegressor)", "neural")]:
            tk.Radiobutton(ctrl, text=text, variable=self.policy_var,
                           value=val, command=self.refresh, bg=BG_CARD,
                           font=("Helvetica", 11))\
                .pack(anchor="w", padx=8,
                      pady=(8, 8) if val == "rbc" else (0, 8))

        trn = section(inner, "Neural policy")
        btn(trn, "Load model", self.load_model, BLUE, width=18)\
            .pack(anchor="w", padx=8, pady=(10, 6))
        self.neural_status = lbl(trn, "Status: not loaded", fg="gray")
        self.neural_status.pack(fill="x", padx=8, pady=(0, 10))

        ds = section(inner, "Dataset (baseline)")
        btn(ds, "Load dataset", self.load_dataset, "#5856D6", width=18)\
            .pack(anchor="w", padx=8, pady=(10, 6))
        self.dataset_status = lbl(ds, "No dataset loaded", fg="gray")
        self.dataset_status.pack(fill="x", padx=8, pady=(0, 4))
        btn(ds, "Change obs", self._show_ext_obs_picker, GRAY,
            fg="black", width=18)\
            .pack(anchor="w", padx=8, pady=(0, 10))

        pb = section(inner, "Perturbation")
        lbl(pb, "Feature to perturb:").pack(fill="x", padx=8, pady=(10, 2))
        ttk.Combobox(pb, textvariable=self.pert_feat,
                     values=PERTURB_TARGETS, state="readonly",
                     font=("Helvetica", 11))\
            .pack(fill="x", padx=8, pady=(0, 8))
        lbl(pb, "Perturbation amount:").pack(fill="x", padx=8)
        er = tk.Frame(pb, bg=BG_CARD)
        er.pack(fill="x", padx=8, pady=(4, 8))
        self.entry = tk.Entry(er, font=("Helvetica", 11), width=10,
                              relief="solid", bd=1, justify="center")
        self.entry.insert(0, "0.00")
        self.entry.pack(side="left", padx=(0, 8), ipady=5)
        self.entry.bind("<Return>",    lambda e: self.entry_changed())
        self.entry.bind("<FocusOut>",  lambda e: self.entry_changed())
        lbl(er, "press Enter to update", fg="gray", size=10).pack(side="left")
        br = tk.Frame(pb, bg=BG_CARD)
        br.pack(fill="x", padx=8, pady=(0, 8))
        btn(br, "Apply", self.apply_pert, ORANGE, width=11)\
            .pack(side="left", padx=(0, 8))
        btn(br, "Clear", self.clear_pert, GRAY, fg="black", width=11)\
            .pack(side="left")
        self.pert_status = lbl(pb, "Inactive", fg="gray")
        self.pert_status.pack(fill="x", padx=8, pady=(0, 10))

        sim = section(inner, "Simulation")
        r1 = tk.Frame(sim, bg=BG_CARD)
        r1.pack(fill="x", padx=8, pady=(10, 6))
        btn(r1, "Start", self.start, BLUE,        width=12).pack(side="left", padx=(0, 8))
        btn(r1, "Stop",  self.stop,  RED,          width=12).pack(side="left")
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
                     anchor="center", pady=10)\
                .grid(row=0, column=col, sticky="ew", padx=(0, 4), pady=(0, 8))

        for row, dev in enumerate(ACTION_NAMES, start=1):
            lbl(tbl, dev, bold=True, size=12, padx=10)\
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
            lbl(leg, label, fg=color, bold=True, size=11)\
                .pack(side="left", padx=(0, 4))
            lbl(leg, desc, fg="gray", size=11)\
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
            lbl(obs_f, name, fg=fg_name, size=11)\
                .grid(row=i, column=0, sticky="w", padx=(0, 12), pady=2)
            w = lbl(obs_f, "---", bold=True, size=11, anchor="e")
            w.grid(row=i, column=1, sticky="e", pady=2)
            self.obs_lbl[name] = w

        lbl(inner, "Console", bold=True, size=12)\
            .pack(fill="x", pady=(12, 4))
        self.console = tk.Text(inner, wrap="word", state="disabled",
                               font=("Courier", 11), relief="flat", bd=0,
                               bg="#F8F8F8", padx=8, pady=8)
        self.console.pack(fill="both", expand=True)
        log(self.console, "GUI ready.")
        log(self.console, "Load the neural policy before switching to it.")
        log(self.console, "Set a perturbation and press Apply to test robustness.")

    def build_unified_plot(self, parent):
        """Pannello plot unificato: baseline (blu) + sim (arancione)."""
        outer, inner = card(parent,
                            "Dataset plot — baseline vs simulation",
                            col=0, row=1, padx=(0, 0), colspan=3)
        outer.grid(pady=(6, 0))
        self.unified_plot_inner = inner

        # Barra di controllo
        ctrl_bar = tk.Frame(inner, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 6))

        lbl(ctrl_bar, "Column:", fg="gray", size=11)\
            .pack(side="left", padx=(0, 6))

        all_choices = list(dict.fromkeys(OBS_NAMES + SIM_PLOT_CHOICES))
        self.sim_col_combo = ttk.Combobox(
            ctrl_bar, textvariable=self.unified_plot_col_var,
            values=all_choices, state="readonly",
            font=("Helvetica", 11), width=32)
        self.sim_col_combo.pack(side="left", padx=(0, 10))
        self.sim_col_combo.bind("<<ComboboxSelected>>",
                                lambda e: self._draw_unified_plot())

        btn(ctrl_bar, "Plot", self._draw_unified_plot, BLUE, width=8)\
            .pack(side="left", padx=(0, 10))
        btn(ctrl_bar, "Load sim file", self.load_sim_file,
            "#5856D6", width=14).pack(side="left", padx=(0, 10))

        self.sim_status_lbl = lbl(ctrl_bar,
                                  "No simulation loaded yet.",
                                  fg="gray", size=10)
        self.sim_status_lbl.pack(side="left")

        # Placeholder
        self.unified_placeholder = lbl(
            inner,
            "Load a dataset CSV (baseline) and/or a simulation CSV to display the plot here.",
            fg="gray", size=11, anchor="center")
        self.unified_placeholder.pack(expand=True, fill="both", pady=20)

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    App(tk.Tk()).mainloop()