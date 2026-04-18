import sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import io
import contextlib
import tkinter as tk
from tkinter import ttk
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
    _CITYLEARN_ERROR = str(e)
    AdvancedRBC = None
    CityLearnEnv = None

# ─────────────────────────────────────────────
# COLORI
# ─────────────────────────────────────────────
BG_APP       = "#ECECEC"
BG_CARD      = "#FFFFFF"
BG_CARD2     = "#F8F8FA"
BORDER       = "#D8D8DC"
TEXT         = "#1C1C1E"
TEXT_MUTED   = "#6E6E73"
BLUE         = "#007AFF"
BLUE_HOVER   = "#0062CC"
BLUE_LIGHT   = "#E8F2FF"
RED          = "#FF3B30"
RED_HOVER    = "#D9362B"
GREEN        = "#34C759"
ORANGE       = "#FF9500"
ORANGE_LIGHT = "#FFF4E0"
GRAY_BUTTON  = "#E5E5EA"
GRAY_HOVER   = "#DCDCE1"
CONSOLE_BG   = "#FBFBFD"
CONSOLE_FG   = "#1C1C1E"

# ─────────────────────────────────────────────
# FONT
# ─────────────────────────────────────────────
F_TITLE    = ("Helvetica", 22, "bold")
F_SUBTITLE = ("Helvetica", 12)
F_SECTION  = ("Helvetica", 15, "bold")
F_LABEL_H  = ("Helvetica", 11, "bold")
F_BODY     = ("Helvetica", 12)
F_BODY_B   = ("Helvetica", 12, "bold")
F_RADIO    = ("Helvetica", 12)
F_BTN      = ("Helvetica", 11, "bold")
F_TABLE_H  = ("Helvetica", 12, "bold")
F_TABLE_V  = ("Helvetica", 24, "bold")
F_DIFF     = ("Helvetica", 18, "bold")
F_DEVICE   = ("Helvetica", 12, "bold")
F_OBS_NAME = ("Helvetica", 11)
F_OBS_VAL  = ("Helvetica", 11, "bold")
F_CONSOLE  = ("Courier", 11)

# ─────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────
RBC_OBSERVATION_NAMES = [
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "cooling_demand",
    "electricity_pricing",
    "carbon_intensity",
    "solar_generation",
    "occupant_count",
    "hour",
    "electrical_storage_soc",
    "dhw_storage_soc",
    "dhw_demand",
]

RBC_ACTION_NAMES = ["cooling_device", "dhw_storage", "electrical_storage"]

PERTURBATION_TARGETS = [
    "outdoor_dry_bulb_temperature",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
    "occupant_count",
    "cooling_demand",
    "dhw_demand",
    "indoor_dry_bulb_temperature",
    "electrical_storage_soc",
    "dhw_storage_soc",
]

CITYLEARN_TO_LR = {
    "indoor_dry_bulb_temperature":                   "indoor_temp",
    "indoor_dry_bulb_temperature_cooling_set_point": "cooling_setpoint",
    "outdoor_dry_bulb_temperature":                  "outdoor_temp",
    "outdoor_dry_bulb_temperature_predicted_1":      "outdoor_temp_predicted",
    "cooling_demand":                                "cooling_demand",
    "electricity_pricing":                           "elec_price",
    "carbon_intensity":                              "carbon_intensity",
    "solar_generation":                              "solar_generation",
    "occupant_count":                                "occupant_count",
    "hour":                                          "hour",
    "electrical_storage_soc":                        "electrical_storage_soc",
    "dhw_storage_soc":                               "dhw_storage_soc",
    "dhw_demand":                                    "dhw_demand",
}

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────
def append_text(text_widget, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    text_widget.config(state="normal")
    text_widget.insert("end", f"[{timestamp}] {message}\n")
    text_widget.see("end")
    text_widget.config(state="disabled")
    text_widget.update_idletasks()


class TextRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                append_text(self.text_widget, line.rstrip())

    def flush(self):
        if self._buffer.strip():
            append_text(self.text_widget, self._buffer.rstrip())
        self._buffer = ""


def create_button(parent, text, command, bg, fg="white", active_bg=None, width=14):
    if active_bg is None:
        active_bg = bg
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=active_bg, activeforeground=fg,
        font=F_BTN, relief="flat", bd=0,
        padx=10, pady=10, width=width, cursor="hand2",
    )


def clamp_action(target, value):
    value = float(value)
    if target == "cooling_device":
        return max(0.0, min(1.0, value))
    return max(-1.0, min(1.0, value))


def obs_vector_to_dict(obs_vector, obs_names):
    return {name: obs_vector[i] for i, name in enumerate(obs_names)}


def diff_color(diff):
    if diff < 0.05:
        return GREEN
    elif diff < 0.20:
        return ORANGE
    else:
        return RED


# ─────────────────────────────────────────────
# POLICY NEURALE
# ─────────────────────────────────────────────
class LinearRegressorPolicy:
    def __init__(self):
        self.available = lr is not None
        self.models    = {}
        self.features  = {}
        self.mean      = None
        self.std       = None
        self.is_ready  = False

    def train(self):
        if not self.available:
            return False, "linear_regressor.py non importabile"
        try:
            self.models, self.features, self.mean, self.std = lr.main()
            self.is_ready = True
            return True, "LinearRegressorPolicy pronta"
        except Exception as e:
            return False, f"Errore training: {e}"

    def predict(self, obs_dict: dict) -> dict:
        if not self.is_ready:
            return {d: 0.0 for d in RBC_ACTION_NAMES}

        lr_obs = {
            lr_name: float(obs_dict.get(cl_name, 0.0))
            for cl_name, lr_name in CITYLEARN_TO_LR.items()
        }

        feature_cols = list(lr.FEATURE_COLS)
        full_row     = [lr_obs.get(col, 0.0) for col in feature_cols]

        norm_row = [
            (v - float(self.mean[i])) / (float(self.std[i]) if float(self.std[i]) != 0 else 1.0)
            for i, v in enumerate(full_row)
        ]

        actions = {}
        for target in lr.TARGET_COLS:
            selected = [norm_row[feature_cols.index(f)] for f in self.features[target]]
            x = lr.torch.tensor([selected], dtype=lr.torch.float32)
            with lr.torch.no_grad():
                y = self.models[target](x).item()
            actions[target] = clamp_action(target, y)

        return actions


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RBC vs Policy Neurale - Robustezza a Perturbazioni")
        self.root.geometry("1900x980")
        self.root.minsize(1600, 820)
        self.root.configure(bg=BG_APP)

        self.clean_labels     = {}
        self.perturbed_labels = {}
        self.diff_labels      = {}
        self.obs_labels       = {}

        self.env                  = None
        self.rbc_agent            = None
        self.current_observations = None
        self.current_obs_dict     = {}

        self.neural_policy = LinearRegressorPolicy()

        self.policy_mode      = tk.StringVar(value="rbc")
        self.perturb_enabled  = tk.BooleanVar(value=False)
        self.perturb_feature  = tk.StringVar(value="outdoor_dry_bulb_temperature")
        self.perturb_strength = tk.DoubleVar(value=0.0)

        self.simulation_running = False
        self.stop_requested     = False
        self.step_count         = 0

        self._last_clean     = {d: 0.0 for d in RBC_ACTION_NAMES}
        self._last_perturbed = {d: 0.0 for d in RBC_ACTION_NAMES}

        self.build_layout()
        self._init_environment()

    # ─────────────────────────────────────────
    # AMBIENTE
    # ─────────────────────────────────────────
    def _init_environment(self):
        if not CITYLEARN_AVAILABLE:
            append_text(self.console, f"ERRORE CityLearn: {_CITYLEARN_ERROR}")
            return
        try:
            append_text(self.console, "Inizializzazione ambiente CityLearn...")
            conf   = Config()
            schema = default_env_config(conf.args.data)
            self.env       = CityLearnEnv(schema=schema, central_agent=True)
            self.rbc_agent = AdvancedRBC(self.env)
            obs, _         = self.env.reset()
            self._store_obs(obs)
            self.step_count = 0
            append_text(self.console, "Ambiente CityLearn pronto.")
            append_text(self.console, f"Episodio: {self.env.time_steps - 1} step totali")
            self._update_obs_display()
            self._refresh_action_display()
        except Exception as e:
            append_text(self.console, f"Errore init CityLearn: {e}")

    def _reset_environment(self):
        if self.env is None:
            self._init_environment()
            return
        try:
            obs, _ = self.env.reset()
            self._store_obs(obs)
            self.step_count = 0
            append_text(self.console, "Ambiente resettato.")
            self._update_obs_display()
            self._refresh_action_display()
        except Exception as e:
            append_text(self.console, f"Errore reset: {e}")

    def _store_obs(self, obs):
        self.current_observations = obs
        obs_names = self.rbc_agent.observation_names[0]
        self.current_obs_dict = obs_vector_to_dict(obs[0], obs_names)

    def _update_obs_display(self):
        for name, label in self.obs_labels.items():
            val = self.current_obs_dict.get(name, 0.0)
            label.config(text=f"{val:.3f}")

    # ─────────────────────────────────────────
    # PREDIZIONE
    # ─────────────────────────────────────────
    def _predict_from_obs(self, obs_dict: dict) -> dict:
        if self.policy_mode.get() == "rbc":
            if self.rbc_agent is None or self.current_observations is None:
                return {d: 0.0 for d in RBC_ACTION_NAMES}
            obs_names  = self.rbc_agent.observation_names[0]
            obs_vector = [obs_dict.get(name, 0.0) for name in obs_names]
            raw  = self.rbc_agent.predict([obs_vector])
            acts = raw[0] if raw else [0.0, 0.0, 0.0]
            return {name: clamp_action(name, acts[i])
                    for i, name in enumerate(RBC_ACTION_NAMES)}
        else:
            return self.neural_policy.predict(obs_dict)

    def _compute_both(self):
        obs_clean     = dict(self.current_obs_dict)
        obs_perturbed = dict(self.current_obs_dict)
        if self.perturb_enabled.get():
            feature  = self.perturb_feature.get()
            strength = self.perturb_strength.get()
            obs_perturbed[feature] = obs_perturbed.get(feature, 0.0) + strength
        return self._predict_from_obs(obs_clean), self._predict_from_obs(obs_perturbed)

    def _refresh_action_display(self):
        clean, perturbed = self._compute_both()
        self._last_clean     = clean
        self._last_perturbed = perturbed
        for device in RBC_ACTION_NAMES:
            c = clean[device]
            p = perturbed[device]
            d = abs(p - c)
            self.clean_labels[device].config(text=f"{c:.3f}")
            self.perturbed_labels[device].config(text=f"{p:.3f}")
            self.diff_labels[device].config(text=f"D {d:.3f}", fg=diff_color(d))

    # ─────────────────────────────────────────
    # SIMULAZIONE
    # ─────────────────────────────────────────
    def run_single_step(self):
        if self.env is None or self.current_observations is None:
            append_text(self.console, "Ambiente non inizializzato.")
            return

        clean, perturbed = self._compute_both()
        self._last_clean     = clean
        self._last_perturbed = perturbed

        actions_for_env = perturbed if self.perturb_enabled.get() else clean
        env_actions     = [[actions_for_env[d] for d in RBC_ACTION_NAMES]]

        try:
            obs, reward, terminated, truncated, _ = self.env.step(env_actions)
            self._store_obs(obs)
            self.step_count += 1

            ctrl = "Policy neurale" if self.policy_mode.get() == "neural" else "RBC"
            append_text(self.console,
                        f"Step {self.step_count} | {ctrl} | reward: {reward[0]:.4f}")

            for device in RBC_ACTION_NAMES:
                d = abs(perturbed[device] - clean[device])
                if d > 0.05:
                    append_text(
                        self.console,
                        f"  ! {device}: pulito={clean[device]:.3f}  "
                        f"perturbato={perturbed[device]:.3f}  D={d:.3f}"
                    )

            self._update_obs_display()
            self._refresh_action_display()

            if terminated or truncated:
                append_text(self.console, "Episodio terminato.")
                self.stop_requested = True

        except Exception as e:
            append_text(self.console, f"Errore step: {e}")
            self.stop_requested = True

    def start_simulation(self):
        if self.simulation_running:
            append_text(self.console, "Simulazione gia in esecuzione")
            return
        if self.env is None:
            append_text(self.console, "Ambiente non inizializzato")
            return
        if self.policy_mode.get() == "neural" and not self.neural_policy.is_ready:
            append_text(self.console, "Policy neurale non addestrata. Premi Train model prima.")
            return
        self.simulation_running = True
        self.stop_requested     = False
        ctrl = "Policy neurale" if self.policy_mode.get() == "neural" else "RBC"
        info = (
            f" | perturbazione: {self.perturb_feature.get()} "
            f"({self.perturb_strength.get():+.2f})"
            if self.perturb_enabled.get() else ""
        )
        append_text(self.console, f"Simulazione avviata con {ctrl}{info}")
        self._schedule_step()

    def _schedule_step(self):
        if self.stop_requested:
            self.simulation_running = False
            append_text(self.console, "Simulazione fermata")
            return
        self.run_single_step()
        self.root.after(400, self._schedule_step)

    def stop_simulation(self):
        self.stop_requested = True
        append_text(self.console, "Richiesta di stop inviata")

    # ─────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────
    def train_neural_policy(self):
        append_text(self.console, "Avvio training LinearRegressorPolicy...")
        redirector = TextRedirector(self.console)
        try:
            with contextlib.redirect_stdout(redirector), contextlib.redirect_stderr(redirector):
                ok, message = self.neural_policy.train()
                redirector.flush()
        except Exception as e:
            ok      = False
            message = f"Errore: {e}"
        self.neural_status_label.config(
            text=f"Stato: {message}",
            fg=GREEN if ok else RED
        )
        append_text(self.console, message)

    # ─────────────────────────────────────────
    # PERTURBAZIONI
    # ─────────────────────────────────────────
    def _on_entry_change(self):
        try:
            value = float(self.perturb_entry.get().replace(",", "."))
            self.perturb_strength.set(value)
            self._refresh_action_display()
        except ValueError:
            self.perturb_entry.delete(0, "end")
            self.perturb_entry.insert(0, "0.00")

    def apply_perturbation(self):
        try:
            value = float(self.perturb_entry.get().replace(",", "."))
            self.perturb_strength.set(value)
        except ValueError:
            value = 0.0
            self.perturb_entry.delete(0, "end")
            self.perturb_entry.insert(0, "0.00")
        self.perturb_enabled.set(True)
        feature = self.perturb_feature.get()
        append_text(self.console, f"Perturbazione attiva: {feature} {value:+.2f}")
        self._refresh_action_display()
        self._update_perturb_status()

    def clear_perturbation(self):
        self.perturb_enabled.set(False)
        self.perturb_strength.set(0.0)
        self.perturb_entry.delete(0, "end")
        self.perturb_entry.insert(0, "0.00")
        append_text(self.console, "Perturbazione disattivata")
        self._refresh_action_display()
        self._update_perturb_status()

    def _update_perturb_status(self):
        if self.perturb_enabled.get():
            feature  = self.perturb_feature.get()
            strength = self.perturb_strength.get()
            self.perturb_status_label.config(
                text=f"Attiva: {feature} {strength:+.2f}", fg=ORANGE
            )
        else:
            self.perturb_status_label.config(text="Disattivata", fg=TEXT_MUTED)

    # ─────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────
    def build_layout(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=BG_CARD,
                          highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        tk.Label(
            header,
            text="RBC vs Policy Neurale  -  Robustezza a Perturbazioni",
            bg=BG_CARD, fg=TEXT, font=F_TITLE, anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(
            header,
            text="Le perturbazioni vengono applicate sull'input della policy  -  "
                 "confronta azioni pulite e perturbate in tempo reale",
            bg=BG_CARD, fg=TEXT_MUTED, font=F_SUBTITLE, anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 14))

        content = tk.Frame(self.root, bg=BG_APP)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.columnconfigure(2, weight=3)

        self._build_left(content)
        self._build_center(content)
        self._build_right(content)

    # ── Pannello sinistro ──────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(left, text="Controlli", bg=BG_CARD, fg=TEXT,
                 font=F_SECTION, anchor="w"
                 ).pack(fill="x", padx=16, pady=(16, 10))

        # Controller
        mode_box = tk.LabelFrame(left, text="Controller", bg=BG_CARD, fg=TEXT,
                                  font=F_LABEL_H)
        mode_box.pack(fill="x", padx=16, pady=(0, 14))

        tk.Radiobutton(
            mode_box, text="RBC  (AdvancedRBC)",
            variable=self.policy_mode, value="rbc",
            command=self._refresh_action_display,
            bg=BG_CARD, font=F_RADIO
        ).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Radiobutton(
            mode_box, text="Policy neurale  (LinearRegressor)",
            variable=self.policy_mode, value="neural",
            command=self._refresh_action_display,
            bg=BG_CARD, font=F_RADIO
        ).pack(anchor="w", padx=12, pady=(0, 12))

        # Training
        train_box = tk.LabelFrame(left, text="Policy neurale", bg=BG_CARD, fg=TEXT,
                                   font=F_LABEL_H)
        train_box.pack(fill="x", padx=16, pady=(0, 14))

        create_button(train_box, "Train model", self.train_neural_policy,
                      BLUE, active_bg=BLUE_HOVER, width=20
                      ).pack(anchor="w", padx=12, pady=(12, 8))

        self.neural_status_label = tk.Label(
            train_box, text="Stato: non addestrata",
            bg=BG_CARD, fg=TEXT_MUTED, anchor="w", font=F_BODY
        )
        self.neural_status_label.pack(fill="x", padx=12, pady=(0, 12))

        # Perturbazioni
        perturb_box = tk.LabelFrame(left, text="Perturbazione input policy",
                                     bg=BG_CARD, fg=TEXT, font=F_LABEL_H)
        perturb_box.pack(fill="x", padx=16, pady=(0, 14))

        tk.Label(perturb_box, text="Feature da perturbare:",
                 bg=BG_CARD, fg=TEXT_MUTED, font=F_BODY
                 ).pack(anchor="w", padx=12, pady=(12, 4))

        style = ttk.Style()
        style.configure("Big.TCombobox", font=F_BODY)
        ttk.Combobox(
            perturb_box, textvariable=self.perturb_feature,
            values=PERTURBATION_TARGETS, state="readonly",
            font=F_BODY, style="Big.TCombobox"
        ).pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(perturb_box, text="Intensita perturbazione:",
                 bg=BG_CARD, fg=TEXT_MUTED, font=F_BODY
                 ).pack(anchor="w", padx=12)

        entry_row = tk.Frame(perturb_box, bg=BG_CARD)
        entry_row.pack(fill="x", padx=12, pady=(6, 12))

        self.perturb_entry = tk.Entry(
            entry_row, font=F_BODY, width=10,
            relief="solid", bd=1, justify="center"
        )
        self.perturb_entry.insert(0, "0.00")
        self.perturb_entry.pack(side="left", padx=(0, 10), ipady=6)
        self.perturb_entry.bind("<Return>",   lambda e: self._on_entry_change())
        self.perturb_entry.bind("<FocusOut>", lambda e: self._on_entry_change())

        tk.Label(entry_row, text="(Invio per aggiornare)",
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Helvetica", 10, "italic")
                 ).pack(side="left")

        btn_p = tk.Frame(perturb_box, bg=BG_CARD)
        btn_p.pack(fill="x", padx=12, pady=(0, 10))
        create_button(btn_p, "Applica", self.apply_perturbation,
                      ORANGE, active_bg="#CC7700", width=11
                      ).pack(side="left", padx=(0, 8))
        create_button(btn_p, "Rimuovi", self.clear_perturbation,
                      GRAY_BUTTON, fg=TEXT, active_bg=GRAY_HOVER, width=11
                      ).pack(side="left")

        self.perturb_status_label = tk.Label(
            perturb_box, text="Disattivata",
            bg=BG_CARD, fg=TEXT_MUTED, anchor="w",
            font=("Helvetica", 11, "italic")
        )
        self.perturb_status_label.pack(fill="x", padx=12, pady=(4, 12))

        # Simulazione
        sim_box = tk.LabelFrame(left, text="Simulazione", bg=BG_CARD, fg=TEXT,
                                 font=F_LABEL_H)
        sim_box.pack(fill="x", padx=16, pady=(0, 16))

        row1 = tk.Frame(sim_box, bg=BG_CARD)
        row1.pack(fill="x", padx=12, pady=(14, 8))
        create_button(row1, "Start", self.start_simulation,
                      BLUE, active_bg=BLUE_HOVER, width=12
                      ).pack(side="left", padx=(0, 10))
        create_button(row1, "Stop", self.stop_simulation,
                      RED, active_bg=RED_HOVER, width=12
                      ).pack(side="left")

        row2 = tk.Frame(sim_box, bg=BG_CARD)
        row2.pack(fill="x", padx=12, pady=(0, 14))
        create_button(row2, "Step", self.run_single_step,
                      GREEN, active_bg=GREEN, width=12
                      ).pack(side="left", padx=(0, 10))
        create_button(row2, "Reset", self._reset_environment,
                      GRAY_BUTTON, fg=TEXT, active_bg=GRAY_HOVER, width=12
                      ).pack(side="left")

    # ── Pannello centrale ──────────────────────
    def _build_center(self, parent):
        center = tk.Frame(parent, bg=BG_CARD,
                          highlightbackground=BORDER, highlightthickness=1)
        center.grid(row=0, column=1, sticky="nsew", padx=8)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        tk.Label(center, text="Confronto azioni", bg=BG_CARD, fg=TEXT,
                 font=F_SECTION, anchor="w"
                 ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))

        table = tk.Frame(center, bg=BG_CARD)
        table.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        for c in range(4):
            table.columnconfigure(c, weight=1)

        headers = ["Dispositivo", "Input pulito", "Input perturbato", "Differenza"]
        col_bgs = [BG_CARD2, BLUE_LIGHT, ORANGE_LIGHT, BG_CARD2]
        col_fgs = [TEXT_MUTED, BLUE, ORANGE, TEXT_MUTED]
        for col, (h, bg, fg) in enumerate(zip(headers, col_bgs, col_fgs)):
            tk.Label(
                table, text=h, bg=bg, fg=fg,
                font=F_TABLE_H, anchor="center", padx=8, pady=12
            ).grid(row=0, column=col, sticky="ew", padx=(0, 4), pady=(0, 8))

        for row, device in enumerate(RBC_ACTION_NAMES, start=1):
            tk.Label(
                table, text=device, bg=BG_CARD2, fg=TEXT,
                font=F_DEVICE, anchor="w", padx=14, pady=22
            ).grid(row=row, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))

            clean_lbl = tk.Label(
                table, text="0.000", bg=BLUE_LIGHT, fg=BLUE,
                font=F_TABLE_V, anchor="center", padx=8, pady=22
            )
            clean_lbl.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.clean_labels[device] = clean_lbl

            pert_lbl = tk.Label(
                table, text="0.000", bg=ORANGE_LIGHT, fg=ORANGE,
                font=F_TABLE_V, anchor="center", padx=8, pady=22
            )
            pert_lbl.grid(row=row, column=2, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.perturbed_labels[device] = pert_lbl

            diff_lbl = tk.Label(
                table, text="D 0.000", bg=BG_CARD2, fg=GREEN,
                font=F_DIFF, anchor="center", padx=8, pady=22
            )
            diff_lbl.grid(row=row, column=3, sticky="ew", padx=(0, 4), pady=(0, 8))
            self.diff_labels[device] = diff_lbl

        # Legenda
        legend = tk.Frame(center, bg=BG_CARD)
        legend.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        for color, label, desc in [
            (GREEN,  "D < 0.05",    "robusta"),
            (ORANGE, "D 0.05-0.20", "sensibile"),
            (RED,    "D > 0.20",    "fragile"),
        ]:
            tk.Label(legend, text=label, bg=BG_CARD, fg=color,
                     font=F_BODY_B).pack(side="left", padx=(0, 6))
            tk.Label(legend, text=desc, bg=BG_CARD, fg=TEXT_MUTED,
                     font=F_BODY).pack(side="left", padx=(0, 24))

    # ── Pannello destro ────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="Osservazioni correnti", bg=BG_CARD, fg=TEXT,
                 font=F_SECTION, anchor="w"
                 ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))

        obs_frame = tk.Frame(right, bg=BG_CARD)
        obs_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        obs_frame.columnconfigure(1, weight=1)

        for i, name in enumerate(RBC_OBSERVATION_NAMES):
            tk.Label(
                obs_frame, text=name, bg=BG_CARD, fg=TEXT_MUTED,
                font=F_OBS_NAME, anchor="w"
            ).grid(row=i, column=0, sticky="w", padx=(0, 16), pady=3)

            val_lbl = tk.Label(
                obs_frame, text="---", bg=BG_CARD, fg=TEXT,
                font=F_OBS_VAL, anchor="e"
            )
            val_lbl.grid(row=i, column=1, sticky="e", pady=3)
            self.obs_labels[name] = val_lbl

        tk.Label(right, text="Console", bg=BG_CARD, fg=TEXT,
                 font=F_SECTION, anchor="w"
                 ).grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 6))

        self.console = tk.Text(
            right, wrap="word", state="disabled",
            bg=CONSOLE_BG, fg=CONSOLE_FG, relief="flat", bd=0,
            font=F_CONSOLE, padx=10, pady=10
        )
        self.console.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))

        append_text(self.console, "GUI pronta")
        append_text(self.console, "Per la policy neurale premi prima Train model")
        append_text(self.console, "Imposta una perturbazione e premi Applica per testare la robustezza")

    def mainloop(self):
        self.root.mainloop()


def main():
    root = tk.Tk()
    app  = App(root)
    app.mainloop()


if __name__ == "__main__":
    main()