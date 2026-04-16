import random
import tkinter as tk
from tkinter import ttk
from datetime import datetime

try:
    import linear_regressor as lr
except Exception:
    lr = None


BG_APP = "#ECECEC"
BG_CARD = "#FFFFFF"
BORDER = "#D8D8DC"
TEXT = "#1C1C1E"
TEXT_MUTED = "#6E6E73"
BLUE = "#007AFF"
BLUE_HOVER = "#0062CC"
RED = "#FF3B30"
RED_HOVER = "#D9362B"
GREEN = "#34C759"
GRAY_BUTTON = "#E5E5EA"
GRAY_HOVER = "#DCDCE1"
CONSOLE_BG = "#FBFBFD"
CONSOLE_FG = "#1C1C1E"


DEFAULT_STATE = {
    "hour": 14.0,
    "indoor_temp": 27.0,
    "cooling_setpoint": 24.0,
    "outdoor_temp": 31.0,
    "outdoor_temp_predicted": 30.0,
    "cooling_demand": 0.65,
    "elec_price": 0.028,
    "carbon_intensity": 0.35,
    "solar_generation": 0.50,
    "occupant_count": 2.0,
    "electrical_storage_soc": 0.45,
    "dhw_storage_soc": 0.55,
    "dhw_demand": 0.40,
}

DEVICE_FEATURES = {
    "cooling_device": [
        "hour",
        "indoor_temp",
        "cooling_setpoint",
        "outdoor_temp",
        "outdoor_temp_predicted",
        "cooling_demand",
        "elec_price",
        "carbon_intensity",
        "solar_generation",
        "occupant_count",
    ],
    "dhw_storage": [
        "hour",
        "dhw_storage_soc",
        "dhw_demand",
        "elec_price",
        "solar_generation",
    ],
    "electrical_storage": [
        "hour",
        "electrical_storage_soc",
        "solar_generation",
        "elec_price",
        "carbon_intensity",
    ],
}

PERTURBATION_TARGETS = [
    "outdoor_temp",
    "solar_generation",
    "elec_price",
    "carbon_intensity",
    "occupant_count",
    "cooling_demand",
    "dhw_demand",
]


def append_text(text_widget, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    text_widget.config(state="normal")
    text_widget.insert("end", f"[{timestamp}] {message}\n")
    text_widget.see("end")
    text_widget.config(state="disabled")


def create_button(parent, text, command, bg, fg="white", active_bg=None, width=16):
    if active_bg is None:
        active_bg = bg

    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        font=("Helvetica", 10, "bold"),
        relief="flat",
        bd=0,
        padx=12,
        pady=10,
        width=width,
        cursor="hand2",
    )


def clamp_action(target, value):
    value = float(value)
    if target == "cooling_device":
        return max(0.0, min(1.0, value))
    return max(-1.0, min(1.0, value))


def apply_perturbation(state, feature_name, strength):
    new_state = dict(state)

    if feature_name == "outdoor_temp":
        new_state[feature_name] += strength * 6.0
    elif feature_name == "solar_generation":
        new_state[feature_name] = max(0.0, min(1.0, new_state[feature_name] + strength * 0.35))
    elif feature_name == "elec_price":
        new_state[feature_name] = max(0.0, new_state[feature_name] + strength * 0.015)
    elif feature_name == "carbon_intensity":
        new_state[feature_name] = max(0.0, min(1.0, new_state[feature_name] + strength * 0.20))
    elif feature_name == "occupant_count":
        new_state[feature_name] = max(0.0, round(new_state[feature_name] + strength * 2.0))
    elif feature_name == "cooling_demand":
        new_state[feature_name] = max(0.0, min(1.0, new_state[feature_name] + strength * 0.25))
    elif feature_name == "dhw_demand":
        new_state[feature_name] = max(0.0, min(1.0, new_state[feature_name] + strength * 0.25))

    return new_state


class SimpleRBCPolicy:
    def predict(self, state):
        return {
            "cooling_device": self.predict_cooling(state),
            "dhw_storage": self.predict_dhw(state),
            "electrical_storage": self.predict_battery(state),
        }

    def predict_cooling(self, s):
        if s["occupant_count"] <= 0:
            return 0.0

        if s["indoor_temp"] > s["cooling_setpoint"] + 2.0:
            base = 0.75
        elif s["cooling_demand"] >= 0.7:
            base = 0.66
        elif s["cooling_demand"] >= 0.3:
            base = 0.4
        else:
            base = 0.2

        if s["elec_price"] > 0.03:
            base -= 0.10
        if s["solar_generation"] > 0.6:
            base += 0.10
        if s["carbon_intensity"] > 0.40:
            base -= 0.05

        return clamp_action("cooling_device", base)

    def predict_dhw(self, s):
        soc = s["dhw_storage_soc"]
        hour = s["hour"]
        demand = s["dhw_demand"]
        price = s["elec_price"]
        solar = s["solar_generation"]

        if 6 <= hour <= 9 or 18 <= hour <= 22:
            if soc >= 0.7:
                action = -0.6 if (demand >= 0.6 or price > 0.03) else -0.2
            elif soc >= 0.4:
                action = -0.4 if (demand >= 0.6 or price > 0.03) else -0.15
            else:
                action = -0.01
        else:
            if soc >= 0.7:
                action = 0.4 if (solar > 0.6 and price < 0.03) else 0.0
            elif soc >= 0.4:
                action = 0.65 if (solar > 0.6 and price < 0.03) else 0.15
            else:
                action = 0.8 if (solar > 0.6 and price < 0.03) else 0.4

        return clamp_action("dhw_storage", action)

    def predict_battery(self, s):
        soc = s["electrical_storage_soc"]
        hour = s["hour"]
        solar = s["solar_generation"]
        price = s["elec_price"]
        peak = 8 <= hour <= 10 or 18 <= hour <= 21

        if peak:
            if soc >= 0.5:
                action = -0.5 if solar > 0.6 else -0.3
            elif soc >= 0.2:
                action = -0.1
            else:
                action = 0.0
        else:
            if soc >= 0.5:
                action = 0.5 if solar > 0.6 else 0.3
            elif soc >= 0.2:
                action = 0.7 if solar > 0.6 else 0.5
            else:
                action = 0.9 if solar > 0.6 else 0.6

            if price > 0.03:
                action -= 0.15

        return clamp_action("electrical_storage", action)


class SimpleNeuralPolicy:
    def __init__(self):
        self.available = lr is not None
        self.models = {}
        self.feature_order = {}
        self.mean = None
        self.std = None
        self.is_ready = False

    def train_if_possible(self):
        if not self.available:
            return False, "linear_regressor.py non importabile"

        try:
            self.models, self.feature_order, self.mean, self.std = lr.main()
            self.is_ready = True
            return True, "Policy neurale addestrata con successo"
        except Exception as e:
            return False, f"Errore training policy neurale: {e}"

    def predict(self, state):
        if not self.is_ready:
            return {
                "cooling_device": 0.0,
                "dhw_storage": 0.0,
                "electrical_storage": 0.0,
            }

        actions = {}
        feature_cols = list(lr.FEATURE_COLS)

        for target in lr.TARGET_COLS:
            features = self.feature_order[target]

            full_row = []
            for col in feature_cols:
                full_row.append(float(state.get(col, 0.0)))

            norm = []
            for i, value in enumerate(full_row):
                norm.append((value - float(self.mean[i])) / float(self.std[i]))

            selected = [norm[feature_cols.index(name)] for name in features]
            x = lr.torch.tensor([selected], dtype=lr.torch.float32)

            with lr.torch.no_grad():
                y = self.models[target](x).item()

            actions[target] = clamp_action(target, y)

        return actions


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RBC vs Neural Policy GUI")
        self.root.geometry("1400x820")
        self.root.minsize(1200, 720)
        self.root.configure(bg=BG_APP)

        self.state_vars = {}
        self.metric_labels = {}
        self.detail_labels = {}

        self.base_state = dict(DEFAULT_STATE)
        self.current_state = dict(DEFAULT_STATE)

        self.rbc_policy = SimpleRBCPolicy()
        self.neural_policy = SimpleNeuralPolicy()

        self.policy_mode = tk.StringVar(value="rbc")
        self.perturbation_feature = tk.StringVar(value="outdoor_temp")
        self.perturbation_strength = tk.DoubleVar(value=0.0)
        self.auto_perturb = tk.BooleanVar(value=False)

        self.simulation_running = False
        self.stop_requested = False
        self.step_count = 0

        self.build_layout()
        self.refresh_view(first_time=True)

    def build_layout(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        tk.Label(
            header,
            text="Confronto RBC vs Policy Neurale",
            bg=BG_CARD,
            fg=TEXT,
            font=("Helvetica", 20, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 4))

        tk.Label(
            header,
            text="GUI semplice per testare azioni e robustezza rispetto a perturbazioni",
            bg=BG_CARD,
            fg=TEXT_MUTED,
            font=("Helvetica", 11),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 14))

        content = tk.Frame(self.root, bg=BG_APP)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.columnconfigure(2, weight=3)

        self.build_left_panel(content)
        self.build_center_panel(content)
        self.build_right_panel(content)

    def build_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(
            left,
            text="Controlli",
            bg=BG_CARD,
            fg=TEXT,
            font=("Helvetica", 15, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(16, 10))

        mode_box = tk.LabelFrame(left, text="Controller", bg=BG_CARD, fg=TEXT)
        mode_box.pack(fill="x", padx=16, pady=(0, 12))

        tk.Radiobutton(
            mode_box,
            text="RBC",
            variable=self.policy_mode,
            value="rbc",
            command=self.refresh_view,
            bg=BG_CARD,
        ).pack(anchor="w", padx=10, pady=(8, 4))

        tk.Radiobutton(
            mode_box,
            text="Policy neurale",
            variable=self.policy_mode,
            value="neural",
            command=self.refresh_view,
            bg=BG_CARD,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        model_box = tk.LabelFrame(left, text="Policy neurale", bg=BG_CARD, fg=TEXT)
        model_box.pack(fill="x", padx=16, pady=(0, 12))

        create_button(
            model_box,
            "Train model",
            self.train_neural_policy,
            BLUE,
            active_bg=BLUE_HOVER,
            width=18,
        ).pack(anchor="w", padx=10, pady=(10, 8))

        self.neural_status_label = tk.Label(
            model_box,
            text="Stato: non addestrata",
            bg=BG_CARD,
            fg=TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        self.neural_status_label.pack(fill="x", padx=10, pady=(0, 10))

        perturb_box = tk.LabelFrame(left, text="Perturbazioni", bg=BG_CARD, fg=TEXT)
        perturb_box.pack(fill="x", padx=16, pady=(0, 12))

        ttk.Combobox(
            perturb_box,
            textvariable=self.perturbation_feature,
            values=PERTURBATION_TARGETS,
            state="readonly",
        ).pack(fill="x", padx=10, pady=(10, 8))

        tk.Scale(
            perturb_box,
            from_=-1.0,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.perturbation_strength,
            bg=BG_CARD,
            highlightthickness=0,
        ).pack(fill="x", padx=10)

        tk.Checkbutton(
            perturb_box,
            text="Perturbazione casuale a ogni step",
            variable=self.auto_perturb,
            bg=BG_CARD,
        ).pack(anchor="w", padx=10, pady=(4, 8))

        create_button(
            perturb_box,
            "Applica perturbazione",
            self.apply_current_perturbation,
            GRAY_BUTTON,
            fg=TEXT,
            active_bg=GRAY_HOVER,
            width=18,
        ).pack(anchor="w", padx=10, pady=(0, 6))

        create_button(
            perturb_box,
            "Reset stato",
            self.reset_state,
            GRAY_BUTTON,
            fg=TEXT,
            active_bg=GRAY_HOVER,
            width=18,
        ).pack(anchor="w", padx=10, pady=(0, 10))

        sim_box = tk.LabelFrame(left, text="Simulazione", bg=BG_CARD, fg=TEXT)
        sim_box.pack(fill="x", padx=16, pady=(0, 12))

        create_button(sim_box, "Step", self.run_single_step, GREEN, active_bg=GREEN, width=12).grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )

        create_button(sim_box, "Start", self.start_simulation, BLUE, active_bg=BLUE_HOVER, width=12).grid(
            row=0, column=1, padx=0, pady=10, sticky="w"
        )

        create_button(sim_box, "Stop", self.stop_simulation, RED, active_bg=RED_HOVER, width=12).grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="w"
        )

        create_button(
            sim_box,
            "Confronta",
            self.compare_policies,
            GRAY_BUTTON,
            fg=TEXT,
            active_bg=GRAY_HOVER,
            width=12,
        ).grid(row=1, column=1, padx=0, pady=(0, 10), sticky="w")

        inputs_box = tk.LabelFrame(left, text="Stato osservato", bg=BG_CARD, fg=TEXT)
        inputs_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        for name, value in DEFAULT_STATE.items():
            row = tk.Frame(inputs_box, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=4)

            tk.Label(
                row,
                text=name,
                width=20,
                anchor="w",
                bg=BG_CARD,
                fg=TEXT,
            ).pack(side="left")

            var = tk.StringVar(value=str(value))
            entry = tk.Entry(row, textvariable=var, relief="solid", bd=1)
            entry.pack(side="right", fill="x", expand=True)

            self.state_vars[name] = var

    def build_center_panel(self, parent):
        center = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        center.grid(row=0, column=1, sticky="nsew", padx=8)
        center.rowconfigure(2, weight=1)
        center.columnconfigure(0, weight=1)

        tk.Label(
            center,
            text="Azioni correnti",
            bg=BG_CARD,
            fg=TEXT,
            font=("Helvetica", 15, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        cards = tk.Frame(center, bg=BG_CARD)
        cards.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        cards.columnconfigure(2, weight=1)

        for i, device in enumerate(["cooling_device", "dhw_storage", "electrical_storage"]):
            card = tk.Frame(cards, bg="#F8F8FA", highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))

            tk.Label(
                card,
                text=device,
                bg="#F8F8FA",
                fg=TEXT_MUTED,
                font=("Helvetica", 10),
            ).pack(anchor="w", padx=12, pady=(10, 4))

            value_label = tk.Label(
                card,
                text="0.000",
                bg="#F8F8FA",
                fg=TEXT,
                font=("Helvetica", 18, "bold"),
            )
            value_label.pack(anchor="w", padx=12, pady=(0, 10))
            self.metric_labels[device] = value_label

        details_box = tk.LabelFrame(center, text="Dettaglio device", bg=BG_CARD, fg=TEXT)
        details_box.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        for device in ["cooling_device", "dhw_storage", "electrical_storage"]:
            section = tk.Frame(details_box, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
            section.pack(fill="x", padx=10, pady=8)

            tk.Label(
                section,
                text=device,
                bg=BG_CARD,
                fg=TEXT,
                font=("Helvetica", 12, "bold"),
            ).pack(anchor="w", padx=10, pady=(8, 4))

            detail = tk.Label(
                section,
                text="",
                justify="left",
                anchor="w",
                bg=BG_CARD,
                fg=TEXT_MUTED,
                font=("Helvetica", 10),
            )
            detail.pack(fill="x", padx=10, pady=(0, 8))
            self.detail_labels[device] = detail

    def build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(
            right,
            text="Console e confronto",
            bg=BG_CARD,
            fg=TEXT,
            font=("Helvetica", 15, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        self.console = tk.Text(
            right,
            wrap="word",
            state="disabled",
            bg=CONSOLE_BG,
            fg=CONSOLE_FG,
            relief="flat",
            bd=0,
            font=("Menlo", 11),
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        append_text(self.console, "GUI pronta")
        append_text(self.console, "Seleziona RBC o policy neurale")
        append_text(self.console, "Usa le perturbazioni per mostrare robustezza")

    def read_state_from_inputs(self):
        state = {}
        for name, var in self.state_vars.items():
            try:
                state[name] = float(var.get())
            except ValueError:
                state[name] = DEFAULT_STATE[name]
                var.set(str(DEFAULT_STATE[name]))

        self.current_state = state
        return state

    def write_state_to_inputs(self, state):
        for name, value in state.items():
            self.state_vars[name].set(str(round(float(value), 4)))

    def get_active_policy(self):
        if self.policy_mode.get() == "neural":
            return self.neural_policy, "Policy neurale"
        return self.rbc_policy, "RBC"

    def compute_actions(self, state):
        policy, _ = self.get_active_policy()
        return policy.predict(state)

    def refresh_view(self, first_time=False):
        state = self.read_state_from_inputs()
        actions = self.compute_actions(state)

        for name, value in actions.items():
            self.metric_labels[name].config(text=f"{value:.3f}")

        for device in ["cooling_device", "dhw_storage", "electrical_storage"]:
            features = DEVICE_FEATURES[device]
            rows = [f"Azione: {actions[device]:.3f}", "Feature usate:"]
            for feature in features:
                rows.append(f"- {feature}: {state[feature]:.3f}")
            self.detail_labels[device].config(text="\n".join(rows))

        if not first_time:
            controller = "policy neurale" if self.policy_mode.get() == "neural" else "RBC"
            append_text(self.console, f"Vista aggiornata con {controller}")

    def train_neural_policy(self):
        append_text(self.console, "Avvio training policy neurale...")
        ok, message = self.neural_policy.train_if_possible()
        self.neural_status_label.config(text=f"Stato: {message}")
        append_text(self.console, message)
        self.refresh_view()

    def apply_current_perturbation(self):
        state = self.read_state_from_inputs()
        feature = self.perturbation_feature.get()
        strength = self.perturbation_strength.get()

        new_state = apply_perturbation(state, feature, strength)
        self.current_state = new_state
        self.write_state_to_inputs(new_state)

        append_text(
            self.console,
            f"Perturbazione applicata su {feature} con intensità {strength:+.1f}"
        )
        self.refresh_view()

    def reset_state(self):
        self.current_state = dict(self.base_state)
        self.write_state_to_inputs(self.current_state)
        append_text(self.console, "Stato riportato ai valori iniziali")
        self.refresh_view()

    def random_perturbation(self):
        feature = random.choice(PERTURBATION_TARGETS)
        strength = random.choice([-0.5, -0.3, 0.3, 0.5])
        self.perturbation_feature.set(feature)
        self.perturbation_strength.set(strength)
        self.apply_current_perturbation()

    def run_single_step(self):
        if self.auto_perturb.get():
            self.random_perturbation()
        else:
            self.read_state_from_inputs()

        self.step_count += 1
        actions = self.compute_actions(self.current_state)
        _, policy_name = self.get_active_policy()

        append_text(self.console, f"Step {self.step_count} | controller: {policy_name}")
        append_text(
            self.console,
            f"Azioni -> cooling: {actions['cooling_device']:.3f}, "
            f"dhw: {actions['dhw_storage']:.3f}, "
            f"battery: {actions['electrical_storage']:.3f}"
        )
        self.refresh_view()

    def start_simulation(self):
        if self.simulation_running:
            append_text(self.console, "Simulazione già in esecuzione")
            return

        self.simulation_running = True
        self.stop_requested = False
        append_text(self.console, "Simulazione avviata")
        self.schedule_step()

    def schedule_step(self):
        if self.stop_requested:
            self.simulation_running = False
            append_text(self.console, "Simulazione fermata")
            return

        self.run_single_step()
        self.root.after(1200, self.schedule_step)

    def stop_simulation(self):
        self.stop_requested = True
        append_text(self.console, "Richiesta di stop inviata")

    def compare_policies(self):
        state = self.read_state_from_inputs()
        rbc_actions = self.rbc_policy.predict(state)
        neural_actions = self.neural_policy.predict(state)

        append_text(self.console, "Confronto RBC vs policy neurale")
        for device in ["cooling_device", "dhw_storage", "electrical_storage"]:
            diff = abs(rbc_actions[device] - neural_actions[device])
            append_text(
                self.console,
                f"{device}: RBC={rbc_actions[device]:.3f} | "
                f"Neural={neural_actions[device]:.3f} | "
                f"|diff|={diff:.3f}"
            )


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()