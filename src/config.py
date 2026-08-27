"""
config.py
---------
Training configuration for the Omnisafe constrained RL agent on CityLearn.

TrainConfig extends the base Config class from utils.py with arguments
that are specific to the RL training phase (device, train/test split
fraction, experiment naming, rendering toggle) and adds a save_yaml()
method for experiment reproducibility.

Inheritance chain
-----------------
Config (utils.py)
    └── TrainConfig (this file)

How the MRO makes the extension work
-------------------------------------
Config.__init__() calls ``self._add_args()``.  Because Python uses MRO,
that call resolves to TrainConfig._add_args() even when triggered from
Config.__init__().  TrainConfig._add_args() calls super()._add_args()
first (adding the base arguments) and then appends the training-specific
ones.  By the time Config.__init__() calls parse_args(), all arguments —
base + extra — are already registered on the parser.  No double-parsing
occurs.

--tag and reward selection
--------------------------
The --tag argument (inherited from Config) serves a dual purpose:
  1. W&B run tagging for experiment tracking.
  2. Reward-function selection key in train.py.

Expected values for reward selection:
  'standard'   → native CityLearn reward (net electricity consumption).
  'custom'     → Challenge2023Reward with w3=0.0, so that discomfort is
                 removed from the reward and handled exclusively by the
                 CMDP cost channel.  This avoids double-counting.

Usage example
-------------
    conf = TrainConfig()
    args = conf.args          # standard argparse Namespace
    ...                       # run training
    conf.save_yaml(exp_dir)   # persist config for reproducibility
"""

import os
import yaml

from utils import Config


class TrainConfig(Config):
    """
    Omnisafe training configuration.

    Inherits all base arguments from Config and adds the ones listed below.
    Calling ``TrainConfig()`` parses sys.argv immediately, exactly like
    the parent class.

    Additional arguments
    --------------------
    --device : str, default='cpu'
        PyTorch device for the Omnisafe actor/critic networks.
        Use 'cuda' when a GPU is available to speed up training.

    --frac : float, default=0.8
        Fraction of the schema's simulation time window allocated to
        training.  The remaining (1 - frac) is reserved for evaluation.
        The split preserves temporal ordering (no shuffling), so the
        agent is tested on data it has never seen.

    --name : str, default=None
        Human-readable label for the experiment.  Currently accepted but
        not yet wired into the output path — train.py always writes to
        a fixed ``results/rl_model/`` regardless of this value.

    --render : flag, default=False
        Enable CityLearn episode rendering during training.
        Saves frames to disk; significantly increases runtime.
        Recommended only for qualitative debugging of learned behaviour.
    """

    def __init__(self) -> None:
        # Delegates to Config.__init__(), which will call self._add_args()
        # (resolved to TrainConfig._add_args() via MRO) and then parse_args().
        super().__init__()

    # ── Argument registration ─────────────────────────────────────────────────

    def _add_args(self) -> None:
        # Register the base arguments first (seed, data, algo, episodes, wandb …)
        super()._add_args()

        # Override the default algorithm to a Lagrangian-constrained variant.
        # The base Config defaults to plain 'PPO' (reward-only, no constraints).
        # PPOLag adds a learnable Lagrange multiplier that enforces the CMDP
        # cost budget defined in citylearn_cmdp.py.
        # Other valid Omnisafe choices: 'CPO', 'PCPO', 'SACLag'.
        self.parser.set_defaults(algo='PPOLag')

        # Override the default tag to reflect the constrained RL use-case.
        # 'standard' maps to the native CityLearn reward in train.py.
        # Can be overridden at runtime: --tag custom
        self.parser.set_defaults(tag='standard')

        # ── Training-specific arguments ───────────────────────────────────────

        self.parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            choices=['cpu', 'cuda'],
            help="Device used for Omnisafe actor/critic networks (default: 'cpu')",
        )

        self.parser.add_argument(
            '--frac',
            type=float,
            default=0.8,
            help=(
                "Fraction of the simulation time window used for training. "
                "The remaining (1 - frac) is used for evaluation. "
                "Must be in (0, 1). (default: 0.8)"
            ),
        )

        self.parser.add_argument(
            '--name',
            type=str,
            default=None,
            help=(
                "Human-readable experiment name. Currently accepted but not "
                "yet wired into the output path — train.py always writes to "
                "a fixed results/rl_model/ regardless of this value."
            ),
        )

        self.parser.add_argument(
            '--render',
            action='store_true',
            help=(
                "Enable CityLearn episode rendering during training. "
                "Frames are saved to the experiment render/ sub-folder. "
                "Increases runtime significantly; use only for debugging."
            ),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_yaml(self, dir: str) -> None:
        """
        Persist all parsed arguments to a YAML file for reproducibility.

        The file is written as ``train_config.yaml`` inside ``dir``.
        Call this after the experiment directory has been created by
        train.py, so the file lands alongside the saved model checkpoints
        and schemas.

        Parameters
        ----------
        dir : str
            Destination directory.  Must already exist when this is called.

        Example output (train_config.yaml)
        ------------------------------------
        algo: PPOLag
        custom: false
        data: citylearn_challenge_2023_phase_1
        device: cpu
        entity: luca0
        episodes: 200
        frac: 0.8
        name: run_01
        project: citylearn_omnisafe
        render: false
        seed: 42
        tag: standard
        wandb: false
        wrapper: omnisafe
        """
        config_path = os.path.join(dir, 'train_config.yaml')

        # vars() converts the argparse Namespace to a plain dict.
        # sort_keys=True keeps the file deterministic across runs, making
        # diffs between experiment configs easy to read.
        with open(config_path, 'w') as f:
            yaml.dump(vars(self.args), f, default_flow_style=False, sort_keys=True)

        print(f"Training configuration saved → {config_path}")