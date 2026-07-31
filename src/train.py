"""
train.py
--------
Training script for the Omnisafe constrained RL agent on CityLearn.

Two experiments are launched sequentially, one for each entry in
REWARD_CONFIGS, following the tutor's indication to compare a standard
reward against a custom multi-objective one:

  'standard'  — native CityLearn reward (net electricity consumption).
                Discomfort is handled exclusively by the CMDP cost
                channel defined in citylearn_cmdp.py.

  'custom'    — Challenge2023Reward (from Lorenzo's reference script)
                with the discomfort weight w3 set to 0.0.  The other
                weights (electricity cost w1, carbon w2) are preserved.
                Setting w3=0.0 prevents double-counting: discomfort is
                already captured by the cost channel, so keeping it in
                the reward as well would penalise it twice and confuse
                the Lagrangian optimisation.

Both experiments run on the same training schema, seed, and algorithm
so that their results are directly comparable for the thesis evaluation.

Output layout
-------------
results/
└── experiments/
    └── {algo}_{name}/
        ├── seed{n}/
        │   ├── actor_net_standard.pt   ← TorchScript actor (experiment 1)
        │   └── actor_net_custom.pt     ← TorchScript actor (experiment 2)
        ├── schemas/
        │   ├── schema.json             ← full original schema
        │   ├── train_schema.json       ← first frac% of simulation window
        │   └── eval_schema.json        ← remaining (1-frac)% of window
        └── train_config.yaml           ← all CLI arguments for reproducibility

Usage
-----
    # Default run (PPOLag, cpu, 200 episodes, no wandb)
    python train.py --episodes 200

    # With wandb logging and a custom experiment name
    python train.py --episodes 500 --wandb --name thesis_run1

    # GPU training with interactive schema selection
    python train.py --device cuda --custom --episodes 300
"""

from __future__ import annotations

import copy
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing citylearn_cmdp triggers the @env_register decorator, which
# registers 'CityLearn-v0' in Omnisafe's environment registry.
# This import MUST occur before omnisafe.Agent() is called.
import citylearn_cmdp  # noqa: F401  (side-effect import)

import omnisafe.omnisafe as omnisafe

from config import TrainConfig
from utils import default_env_config, select_env_config


# ── Reward configurations ─────────────────────────────────────────────────────

# Keys map to the reward_tag used for naming output files and W&B runs.
# Values are either None (→ use CityLearn's native reward unchanged) or
# a schema-compatible reward_function dict to inject before training.
REWARD_CONFIGS: dict[str, dict | None] = {

    # Experiment 1: native CityLearn reward.
    # CityLearnEnv computes net electricity consumption by default; no schema
    # modification is needed.  Cost channel handles discomfort exclusively.
    'standard': None,

    # Experiment 2: multi-objective reward (Lorenzo's Challenge2023Reward),
    # but with the discomfort component zeroed out.
    # w1=electricity cost, w2=carbon emissions, w3=discomfort (→ 0), w4=unused.
    'custom': {
        'type': 'citylearn.reward_function.Challenge2023Reward',
        'attributes': {
            'band': 2.0,
            'lower_exponent': 2.0,
            'higher_exponent': 2.0,
            'weights': {
                'w1': 0.3,   # electricity cost
                'w2': 0.1,   # carbon emissions
                'w3': 0.0,   # discomfort — zeroed: handled by CMDP cost channel
                'w4': 0.0,   # unused
            },
        },
    },
}


# ── Schema helpers ────────────────────────────────────────────────────────────

def split_schema(schema: dict, frac: float) -> tuple[dict, dict]:
    """
    Split a CityLearn schema's simulation window into training and evaluation
    portions while preserving temporal ordering.

    The split is applied to the simulation time indices, not to buildings,
    so no shuffling occurs.  The first ``frac`` of time steps go to training
    and the remaining ``(1 - frac)`` to evaluation.

    Parameters
    ----------
    schema : dict
        Full CityLearn schema dict as returned by ``default_env_config`` or
        ``select_env_config``.
    frac : float
        Fraction of the simulation window allocated to training (0 < frac < 1).

    Returns
    -------
    train_schema : dict
        Schema with ``simulation_end_time_step`` set to the split point.
    eval_schema : dict
        Schema with ``simulation_start_time_step`` set to the split point.
    """
    assert 0.0 < frac < 1.0, f"frac must be in (0, 1), got {frac}."

    start = schema['simulation_start_time_step']
    end   = schema['simulation_end_time_step']
    total = end - start
    split = start + int(total * frac)

    train_schema = copy.deepcopy(schema)
    train_schema['simulation_end_time_step'] = split

    eval_schema  = copy.deepcopy(schema)
    eval_schema['simulation_start_time_step'] = split

    return train_schema, eval_schema


def apply_reward_config(schema: dict, reward_cfg: dict | None) -> dict:
    """
    Return a copy of ``schema`` with ``reward_function`` set to ``reward_cfg``.

    If ``reward_cfg`` is None the schema is returned unchanged (deep-copied),
    preserving whatever reward function CityLearn would use by default.

    Parameters
    ----------
    schema : dict
        Base training schema (typically the output of ``split_schema``).
    reward_cfg : dict or None
        Reward function specification to inject, or None to keep the default.

    Returns
    -------
    dict
        Deep copy of ``schema`` with the reward function optionally replaced.
    """
    schema = copy.deepcopy(schema)
    if reward_cfg is not None:
        schema['reward_function'] = reward_cfg
    return schema


# ── Core training function ────────────────────────────────────────────────────

def main_omnisafe(
    conf: TrainConfig,
    schema: dict,
    reward_tag: str,
    exp_dir: str,
) -> None:
    """
    Run one Omnisafe training experiment and save the resulting actor network.

    Parameters
    ----------
    conf : TrainConfig
        Parsed training configuration (CLI arguments).
    schema : dict
        Training schema, already modified with the desired reward function
        and restricted to the training time window.
    reward_tag : str
        Short label for this experiment ('standard' or 'custom').
        Used to name the output actor file and W&B run.
    exp_dir : str
        Root directory for this experiment group.  A ``seed{n}/`` sub-folder
        is created here to hold the saved actor and other artefacts.
    """
    args = conf.args

    # Create the per-seed output directory
    seed_dir = os.path.join(exp_dir, f'seed{args.seed}')
    os.makedirs(seed_dir, exist_ok=True)

    # Derive training length from the schema's simulation window
    ep_steps = (
        schema['simulation_end_time_step']
        - schema['simulation_start_time_step']
    )
    assert args.episodes > 0, f"Invalid number of episodes: {args.episodes}."
    train_steps = args.episodes * ep_steps

    print(f"\n{'=' * 60}")
    print(f"  Experiment : {args.algo} | reward={reward_tag} | seed={args.seed}")
    print(f"  Episodes   : {args.episodes}  |  Steps/ep : {ep_steps}  |  Total : {train_steps}")
    print(f"{'=' * 60}\n")

    # ── Omnisafe configuration dict ───────────────────────────────────────────
    custom_cfgs: dict = {
        'seed': args.seed,

        'train_cfgs': {
            'total_steps': train_steps,
            'device': args.device,
        },

        'algo_cfgs': {
            # steps_per_epoch aligns one Omnisafe "epoch" with one CityLearn
            # episode, so logged metrics map directly to episode boundaries.
            'steps_per_epoch': ep_steps,

            # IMPORTANT: must be False.  NormalizedObservationWrapper inside
            # CityLearnCMDP already normalises observations; a second pass
            # here would distort them.
            'obs_normalize': False,
        },

        'model_cfgs': {
            # gaussian_sac actor outputs [mean, log_std] concatenated along the
            # last dimension.  The OmnisafeActorAgent in rl_agent.py relies on
            # this format when loading the TorchScript actor for evaluation.
            'actor_type': 'gaussian_sac',
        },

        'logger_cfgs': {
            'use_wandb': args.wandb,
            'wandb_project': args.project,
            'entity': args.entity,
            # Tag identifies both the reward variant and the seed in the W&B UI
            'tag': f'{args.algo}_{reward_tag}_seed{args.seed}',
            'mode': 'online',
        },

        # env_cfgs are forwarded verbatim to CityLearnCMDP.__init__() as
        # **kwargs.  'cost_fn' is intentionally absent: cost is computed
        # internally by CityLearnCMDP._compute_cost() from building state.
        'env_cfgs': {
            'schema': schema,
            'central_agent': True,
        },
    }

    # Optionally enable CityLearn's built-in episode rendering
    if args.render:
        custom_cfgs['env_cfgs'].update({
            'render_mode': 'during',
            'render_directory': os.path.join(seed_dir, 'render'),
            'render_session_name': f'{args.algo}_{reward_tag}_seed{args.seed}',
        })

    # ── Training ──────────────────────────────────────────────────────────────
    # 'CityLearn-v0' was registered by the `import citylearn_cmdp` at the top
    # of this file.  Omnisafe resolves the string to CityLearnCMDP via its
    # internal ENV_REGISTRY.
    agent = omnisafe.Agent(args.algo, 'CityLearn-v0', custom_cfgs=custom_cfgs)
    agent.learn()

     # ── Save actor network ────────────────────────────────────────────────────
    # agent.agent is the algorithm object (e.g. PPOLag).
    # _actor_critic.actor.net is the raw MLP that maps obs → [mean || log_std].
    # TorchScript serialisation makes the net loadable without importing
    # Omnisafe at evaluation time (only torch.jit.load is needed).
    #
    # _actor_critic is not declared in BaseAlgo's public type stubs, so the
    # type checker cannot see it statically.  # type: ignore[attr-defined]
    # suppresses that specific error without affecting runtime behaviour.
    # The try/except still catches genuine AttributeError at runtime
    # (e.g. if the internal structure changes across Omnisafe versions).
    try:
        actor_net = agent.agent._actor_critic.actor.net.cpu()  # type: ignore[attr-defined]
        actor_script = torch.jit.script(actor_net)
        actor_path = os.path.join(seed_dir, f'actor_net_{reward_tag}.pt')
        actor_script.save(actor_path)
        print(f"Actor network saved → {actor_path}")
    except AttributeError as exc:
        # Emit a warning rather than crashing so that training logs and
        # W&B metrics are not lost if the actor path changes in a future
        # Omnisafe version.
        print(
            f"[WARN] Could not serialise actor network: {exc}\n"
            f"       Inspect agent.agent._actor_critic structure for '{args.algo}'."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Parse CLI arguments (TrainConfig extends Config from utils.py)
    conf = TrainConfig()
    args = conf.args

    # ── Experiment directory ──────────────────────────────────────────────────
    # Use a user-supplied name or fall back to the current timestamp.
    # Colons are replaced with dashes for Windows/WSL2 filesystem compatibility.
    exp_name = (
        args.name
        if args.name is not None
        else datetime.now().strftime('%d-%m-%y_%H-%M')
    )
    exp_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'experiments',
        f'{args.algo}_{exp_name}',
    )
    os.makedirs(exp_dir, exist_ok=True)

    # ── Schema loading ────────────────────────────────────────────────────────
    # --custom triggers an interactive terminal menu (select_env_config) for
    # choosing buildings, observations, and actions.
    # Without --custom, the default config (Building_1, all active obs/act)
    # is used via default_env_config.
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)

    # ── Train / eval split ────────────────────────────────────────────────────
    # The split operates on the simulation time window, not on buildings.
    # Temporal order is preserved: training always precedes evaluation.
    train_schema, eval_schema = split_schema(schema, args.frac)

    n_train = (
        train_schema['simulation_end_time_step']
        - train_schema['simulation_start_time_step']
    )
    n_eval  = (
        eval_schema['simulation_end_time_step']
        - eval_schema['simulation_start_time_step']
    )
    print(f"Schema split: {n_train} train steps | {n_eval} eval steps (frac={args.frac})")

    # ── Persist schemas ───────────────────────────────────────────────────────
    # All three schemas are saved as JSON so that evaluation can be run later
    # on exactly the same data split without re-running this script.
    schema_dir = os.path.join(exp_dir, 'schemas')
    os.makedirs(schema_dir, exist_ok=True)
    for name, s in [
        ('schema',       schema),
        ('train_schema', train_schema),
        ('eval_schema',  eval_schema),
    ]:
        with open(os.path.join(schema_dir, f'{name}.json'), 'w') as f:
            json.dump(s, f, indent=2)
    print(f"Schemas saved → {schema_dir}/")

    # ── Two-experiment loop ───────────────────────────────────────────────────
    # Each iteration applies a different reward function to a fresh deep copy
    # of the training schema, then runs a full training session.
    # The loop is sequential so GPU memory is not shared across experiments.
    for reward_tag, reward_cfg in REWARD_CONFIGS.items():
        experiment_schema = apply_reward_config(train_schema, reward_cfg)
        main_omnisafe(conf, experiment_schema, reward_tag, exp_dir)

    # ── Persist training configuration ───────────────────────────────────────
    # Written last so it reflects the exact args used for both experiments.
    conf.save_yaml(exp_dir)