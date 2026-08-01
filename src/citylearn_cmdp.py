"""
citylearn_cmdp.py
-----------------
Omnisafe-compatible CMDP wrapper for the CityLearn building energy
management environment.  Importing this module is enough to register
'CityLearn-v0' in Omnisafe's environment registry (the @env_register
decorator fires at import time), so the only required call in train.py is:

    import citylearn_cmdp                         # registers 'CityLearn-v0'
    agent = omnisafe.Agent('PPOLag', 'CityLearn-v0', custom_cfgs=cfgs)

Reward / Cost split
-------------------
The two signals that define a CMDP are kept strictly separate:

* Reward  — taken directly from CityLearn's built-in reward function.
            The choice (native net-consumption vs Challenge2023Reward with
            w3=0) is made at the *schema* level before environment creation,
            so this wrapper stays fully reward-agnostic.

* Cost    — computed here as the sum of upper-bound thermal-comfort
            violations across all active buildings at each time step:

                cost_b = max(0, T_indoor_b − (T_setpoint_b + band_b))

            A positive cost signals overheating. Omnisafe's constrained
            algorithms (PPOLag, CPO, …) keep the cumulative cost below a
            configurable safety budget via Lagrangian multipliers.

Observation normalisation
-------------------------
CityLearn's NormalizedObservationWrapper is applied *inside* this class.
Omnisafe's own normalisation layer must therefore be disabled:

    'algo_cfgs': {'obs_normalize': False}   # ← mandatory in custom_cfgs

Action space convention
-----------------------
Omnisafe actors squash outputs through tanh, producing values in [-1, 1].
step() rescales them linearly to each device's physical CityLearn bounds
before forwarding the action to the inner environment.  The mapping is:

    scaled = low + (action + 1) / 2 * (high − low)

For storage devices (bounds already [-1, 1]) this is the identity.
For cooling_device (bounds [0, 1]) this maps -1 → 0 and +1 → 1.
"""

from __future__ import annotations

import os
import sys
from typing import Any, ClassVar

import numpy as np
import torch
import gymnasium as gym

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, 'omnisafe'))

import types as _types

from omnisafe.envs.core import CMDP, env_register #type: ignore

from CityLearn.citylearn.citylearn import CityLearnEnv
from CityLearn.citylearn.wrappers import NormalizedObservationWrapper


# ── CMDP wrapper ──────────────────────────────────────────────────────────────

@env_register
class CityLearnCMDP(CMDP):
    """
    Omnisafe CMDP wrapper for the CityLearn environment.

    Parameters
    ----------
    env_id : str
        Must be ``'CityLearn-v0'``.
    **kwargs
        Forwarded from the ``env_cfgs`` block of Omnisafe's ``custom_cfgs``.
        Expected keys:

        schema : dict
            CityLearn schema dict (already loaded and optionally modified
            with the desired reward function before training starts).
        central_agent : bool
            Must be ``True`` — a single agent controls all buildings.
        render_mode : str, optional
            Passed to CityLearnEnv. Use ``'during'`` to enable rendering.
        render_directory : str, optional
        render_session_name : str, optional
        cost_fn : str, optional
            Accepted but intentionally ignored.  Cost is computed internally
            by ``_compute_cost()``; passing a string here is a remnant of the
            reference training script and has no effect.
    """

    # Omnisafe registry: links the string 'CityLearn-v0' to this class.
    # Must be a ClassVar so CMDP.support_envs() can read it before __init__.
    _support_envs: ClassVar[list[str]] = ['CityLearn-v0']

    # Tell Omnisafe not to add its own TimeLimit / AutoReset wrappers.
    # CityLearn signals episode termination via the standard `terminated`
    # flag, and Omnisafe's training loop calls reset() when that fires.
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = False

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        # Validates env_id against _support_envs; raises ValueError if unknown
        super().__init__(env_id)

        # Omnisafe expects these instance attributes on every CMDP subclass
        self._num_envs: int = 1
        self._metadata: dict[str, Any] = {}

        # ── Build the inner CityLearn environment ─────────────────────────────
        # Filter to the keys that CityLearnEnv's constructor accepts.
        # 'cost_fn' (and any other omnisafe-internal keys) are silently dropped.
        _CL_ACCEPTED_KEYS = (
            'schema', 'central_agent',
            'render_mode', 'render_directory', 'render_session_name',
        )
        cl_kwargs = {k: v for k, v in kwargs.items() if k in _CL_ACCEPTED_KEYS}
        self._env = CityLearnEnv(**cl_kwargs)

        # Wrap with CityLearn's built-in normalisation so that all observations
        # arrive in a consistent numerical range regardless of physical units.
        # Omnisafe's internal obs_normalize must be set to False to avoid a
        # second normalisation pass on top of this one.
        self._wrapped = NormalizedObservationWrapper(self._env)

        # ── Observation space ─────────────────────────────────────────────────
        # With central_agent=True, CityLearn exposes a single Box that
        # concatenates the normalised observations of all active buildings.
        # No explicit type annotation here: CMDP already declares this slot as
        # OmnisafeSpace; re-annotating it as gym.spaces.Box would cause an
        # invariance conflict in the type checker even though the runtime type
        # is compatible.
        self._observation_space = self._wrapped.observation_space[0]

        # ── Action space (uniform [-1, 1] for Omnisafe compatibility) ─────────
        # The actor outputs tanh-squashed values in [-1, 1]; physical device
        # bounds are stored separately and applied in _rescale_action().
        # Same annotation omission as _observation_space above.
        n_actions: int = self._env.action_space[0].shape[0]
        self._action_space = gym.spaces.Box(
            low   = -np.ones(n_actions, dtype=np.float32),
            high  =  np.ones(n_actions, dtype=np.float32),
            dtype = np.float32,
        )

        # Physical CityLearn action bounds, used for rescaling inside step()
        self._act_low:  np.ndarray = self._env.action_space[0].low.copy()
        self._act_high: np.ndarray = self._env.action_space[0].high.copy()

    # ── Gymnasium / Omnisafe interface ────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset the environment and return the initial normalised observation.

        Returns
        -------
        obs : torch.Tensor
            Initial observation, shape ``(n_obs,)``.
        info : dict
        """
        obs_list, info = self._wrapped.reset()

        # CityLearn returns [[obs_1, ..., obs_N]] — one inner list per agent.
        # With central_agent=True there is exactly one inner list.
        obs = torch.as_tensor(obs_list[0], dtype=torch.float32)
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,   # observation
        torch.Tensor,   # reward
        torch.Tensor,   # cost
        torch.Tensor,   # terminated
        torch.Tensor,   # truncated
        dict[str, Any],
    ]:
        """
        Step the environment with the given action.

        Parameters
        ----------
        action : torch.Tensor
            Action from the Omnisafe actor, in ``[-1, 1]``,
            shape ``(n_actions,)``.

        Returns
        -------
        obs        : torch.Tensor – normalised observation,    shape ``(n_obs,)``
        reward     : torch.Tensor – energy-efficiency signal,  shape ``(1,)``
        cost       : torch.Tensor – comfort violation ≥ 0,     shape ``(1,)``
        terminated : torch.Tensor – episode-end flag,          shape ``(1,)``
        truncated  : torch.Tensor – always False here,         shape ``(1,)``
        info       : dict
        """
        # 1. Convert Omnisafe tensor → numpy and rescale to physical bounds
        action_np     = action.detach().cpu().numpy()
        action_scaled = self._rescale_action(action_np)

        # 2. CityLearn expects List[List[float]] with central_agent=True
        cl_action = [action_scaled.tolist()]

        # 3. Step the wrapped (normalised) CityLearn environment
        obs_list, reward_val, terminated, truncated, info = (
            self._wrapped.step(cl_action)
        )

        # 4. Aggregate reward: CityLearn may return a float, a list of
        #    per-building values, or an ndarray depending on the version.
        #    np.asarray() normalises any of those to an ndarray, on which
        #    .sum() is unambiguous for both the runtime and the type checker.
        reward = float(np.asarray(reward_val, dtype=np.float32).sum())

        # 5. Compute thermal comfort cost from post-step building state
        cost = self._compute_cost()

        # 6. Pack all outputs as torch.Tensor (required by Omnisafe's CMDP API)
        obs_t        = torch.as_tensor(obs_list[0], dtype=torch.float32)
        reward_t     = torch.tensor([reward],        dtype=torch.float32)
        cost_t       = torch.tensor([cost],          dtype=torch.float32)
        terminated_t = torch.tensor([terminated],    dtype=torch.bool)
        truncated_t  = torch.tensor([truncated],     dtype=torch.bool)

        return obs_t, reward_t, cost_t, terminated_t, truncated_t, info

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed.

        CityLearn does not expose a public seed API, so this is a no-op
        kept for Omnisafe interface compliance.  Reproducibility can be
        achieved by setting the ``seed`` field in Omnisafe's ``custom_cfgs``,
        which seeds PyTorch and NumPy at the algorithm level.
        """
        pass

    def render(self) -> Any:
        """Delegate rendering to the inner CityLearn environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment cleanly."""
        self._env.close()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def max_episode_steps(self) -> int:
        """
        Episode length in time steps, derived from the schema's simulation
        window.  Used by Omnisafe for logging and epoch alignment even when
        ``need_time_limit_wrapper`` is False.
        """
        schema = self._env.schema
        return int(
            schema['simulation_end_time_step']
            - schema['simulation_start_time_step']
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Linearly rescale action values from ``[-1, 1]`` to each device's
        physical CityLearn bounds ``[self._act_low, self._act_high]``.

        The mapping is:

            scaled_i = low_i + (action_i + 1) / 2 * (high_i − low_i)

        For storage devices (bounds already ``[-1, 1]``) this is the identity.
        For cooling_device (bounds ``[0, 1]``) it maps ``-1 → 0`` and
        ``+1 → 1``.

        Parameters
        ----------
        action : np.ndarray
            Raw actor output clipped to ``[-1, 1]``, shape ``(n_actions,)``.

        Returns
        -------
        np.ndarray
            Rescaled values in CityLearn's physical device bounds.
        """
        # Clip to guard against minor floating-point drift beyond [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        scaled = (
            self._act_low
            + (action + 1.0) / 2.0 * (self._act_high - self._act_low)
        )
        return scaled.astype(np.float32)

    def _compute_cost(self) -> float:
        """
        Compute the thermal comfort cost for the current time step.

        The cost is the sum of upper-bound cooling violations across all
        active buildings:

            violation_b = max(0, T_indoor_b − (T_setpoint_b + band_b))

        A positive violation means the indoor temperature exceeded the
        comfort ceiling.  The cost is expressed in °C (per step) so it
        scales naturally with overheating severity.

        Lower-bound (heating) violations are not penalised here because
        the controlled devices in this project are cooling-only and the
        dataset is from a warm climate (CityLearn Challenge 2023, Phase 1).

        Returns
        -------
        float
            Total comfort cost ≥ 0.  Returns ``0.0`` if temperature
            history is not yet available (defensive guard for edge cases
            that may arise before the first full simulation step).
        """
        total_cost = 0.0

        for building in self._env.buildings:
            # Guard: temperature history is populated after env.step() but
            # may be empty immediately after reset() on some CityLearn versions
            if not building.indoor_dry_bulb_temperature:
                continue

            # Latest post-step indoor temperature and cooling setpoint
            indoor_temp = building.indoor_dry_bulb_temperature[-1]
            setpoint    = building.indoor_dry_bulb_temperature_cooling_set_point[-1]

            # comfort_band can be a scalar attribute or a time-indexed array
            band = building.comfort_band
            if hasattr(band, '__len__'):
                band = band[-1]

            # Accumulate upper-bound violation (overheating only)
            violation = indoor_temp - (setpoint + float(band))
            if violation > 0.0:
                total_cost += violation

        return total_cost