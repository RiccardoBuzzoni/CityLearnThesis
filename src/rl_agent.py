"""
rl_agent.py
-----------
CityLearn-compatible wrapper for loading and running the Omnisafe actor
network produced by train.py.

Design contract
---------------
OmnisafeActorAgent mirrors the interface of LinearRegressorAgent in
evaluate.py: both classes expose a single

    predict(observations: List[List[float]]) -> List[List[float]]

method that accepts raw CityLearn observations and returns actions in the
physical device bounds expected by CityLearnEnv.step().  This makes the
RL agent a drop-in replacement for the other agents in evaluate_agent()
and in the GUI dispatch without requiring any changes to those call sites.

Normalisation
-------------
During training, CityLearnCMDP wraps the inner environment with
NormalizedObservationWrapper before passing observations to the actor.
At evaluation time, evaluate.py creates a plain CityLearnEnv (no wrapper),
so raw observations would be fed to the actor — producing incorrect
predictions because the actor has only ever seen normalised values.

To bridge this gap, OmnisafeActorAgent creates its own
NormalizedObservationWrapper around the evaluation environment and calls
its observation() method to normalise raw observations before inference.
CityLearn's normalisation is bound-based (fixed from the observation space,
not running statistics), so it is safe to apply without stepping through
the environment first.

Action rescaling
----------------
The Omnisafe actor outputs values in [-1, 1] (tanh-squashed).
_rescale() maps these back to CityLearn's physical device bounds using
the same linear formula as CityLearnCMDP._rescale_action():

    scaled_i = low_i + (action_i + 1) / 2 * (high_i - low_i)

For storage devices (bounds already [-1, 1]) this is the identity.
For cooling_device (bounds [0, 1]) it maps -1 → 0 and +1 → 1.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.citylearn import CityLearnEnv
from CityLearn.citylearn.wrappers import NormalizedObservationWrapper


class OmnisafeActorAgent:
    """
    Evaluation wrapper for an Omnisafe actor saved as a TorchScript file.

    Loads the actor network produced by train.py and exposes a predict()
    method that is interface-compatible with LinearRegressorAgent and
    AdvancedRBC, allowing all three agents to be compared through the
    same evaluate_agent() loop in evaluate.py.

    Parameters
    ----------
    fname : str
        Path to the TorchScript actor file (e.g. 'actor_net_standard.pt'
        or 'actor_net_custom.pt') saved by train.py.
    env : CityLearnEnv
        CityLearn environment used for evaluation.  It must share the same
        schema (observation space, action space) as the training environment
        so that normalisation bounds and action scaling are consistent.
    """

    def __init__(self, fname: str, env: CityLearnEnv) -> None:

        # Load the TorchScript actor. torch.jit.load restores the frozen
        # network without requiring Omnisafe to be importable at inference time.
        self.actor = torch.jit.load(fname)
        self.actor.eval()

        # Physical CityLearn action bounds used by _rescale() to map actor
        # outputs from [-1, 1] to the correct device ranges.
        self._action_space = env.action_space[0]

        # NormalizedObservationWrapper replicates the normalisation applied
        # inside CityLearnCMDP during training.  We use its observation()
        # method directly rather than stepping through it, which is safe
        # because CityLearn's normalisation is derived from fixed observation
        # space bounds, not from running episode statistics.
        self._normalizer = NormalizedObservationWrapper(env)

    def predict(self, observations: List[List[float]]) -> List[List[float]]:
        """
        Predict actions from raw CityLearn observations.

        Applies the training-time normalisation, runs a deterministic forward
        pass through the actor, and rescales the output to physical device
        bounds before returning.

        Parameters
        ----------
        observations : List[List[float]]
            Raw observation vectors from CityLearnEnv.reset() or
            CityLearnEnv.step(), one inner list per building.
            With central_agent=True there is exactly one inner list.

        Returns
        -------
        List[List[float]]
            Action vectors in CityLearn's physical device bounds, one inner
            list per building, ready to be passed to CityLearnEnv.step().
        """
        # Normalise raw observations using the same wrapper as in training.
        # observation() transforms the full List[List[float]] in one call.
        norm_observations = self._normalizer.observation(observations)

        actions = []
        for i, norm_obs in enumerate(norm_observations):

            # Convert to tensor for the actor forward pass
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32)

            # Forward pass through the TorchScript actor.
            # The actor outputs [mean || log_std] concatenated along the last
            # dimension (gaussian_sac actor type, set in train.py model_cfgs).
            with torch.no_grad():
                output = self.actor(obs_t)

            # Split output into mean and log standard deviation
            mean, log_std = output.chunk(2, dim=-1)

            # Clamp log_std for numerical stability (mirrors Omnisafe internals)
            log_std = torch.clamp(log_std, min=-20.0, max=2.0)

            # Deterministic action: tanh(mean).
            # During training Omnisafe samples from the distribution; at
            # evaluation time we use the mode (tanh of the mean) for stable,
            # reproducible predictions without stochastic variance.
            action = torch.tanh(mean).detach().cpu().numpy()

            # Rescale from actor output range [-1, 1] to physical device bounds
            action = self._rescale(action)

            actions.append(action.tolist())

        return actions

    def _rescale(self, action: np.ndarray) -> np.ndarray:
        """
        Linearly rescale action values from [-1, 1] to each device's
        physical CityLearn bounds [low, high].

        Uses the same formula as CityLearnCMDP._rescale_action() to keep
        training and evaluation scaling perfectly consistent:

            scaled_i = low_i + (action_i + 1) / 2 * (high_i - low_i)

        Parameters
        ----------
        action : np.ndarray
            Actor output after tanh squashing, in [-1, 1].

        Returns
        -------
        np.ndarray
            Action values in CityLearn's physical device bounds.
        """
        # Guard against minor floating-point drift beyond [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        low  = self._action_space.low
        high = self._action_space.high
        scaled = low + (action + 1.0) / 2.0 * (high - low)

        return scaled.astype(np.float32)