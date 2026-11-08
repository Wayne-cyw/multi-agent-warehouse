"""Scripted policies. They see only observations, never the environment's internals.

Every policy has the same two methods, so the demo and the evaluation share one loop:
`reset(seed)` starts an episode and `act(observations)` returns one action per robot.
"""

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from python_ta.contracts import check_contracts

from multi_agent_warehouse.rules import ACTION_DELTAS, DOWN, LEFT, RIGHT, STAY, UP

# Observation layout: [row, col, dr, dc, up_blocked, down_blocked, left_blocked, right_blocked].
_DR, _DC, _BLOCKED = 2, 3, 4

MODEL_PATH = Path("models/ppo_multi_agent_warehouse.zip")


@check_contracts
class Policy(ABC):
    """A rule that turns every robot's observation into an action.

    A policy owns a random generator, so a seeded `reset` makes a whole episode repeatable.

    Attributes:
        _rng: The generator behind the policy's random choices.

    Representation Invariants:
        - self._rng is not None
    """

    _rng: np.random.Generator

    def __init__(self) -> None:
        """Create the policy with an unseeded generator."""
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Start an episode with a fresh generator; a seed makes it repeatable.

        Preconditions:
            - seed is None or seed >= 0
        """
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """Pick one action per observed robot.

        Preconditions:
            - all(len(obs) >= 8 for obs in observations.values())
        """


@check_contracts
class RandomPolicy(Policy):
    """Every robot picks one of the five actions uniformly at random.

    Representation Invariants:
        - self._rng is not None
    """

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """A uniform action per robot, drawn in sorted id order so a seed repeats exactly.

        Preconditions:
            - all(len(obs) >= 8 for obs in observations.values())
        """
        return {
            agent: int(self._rng.integers(len(ACTION_DELTAS))) for agent in sorted(observations)
        }


@check_contracts
class GreedyPolicy(Policy):
    """Walk toward the target along the longer axis, sidestepping at random when stuck.

    Representation Invariants:
        - self._rng is not None
    """

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """One greedy action per robot, in sorted id order.

        Preconditions:
            - all(len(obs) >= 8 for obs in observations.values())
        """
        return {agent: self._act_one(observations[agent]) for agent in sorted(observations)}

    def _act_one(self, obs: np.ndarray) -> int:
        """Choose one robot's action from its own observation.

        Prefer the move along the larger offset (rows on a tie), then the other axis, then any
        open neighbour at random, and stay when boxed in or already on the target.

        Preconditions:
            - len(obs) >= 8
        """
        d_row, d_col = float(obs[_DR]), float(obs[_DC])
        if d_row == 0 and d_col == 0:
            return STAY
        row_move = DOWN if d_row > 0 else UP
        col_move = RIGHT if d_col > 0 else LEFT
        if abs(d_row) >= abs(d_col):
            preferred = [row_move, col_move if d_col != 0 else None]
        else:
            preferred = [col_move, row_move if d_row != 0 else None]

        for action in preferred:
            if action is not None and not _is_blocked(obs, action):
                return action
        open_moves = [a for a in (UP, DOWN, LEFT, RIGHT) if not _is_blocked(obs, a)]
        if not open_moves:
            return STAY
        return open_moves[int(self._rng.integers(len(open_moves)))]


def _is_blocked(obs: np.ndarray, action: int) -> bool:
    """Whether the observation flags the neighbour in the direction of `action` as blocked.

    Preconditions:
        - len(obs) >= 8
        - action in (UP, DOWN, LEFT, RIGHT)
    """
    return bool(obs[_BLOCKED + action - 1])


@check_contracts
class RLPolicy(Policy):
    """A trained PPO network shared by every robot, sampling its actions unless deterministic.

    Stable-Baselines3 comes from the optional `rl` extra, so it is imported only when a model is
    loaded.

    Attributes:
        _model: The loaded Stable-Baselines3 PPO model.
        _deterministic: Whether to take the most likely action instead of sampling.

    Representation Invariants:
        - self._rng is not None
    """

    _model: Any
    _deterministic: bool

    def __init__(
        self, model_path: str | Path = MODEL_PATH, deterministic: bool = False, algo: str = "ppo"
    ) -> None:
        """Load the model saved at `model_path`; `deterministic` picks the most likely action.

        Raises ValueError if `algo` is not "ppo".
        """
        super().__init__()
        if algo != "ppo":
            raise ValueError(f"unknown algo {algo!r}; expected 'ppo'")
        from stable_baselines3 import PPO

        self._model = PPO.load(model_path, device="cpu")
        self._deterministic = deterministic

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; a seed also fixes the network's sampling, so the episode repeats.

        Preconditions:
            - seed is None or seed >= 0
        """
        super().reset(seed)
        if seed is not None:
            # SB3 samples actions through torch's global generator, so that is the one to seed.
            import torch

            torch.manual_seed(seed)

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """One action per robot from a single batched forward pass, in sorted id order.

        Actions are sampled unless the policy is deterministic. Raises ValueError when the
        observations are not the size the model was trained on.

        Preconditions:
            - all(len(obs) >= 8 for obs in observations.values())
        """
        agents = sorted(observations)
        batch = np.stack([observations[agent] for agent in agents])
        expected = int(self._model.observation_space.shape[0])
        if batch.shape[1] != expected:
            raise ValueError(f"model expects {expected} observation values, got {batch.shape[1]}")
        actions, _ = self._model.predict(batch, deterministic=self._deterministic)
        return {agent: int(action) for agent, action in zip(agents, actions, strict=True)}


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
}


def available_policies() -> dict[str, Callable[[], Policy]]:
    """`POLICIES`, plus `rl` when the model file exists and Stable-Baselines3 is installed."""
    policies = dict(POLICIES)
    if MODEL_PATH.exists() and importlib.util.find_spec("stable_baselines3") is not None:
        policies["rl"] = lambda: RLPolicy(MODEL_PATH)
    return policies
