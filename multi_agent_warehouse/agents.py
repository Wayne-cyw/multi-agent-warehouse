"""Scripted policies. They see only observations, never the environment's internals.

Every policy has the same two methods, so the demo and the evaluation share one loop:
`reset(seed)` starts an episode and `act(observations)` returns one action per robot.
"""

import importlib.util
import json
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
    """A trained network shared by every robot, sampling its actions unless deterministic.

    With `algo="recurrent_ppo"` the network is an LSTM, and the policy keeps one LSTM state per
    robot between `act` calls. Robots are batched in sorted id order, which is the order SuperSuit
    gives them inside one training warehouse, and `reset` clears the state, so an episode with a
    different number of robots starts cleanly. Stable-Baselines3 and sb3-contrib come from the
    optional `rl` extra, so they are imported only when a model is loaded.

    Attributes:
        _model: The loaded Stable-Baselines3 or sb3-contrib model.
        _deterministic: Whether to take the most likely action instead of sampling.
        _recurrent: Whether the model is recurrent and so carries state between steps.
        _state: The recurrent model's `(hidden, cell)` arrays from the last `act`, each shaped
            `(n_layers, n_robots, hidden_size)`, or None at the start of an episode.
        _episode_start: Whether the next `act` is the first of an episode.

    Representation Invariants:
        - self._rng is not None
        - self._recurrent or self._state is None
        - self._state is None or not self._episode_start
    """

    _model: Any
    _deterministic: bool
    _recurrent: bool
    _state: tuple[np.ndarray, np.ndarray] | None
    _episode_start: bool

    def __init__(
        self, model_path: str | Path = MODEL_PATH, deterministic: bool = False, algo: str = "ppo"
    ) -> None:
        """Load the `algo` model at `model_path`; `deterministic` picks the most likely action.

        Raises ValueError if `algo` is not one of `train.ALGOS`.
        """
        super().__init__()
        from multi_agent_warehouse.train import ALGOS, model_class

        if algo not in ALGOS:
            raise ValueError(f"unknown algo {algo!r}; expected one of {ALGOS}")
        self._model = model_class(algo).load(model_path, device="cpu")
        self._deterministic = deterministic
        self._recurrent = algo == "recurrent_ppo"
        self._state = None
        self._episode_start = True

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; a seed also fixes the network's sampling, so the episode repeats.

        Any recurrent state is cleared, so the next `act` starts a new episode.

        Preconditions:
            - seed is None or seed >= 0
        """
        super().reset(seed)
        self._state = None
        self._episode_start = True
        if seed is not None:
            # SB3 samples actions through torch's global generator, so that is the one to seed.
            import torch

            torch.manual_seed(seed)

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """One action per robot from a single batched forward pass, in sorted id order.

        Actions are sampled unless the policy is deterministic. Raises ValueError when the
        observations are not the size the model was trained on, or when a recurrent policy sees a
        different number of robots than earlier in the episode.

        Preconditions:
            - all(len(obs) >= 8 for obs in observations.values())
        """
        agents = sorted(observations)
        batch = np.stack([observations[agent] for agent in agents])
        expected = int(self._model.observation_space.shape[0])
        if batch.shape[1] != expected:
            raise ValueError(f"model expects {expected} observation values, got {batch.shape[1]}")
        if not self._recurrent:
            actions, _ = self._model.predict(batch, deterministic=self._deterministic)
        else:
            if self._state is not None and self._state[0].shape[1] != len(agents):
                raise ValueError("the robot count changed mid-episode; call reset first")
            episode_start = np.full(len(agents), self._episode_start)
            actions, state = self._model.predict(
                batch,
                state=self._state,
                episode_start=episode_start,
                deterministic=self._deterministic,
            )
            self._state = (state[0], state[1])
            self._episode_start = False
        return {agent: int(action) for agent, action in zip(agents, actions, strict=True)}


def _sidecar(model_path: str | Path) -> dict[str, Any]:
    """The JSON next to `model_path` (same name, `.json`), or an empty dict when there is none."""
    path = Path(model_path).with_suffix(".json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def policy_env_kwargs(model_path: str | Path) -> dict[str, Any]:
    """The env keyword arguments the model at `model_path` was trained with.

    They come from the model's sidecar, `<name>.json` beside `<name>.zip`, which is a training
    config; the result is empty when there is no sidecar, as for the original model.
    """
    return dict(_sidecar(model_path).get("env_kwargs", {}))


def policy_algo(model_path: str | Path) -> str:
    """The algorithm of the model at `model_path` from its sidecar, "ppo" when there is none."""
    return str(_sidecar(model_path).get("algo", "ppo"))


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
}


def available_policies() -> dict[str, Callable[[], Policy]]:
    """`POLICIES`, plus `rl` when the model file exists and its libraries are installed.

    Those are Stable-Baselines3, and sb3-contrib too when the model's sidecar says it is a
    recurrent PPO. The `rl` policy is built with the algorithm recorded in the sidecar.
    """
    policies = dict(POLICIES)
    if MODEL_PATH.exists() and importlib.util.find_spec("stable_baselines3") is not None:
        algo = policy_algo(MODEL_PATH)
        if algo != "recurrent_ppo" or importlib.util.find_spec("sb3_contrib") is not None:
            policies["rl"] = lambda: RLPolicy(MODEL_PATH, algo=algo)
    return policies
