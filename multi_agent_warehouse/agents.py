"""Scripted policies. They see only observations, never the environment's internals.

Every policy has the same two methods, so the demo and the evaluation share one loop:
`reset(seed)` starts an episode and `act(observations)` returns one action per robot.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from multi_agent_warehouse.rules import ACTION_DELTAS, DOWN, LEFT, RIGHT, STAY, UP

# Observation layout: [row, col, dr, dc, up_blocked, down_blocked, left_blocked, right_blocked].
_DR, _DC, _BLOCKED = 2, 3, 4


class Policy(ABC):
    """A rule that turns every robot's observation into an action.

    A policy owns a random generator, so a seeded `reset` makes a whole episode repeatable.
    """

    def __init__(self) -> None:
        """Create the policy with an unseeded generator."""
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Start an episode with a fresh generator; a seed makes it repeatable."""
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """Pick one action per observed robot."""


class RandomPolicy(Policy):
    """Every robot picks one of the five actions uniformly at random."""

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """A uniform action per robot, drawn in sorted id order so a seed repeats exactly."""
        return {
            agent: int(self._rng.integers(len(ACTION_DELTAS))) for agent in sorted(observations)
        }


class GreedyPolicy(Policy):
    """Walk toward the target along the longer axis, sidestepping at random when stuck."""

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """One greedy action per robot, in sorted id order."""
        return {agent: self._act_one(observations[agent]) for agent in sorted(observations)}

    def _act_one(self, obs: np.ndarray) -> int:
        """Choose one robot's action from its own observation.

        Prefers the move along the larger offset (rows on a tie), then the other axis, then any
        open neighbour at random, and stays when boxed in or already on the target.
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
    """Whether the observation flags the neighbour in the direction of `action` as blocked."""
    return bool(obs[_BLOCKED + action - 1])


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
}
