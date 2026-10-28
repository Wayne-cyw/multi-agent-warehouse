"""Scripted policies. They see only observations, never the environment's internals.

Every policy has the same two methods, so the demo, the evaluation and the RL agent can share one
loop: `reset(seed)` starts an episode and `act(observations)` returns one action per robot.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from multi_agent_warehouse.rules import ACTION_DELTAS, DOWN, LEFT, RIGHT, STAY, UP

# Observation layout: [row, col, dr, dc, up_blocked, down_blocked, left_blocked, right_blocked].
_DR, _DC, _BLOCKED = 2, 3, 4


class Policy(Protocol):
    """What the demo and the evaluation need from a policy."""

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; a seed makes the policy's own randomness repeatable."""

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """Pick one action per observed robot."""


class RandomPolicy:
    """Every robot picks one of the five actions uniformly at random."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Make a fresh generator."""
        self._rng = np.random.default_rng(seed)

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """A uniform action per robot, drawn in sorted id order so a seed repeats exactly."""
        return {
            agent: int(self._rng.integers(len(ACTION_DELTAS))) for agent in sorted(observations)
        }


class GreedyPolicy:
    """Walk toward the target along the longer axis, sidestepping at random when stuck."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Make a fresh generator."""
        self._rng = np.random.default_rng(seed)

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        """One greedy action per robot, in sorted id order."""
        return {agent: self._act_one(observations[agent]) for agent in sorted(observations)}

    def _act_one(self, obs: np.ndarray) -> int:
        d_row, d_col = float(obs[_DR]), float(obs[_DC])
        if d_row == 0 and d_col == 0:
            return STAY
        row_move = DOWN if d_row > 0 else UP
        col_move = RIGHT if d_col > 0 else LEFT
        # The larger offset comes first; a tie goes to rows. A zero offset has no second choice.
        if abs(d_row) >= abs(d_col):
            ordered = [row_move, col_move if d_col != 0 else None]
        else:
            ordered = [col_move, row_move if d_row != 0 else None]

        for action in ordered:
            if action is not None and not obs[_BLOCKED + action - 1]:
                return action
        open_moves = [a for a in (UP, DOWN, LEFT, RIGHT) if not obs[_BLOCKED + a - 1]]
        if not open_moves:
            return STAY
        return open_moves[int(self._rng.integers(len(open_moves)))]


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
}
