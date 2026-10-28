"""The multi-agent warehouse as a PettingZoo ParallelEnv.

Robots move on the shelved grid from `config.LAYOUT`, each chasing its own target cell. A robot
that ends a step on its target delivers and immediately gets a new one. Rewards are individual.
"""

from functools import cache

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from multi_agent_warehouse.config import (
    COLLISION_PENALTY,
    DEFAULT_MAX_STEPS,
    DEFAULT_N_ROBOTS,
    DELIVERY_REWARD,
    LAYOUT,
    MAX_ROBOTS,
    RENDER_FPS,
    RENDER_MODES,
    SHAPING_SCALE,
    STEP_PENALTY,
)
from multi_agent_warehouse.rules import ACTION_DELTAS, Cell, resolve_moves


class MultiAgentWarehouseEnv(ParallelEnv):
    """Several robots deliver to personal target cells on a small shelved grid."""

    metadata = {
        "name": "multi_agent_warehouse_v0",
        "render_modes": list(RENDER_MODES),
        "render_fps": RENDER_FPS,
    }

    def __init__(
        self,
        n_robots: int = DEFAULT_N_ROBOTS,
        max_steps: int = DEFAULT_MAX_STEPS,
        shaping: bool = False,
        render_mode: str | None = None,
    ) -> None:
        """Create the env. `n_robots` is 1 to `MAX_ROBOTS`; `shaping` adds a progress reward."""
        if not 1 <= n_robots <= MAX_ROBOTS:
            raise ValueError(f"n_robots must be between 1 and {MAX_ROBOTS}, got {n_robots}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ValueError(
                f"render_mode must be None or one of {RENDER_MODES}, got {render_mode!r}"
            )

        self.max_steps = max_steps
        self.shaping = shaping
        self.render_mode = render_mode
        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents: list[str] = []

        self.shelves = np.array([[ch == "#" for ch in row] for row in LAYOUT], dtype=bool)
        self.height, self.width = self.shelves.shape
        self._scale = max(self.height, self.width) - 1
        self._free_cells: list[Cell] = [
            (int(r), int(c)) for r, c in zip(*np.nonzero(~self.shelves), strict=True)
        ]

        self.positions: dict[str, Cell] = {}
        self.targets: dict[str, Cell] = {}
        self.step_count = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.hud_note = ""
        self._deliveries: dict[str, int] = {}
        self._rng = np.random.default_rng()

    @cache  # noqa: B019 (spaces are tiny and the env lives for the program)
    def observation_space(self, agent: str) -> Box:
        """Eight floats in [-1, 1]: own cell, target offset, and four blocked-neighbour flags."""
        return Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    @cache  # noqa: B019
    def action_space(self, agent: str) -> Discrete:
        """Stay, up, down, left or right."""
        return Discrete(5)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Start an episode. A seed makes a fresh generator; no seed continues the stream."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self._deliveries = dict.fromkeys(self.agents, 0)

        starts = self._rng.choice(len(self._free_cells), size=len(self.agents), replace=False)
        self.positions = {a: self._free_cells[i] for a, i in zip(self.agents, starts, strict=True)}
        self.targets = {a: self._sample_target(self.positions[a]) for a in self.agents}

        infos = {a: {"deliveries": 0, "collided": False} for a in self.agents}
        return self._observations(), infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance one step: validate, resolve moves, deliver, reward, tick, observe."""
        if not self.agents:
            return {}, {}, {}, {}, {}
        self._validate_actions(actions)

        old_positions = self.positions
        new_positions, collided = resolve_moves(
            old_positions, {a: int(actions[a]) for a in self.agents}, self.shelves
        )
        self.positions = new_positions

        # Rewards are measured against the targets held at the start of the step.
        rewards: dict[str, float] = {}
        delivered: list[str] = []
        for agent in self.agents:
            reward = STEP_PENALTY
            if new_positions[agent] == self.targets[agent]:
                reward += DELIVERY_REWARD
                delivered.append(agent)
            if agent in collided:
                reward += COLLISION_PENALTY
            if self.shaping:
                before = _manhattan(old_positions[agent], self.targets[agent])
                after = _manhattan(new_positions[agent], self.targets[agent])
                reward += SHAPING_SCALE * (before - after)
            rewards[agent] = reward

        for agent in delivered:
            self._deliveries[agent] += 1
            self.targets[agent] = self._sample_target(new_positions[agent])
        self.total_deliveries += len(delivered)
        self.total_collisions += len(collided)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        observations = self._observations()
        infos = {
            a: {"deliveries": self._deliveries[a], "collided": a in collided} for a in self.agents
        }
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, truncated)
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """Stub: the renderer arrives in the next task."""
        return None

    def close(self) -> None:
        """Stub: nothing to release yet."""

    def _validate_actions(self, actions: dict[str, int]) -> None:
        """Raise ValueError unless there is exactly one valid action per live robot."""
        if set(actions) != set(self.agents):
            raise ValueError(
                f"actions must have exactly the keys {self.agents}, got {sorted(actions)}"
            )
        for agent, action in actions.items():
            is_int = isinstance(action, int | np.integer) and not isinstance(
                action, bool | np.bool_
            )
            if not is_int or not 0 <= action < len(ACTION_DELTAS):
                raise ValueError(f"invalid action for {agent}: {action!r} (expected int 0 to 4)")

    def _sample_target(self, cell: Cell) -> Cell:
        """A random free cell other than `cell`."""
        candidates = [c for c in self._free_cells if c != cell]
        return candidates[int(self._rng.integers(len(candidates)))]

    def _observations(self) -> dict[str, np.ndarray]:
        """Build every live robot's 8-value observation."""
        occupied = set(self.positions.values())
        observations = {}
        for agent in self.agents:
            row, col = self.positions[agent]
            t_row, t_col = self.targets[agent]
            blocked = [
                self._is_blocked_cell((row + d_row, col + d_col), occupied)
                for d_row, d_col in ACTION_DELTAS[1:]
            ]
            obs = np.array([row, col, t_row - row, t_col - col, *blocked], dtype=np.float32)
            obs[:4] /= self._scale
            observations[agent] = obs
        return observations

    def _is_blocked_cell(self, cell: Cell, occupied: set[Cell]) -> bool:
        """Whether a neighbour is off the grid, a shelf, or holds a robot."""
        row, col = cell
        if not (0 <= row < self.height and 0 <= col < self.width):
            return True
        return bool(self.shelves[cell]) or cell in occupied


def _manhattan(a: Cell, b: Cell) -> int:
    """Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
