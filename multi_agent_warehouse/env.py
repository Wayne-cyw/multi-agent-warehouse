"""The multi-agent warehouse as a PettingZoo ParallelEnv.

Robots move on the shelved grid from `config.LAYOUT`, each chasing its own target cell. A robot
that ends a step on its target delivers and immediately gets a new one. Rewards are individual.
"""

from typing import TYPE_CHECKING

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from python_ta.contracts import check_contracts

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

if TYPE_CHECKING:
    from multi_agent_warehouse.render import Renderer


@check_contracts
class MultiAgentWarehouseEnv(ParallelEnv):
    """Several robots deliver to personal target cells on a small shelved grid.

    Attributes:
        max_steps: The number of steps after which an episode is truncated.
        shaping: Whether each step also pays a small reward for moving closer to the target.
        render_mode: How `render` draws (`human` or `rgb_array`), or None for no drawing.
        render_fps: The frame rate of the `human` window.
        possible_agents: The id of every robot, in order.
        agents: The ids of the robots in the running episode; empty when none is running.
        shelves: A boolean grid that is True where a shelf stands.
        height: The number of rows in the grid.
        width: The number of columns in the grid.
        positions: The cell each live robot stands on.
        targets: The cell each live robot is heading for.
        step_count: The number of steps taken in this episode.
        total_deliveries: The deliveries made by all robots in this episode.
        total_collisions: The collisions suffered by all robots in this episode.
        hud_note: Extra text shown in the HUD; empty for none.
        observation_spaces: The observation space of each robot.
        action_spaces: The action space of each robot.

    Representation Invariants:
        - 1 <= len(self.possible_agents) <= MAX_ROBOTS
        - set(self.agents) <= set(self.possible_agents)
        - self.max_steps >= 1
        - 0 <= self.step_count <= self.max_steps
        - self.total_deliveries >= 0
        - self.total_collisions >= 0
        - self.height > 0 and self.width > 0
    """

    metadata = {
        "name": "multi_agent_warehouse_v0",
        "render_modes": list(RENDER_MODES),
        "render_fps": RENDER_FPS,
    }

    max_steps: int
    shaping: bool
    render_mode: str | None
    render_fps: int
    possible_agents: list[str]
    agents: list[str]
    shelves: np.ndarray
    height: int
    width: int
    positions: dict[str, Cell]
    targets: dict[str, Cell]
    step_count: int
    total_deliveries: int
    total_collisions: int
    hud_note: str

    def __init__(
        self,
        n_robots: int = DEFAULT_N_ROBOTS,
        max_steps: int = DEFAULT_MAX_STEPS,
        shaping: bool = False,
        render_mode: str | None = None,
        render_fps: int = RENDER_FPS,
    ) -> None:
        """Create the env with `n_robots` robots.

        Preconditions:
            - 1 <= n_robots <= MAX_ROBOTS
            - max_steps >= 1
            - render_mode is None or render_mode in RENDER_MODES
            - render_fps > 0

        Raise ValueError if a precondition is violated; python_ta cannot check the
        module constants or keyword arguments these preconditions use.
        """
        if not 1 <= n_robots <= MAX_ROBOTS:
            raise ValueError(f"n_robots must be between 1 and {MAX_ROBOTS}, got {n_robots}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ValueError(
                f"render_mode must be None or one of {RENDER_MODES}, got {render_mode!r}"
            )
        if render_fps < 1:
            raise ValueError(f"render_fps must be at least 1, got {render_fps}")

        self.max_steps = max_steps
        self.shaping = shaping
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents = []

        self.shelves = np.array([[ch == "#" for ch in row] for row in LAYOUT], dtype=bool)
        self.height, self.width = self.shelves.shape
        self._scale = max(self.height, self.width) - 1
        self._free_cells: list[Cell] = [
            (int(r), int(c)) for r, c in zip(*np.nonzero(~self.shelves), strict=True)
        ]

        self.positions = {}
        self.targets = {}
        self.step_count = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.hud_note = ""
        self._deliveries: dict[str, int] = {}
        self._rng = np.random.default_rng()
        self._renderer: Renderer | None = None

        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: Discrete(len(ACTION_DELTAS)) for agent in self.possible_agents}

    def observation_space(self, agent: str) -> Box:
        """Eight floats in [-1, 1]: own cell, target offset, and four blocked-neighbour flags.

        Preconditions:
            - agent in self.possible_agents
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Discrete:
        """Stay, up, down, left or right.

        Preconditions:
            - agent in self.possible_agents
        """
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Start an episode. A seed makes a fresh generator; no seed continues the stream.

        Preconditions:
            - seed is None or seed >= 0
        """
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
        self, actions: dict[str, int | np.integer]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance one step: resolve moves, deliver, reward, tick, observe.

        Preconditions:
            - set(actions) == set(self.agents)
            - all(self.action_spaces[a].contains(action) for a, action in actions.items())
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        old_positions = self.positions
        new_positions, collided = resolve_moves(
            old_positions, {a: int(actions[a]) for a in self.agents}, self.shelves
        )
        self.positions = new_positions

        # Rewards are measured against the targets held at the start of the step.
        rewards = {
            agent: self._reward(agent, old_positions[agent], new_positions[agent], collided)
            for agent in self.agents
        }
        delivered = [a for a in self.agents if new_positions[a] == self.targets[a]]

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

    def render(self) -> np.ndarray | None:
        """Draw the current state. Return the frame in `rgb_array` mode, else None.

        `human` mode shows the window; with no render mode this does nothing. Pygame is only
        imported here, on first use, and rendering never touches the env's RNG.
        """
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from multi_agent_warehouse.render import Renderer

            self._renderer = Renderer(
                self.shelves, window=self.render_mode == "human", fps=self.render_fps
            )
        hud = [
            f"step {self.step_count}",
            f"deliveries {self.total_deliveries}",
            f"collisions {self.total_collisions}",
            self.hud_note,
        ]
        frame = self._renderer.draw(self.positions, self.targets, hud)
        return frame if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        """Release the renderer if one exists. Safe to call more than once."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _reward(self, agent: str, old: Cell, new: Cell, collided: set[str]) -> float:
        """One robot's reward for moving from `old` to `new` toward its current target.

        Preconditions:
            - agent in self.targets
        """
        target = self.targets[agent]
        reward = STEP_PENALTY
        if new == target:
            reward += DELIVERY_REWARD
        if agent in collided:
            reward += COLLISION_PENALTY
        if self.shaping:
            reward += SHAPING_SCALE * (_manhattan(old, target) - _manhattan(new, target))
        return reward

    def _sample_target(self, cell: Cell) -> Cell:
        """A random free cell other than `cell`.

        Preconditions:
            - len(self._free_cells) > 1
        """
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
                _is_blocked_cell((row + d_row, col + d_col), occupied, self.shelves)
                for d_row, d_col in ACTION_DELTAS[1:]
            ]
            obs = np.array([row, col, t_row - row, t_col - col, *blocked], dtype=np.float32)
            obs[:4] /= self._scale
            observations[agent] = obs
        return observations


def _is_blocked_cell(cell: Cell, occupied: set[Cell], shelves: np.ndarray) -> bool:
    """Whether a neighbour is off the grid, a shelf, or holds a robot.

    Preconditions:
        - shelves.ndim == 2
    """
    row, col = cell
    height, width = shelves.shape
    if not (0 <= row < height and 0 <= col < width):
        return True
    return bool(shelves[cell]) or cell in occupied


def _manhattan(a: Cell, b: Cell) -> int:
    """Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
