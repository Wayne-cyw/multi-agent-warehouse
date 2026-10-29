"""Live demo: watch a policy run the warehouse.

Keys: space pause, R reset (same seed + 1), P cycle policy, Esc or Q quit.
Run `python play.py --help` for the flags.
"""

import argparse

import numpy as np
import pygame
from python_ta.contracts import check_contracts

from multi_agent_warehouse.agents import POLICIES, Policy
from multi_agent_warehouse.config import DEFAULT_N_ROBOTS, MAX_ROBOTS, RENDER_FPS
from multi_agent_warehouse.env import MultiAgentWarehouseEnv

DEFAULT_POLICY = "greedy"


@check_contracts
class Demo:
    """A window that steps one policy through endless episodes and reacts to key presses.

    Attributes:
        _policy_names: The names of every policy the P key cycles through.
        _policy_name: The name of the policy that is running.
        _policy: The running policy.
        _env: The environment being shown.
        _seed: The seed of the current episode.
        _paused: Whether stepping is paused.
        _observations: The latest observation of every live robot.

    Representation Invariants:
        - self._policy_name in self._policy_names
        - self._seed >= 0
    """

    _policy_names: list[str]
    _policy_name: str
    _policy: Policy
    _env: MultiAgentWarehouseEnv
    _seed: int
    _paused: bool
    _observations: dict[str, np.ndarray]

    def __init__(self, policy_name: str, n_robots: int, seed: int, fps: int) -> None:
        """Open the env for `n_robots` at `fps` and start the first episode with `seed`.

        Preconditions:
            - policy_name in POLICIES
            - 1 <= n_robots <= MAX_ROBOTS
            - seed >= 0
            - fps > 0
        """
        self._policy_names = list(POLICIES)
        self._policy_name = policy_name
        self._policy = POLICIES[policy_name]()
        self._env = MultiAgentWarehouseEnv(n_robots=n_robots, render_mode="human", render_fps=fps)
        self._seed = seed
        self._paused = False
        self._restart(self._seed)

    def run(self, max_frames: int = 0) -> None:
        """Loop until the user quits, or until `max_frames` frames when it is positive.

        Preconditions:
            - max_frames >= 0
        """
        try:
            # The window opens on the first render, and pygame's event queue needs it to exist.
            self._render()
            frame = 0
            while max_frames == 0 or frame < max_frames:
                if not self._handle_events():
                    return
                if not self._paused:
                    self._advance()
                self._render()
                frame += 1
        finally:
            self._env.close()

    def _handle_events(self) -> bool:
        """Apply pending window events. Returns False once the user asks to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.key == pygame.K_r:
                self._restart(self._seed + 1)
            elif event.key == pygame.K_p:
                self._cycle_policy()
        return True

    def _advance(self) -> None:
        """Step the episode once, starting a new episode first if the last one has ended.

        Preconditions:
            - not self._paused
        """
        if not self._env.agents:
            self._restart(self._seed + 1)
        self._observations, *_ = self._env.step(self._policy.act(self._observations))

    def _restart(self, seed: int) -> None:
        """Start a new episode, seeding both the env and the policy.

        Preconditions:
            - seed >= 0
        """
        self._seed = seed
        self._observations, _ = self._env.reset(seed=seed)
        self._policy.reset(seed)

    def _cycle_policy(self) -> None:
        """Switch to the next policy and seed it for the running episode."""
        index = self._policy_names.index(self._policy_name)
        self._policy_name = self._policy_names[(index + 1) % len(self._policy_names)]
        self._policy = POLICIES[self._policy_name]()
        self._policy.reset(self._seed)

    def _render(self) -> None:
        """Draw the current state with the policy name, and PAUSED while paused, in the HUD."""
        suffix = " | PAUSED" if self._paused else ""
        self._env.hud_note = f"{self._policy_name}{suffix}"
        self._env.render()


@check_contracts
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line, exiting with a usage error on a bad value."""
    parser = argparse.ArgumentParser(description="Watch a policy run the warehouse.")
    parser.add_argument("--policy", choices=list(POLICIES), default=DEFAULT_POLICY)
    parser.add_argument("--robots", type=int, default=DEFAULT_N_ROBOTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=RENDER_FPS)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = parser.parse_args(argv)
    if not 1 <= args.robots <= MAX_ROBOTS:
        parser.error(f"--robots must be between 1 and {MAX_ROBOTS}")
    return args


@check_contracts
def main(argv: list[str] | None = None) -> None:
    """Run the demo with the command-line settings."""
    args = parse_args(argv)
    Demo(args.policy, args.robots, args.seed, args.fps).run(args.frames)


if __name__ == "__main__":
    main()
