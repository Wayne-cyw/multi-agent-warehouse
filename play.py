"""Live demo: watch a policy run the warehouse.

Keys: space pause, R reset (same seed + 1), P cycle policy, Esc or Q quit.
Run `python play.py --help` for the flags.
"""

import argparse

import pygame

from multi_agent_warehouse.agents import POLICIES, Policy
from multi_agent_warehouse.config import DEFAULT_N_ROBOTS, MAX_ROBOTS, RENDER_FPS
from multi_agent_warehouse.env import MultiAgentWarehouseEnv

DEFAULT_POLICY = "greedy"
# `rl` is accepted on the command line so the flag is stable; Task 8 makes it available.
POLICY_CHOICES = [*POLICIES, "rl"]


def available_modes() -> list[str]:
    """Policy names the demo can run right now."""
    return list(POLICIES)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Watch a policy run the warehouse.")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default=DEFAULT_POLICY)
    parser.add_argument("--robots", type=int, default=DEFAULT_N_ROBOTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=RENDER_FPS)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    return parser.parse_args(argv)


def pick_mode(requested: str, modes: list[str]) -> str:
    """Return `requested` if available, else say so in one line and use the default."""
    if requested in modes:
        return requested
    print(f"policy '{requested}' is not available; using {DEFAULT_POLICY}")
    return DEFAULT_POLICY


def main(argv: list[str] | None = None) -> int:
    """Run the demo loop. Returns the process exit code."""
    args = parse_args(argv)
    if not 1 <= args.robots <= MAX_ROBOTS:
        print(f"--robots must be between 1 and {MAX_ROBOTS}")
        return 2

    modes = available_modes()
    mode = pick_mode(args.policy, modes)
    policy: Policy = POLICIES[mode]()

    env = MultiAgentWarehouseEnv(n_robots=args.robots, render_mode="human", render_fps=args.fps)
    seed = args.seed
    observations, _ = env.reset(seed=seed)
    policy.reset(seed)
    paused = False
    frame = 0

    try:
        # The window opens on the first render; pygame's event queue needs it to exist.
        env.hud_note = mode
        env.render()
        while args.frames == 0 or frame < args.frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return 0
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    seed += 1
                    observations, _ = env.reset(seed=seed)
                    policy.reset(seed)
                elif event.key == pygame.K_p:
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
                    policy = POLICIES[mode]()
                    policy.reset(seed)

            if not paused:
                if not env.agents:
                    seed += 1
                    observations, _ = env.reset(seed=seed)
                    policy.reset(seed)
                observations, *_ = env.step(policy.act(observations))

            env.hud_note = f"{mode} | PAUSED" if paused else mode
            env.render()
            frame += 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
