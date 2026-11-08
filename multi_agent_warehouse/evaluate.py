"""Seeded evaluation: mean deliveries and collisions per episode for each policy.

Run `python -m multi_agent_warehouse.evaluate` for the table, each mean with a 95% confidence
interval (`--episodes N` changes the sample size). Shaping is always off here, so the numbers
measure the game itself and not a training aid.
"""

import argparse
from collections.abc import Callable

import numpy as np
from python_ta.contracts import check_contracts

from multi_agent_warehouse.agents import Policy, available_policies
from multi_agent_warehouse.config import DEFAULT_N_ROBOTS
from multi_agent_warehouse.env import MultiAgentWarehouseEnv


@check_contracts
def run_episode(env: MultiAgentWarehouseEnv, policy: Policy, seed: int) -> dict[str, int]:
    """Play one full episode and return the env's delivery and collision totals.

    Preconditions:
        - seed >= 0
    """
    observations, _ = env.reset(seed=seed)
    policy.reset(seed)
    while env.agents:
        observations, *_ = env.step(policy.act(observations))
    return {"deliveries": env.total_deliveries, "collisions": env.total_collisions}


CI_Z = 1.96
DEFAULT_EPISODES = 100


@check_contracts
def evaluate_with_ci(
    make_policy: Callable[[], Policy],
    n_episodes: int = DEFAULT_EPISODES,
    first_seed: int = 1000,
    n_robots: int = DEFAULT_N_ROBOTS,
) -> dict[str, tuple[float, float]]:
    """Mean and 95% confidence half-width per metric over seeds `first_seed` onward.

    The half-width is 1.96 standard errors (normal approximation), and 0.0 for a single episode.

    Preconditions:
        - n_episodes >= 1
        - first_seed >= 0
        - 1 <= n_robots <= MAX_ROBOTS
    """
    results = []
    for seed in range(first_seed, first_seed + n_episodes):
        env = MultiAgentWarehouseEnv(n_robots=n_robots, shaping=False)
        results.append(run_episode(env, make_policy(), seed))
    stats = {}
    for key in ("deliveries", "collisions"):
        values = np.array([result[key] for result in results], dtype=float)
        half_width = CI_Z * values.std(ddof=1) / np.sqrt(n_episodes) if n_episodes > 1 else 0.0
        stats[key] = (float(values.mean()), float(half_width))
    return stats


@check_contracts
def evaluate(
    make_policy: Callable[[], Policy],
    n_episodes: int = DEFAULT_EPISODES,
    first_seed: int = 1000,
    n_robots: int = DEFAULT_N_ROBOTS,
) -> dict[str, float]:
    """Mean deliveries and collisions per episode over seeds `first_seed` onward.

    Preconditions:
        - n_episodes >= 1
        - first_seed >= 0
        - 1 <= n_robots <= MAX_ROBOTS
    """
    stats = evaluate_with_ci(make_policy, n_episodes, first_seed, n_robots)
    return {key: mean for key, (mean, _) in stats.items()}


def main(argv: list[str] | None = None) -> None:
    """Print a fixed-width table with one row per available policy."""
    parser = argparse.ArgumentParser(description="Seeded evaluation of every available policy.")
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="episodes per policy"
    )
    args = parser.parse_args(argv)
    print(f"{args.episodes} episodes per policy, seeds from 1000, mean +/- 95% CI")
    print(f"{'policy':<10}{'deliveries/episode':>24}{'collisions/episode':>24}")
    for name, make_policy in available_policies().items():
        stats = evaluate_with_ci(make_policy, args.episodes)
        cells = [f"{mean:.2f} +/- {half:.2f}" for mean, half in stats.values()]
        print(f"{name:<10}{cells[0]:>24}{cells[1]:>24}")


if __name__ == "__main__":
    main()
