"""Seeded evaluation: mean deliveries and collisions per episode for each policy.

Run `python -m multi_agent_warehouse.evaluate` for the table. Shaping is always off here, so the
numbers measure the game itself and not a training aid.
"""

from collections.abc import Callable

from python_ta.contracts import check_contracts

from multi_agent_warehouse.agents import POLICIES, Policy
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


@check_contracts
def evaluate(
    make_policy: Callable[[], Policy],
    n_episodes: int = 20,
    first_seed: int = 1000,
    n_robots: int = DEFAULT_N_ROBOTS,
) -> dict[str, float]:
    """Mean deliveries and collisions per episode over seeds `first_seed` onward.

    Preconditions:
        - n_episodes >= 1
        - first_seed >= 0
        - 1 <= n_robots <= MAX_ROBOTS
    """
    totals = {"deliveries": 0, "collisions": 0}
    for seed in range(first_seed, first_seed + n_episodes):
        env = MultiAgentWarehouseEnv(n_robots=n_robots, shaping=False)
        result = run_episode(env, make_policy(), seed)
        for key in totals:
            totals[key] += result[key]
    return {key: total / n_episodes for key, total in totals.items()}


def main() -> None:
    """Print a fixed-width table with one row per available policy."""
    print(f"{'policy':<10}{'deliveries/episode':>20}{'collisions/episode':>20}")
    for name, make_policy in POLICIES.items():
        result = evaluate(make_policy)
        print(f"{name:<10}{result['deliveries']:>20.2f}{result['collisions']:>20.2f}")


if __name__ == "__main__":
    main()
