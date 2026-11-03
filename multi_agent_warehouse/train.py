"""Train one shared PPO policy for all robots (parameter sharing through SuperSuit).

Stable-Baselines3 and SuperSuit come from the optional `rl` extra, so they are imported inside
`train` and importing this module does not need them.
"""

import argparse
from pathlib import Path

from multi_agent_warehouse.env import MultiAgentWarehouseEnv


def train(
    total_steps: int = 300_000,
    n_envs: int = 8,
    seed: int = 0,
    out: str = "models/ppo_multi_agent_warehouse",
) -> Path:
    """Train a shared-weights PPO agent and return the path of the saved zip.

    Each robot becomes one sub-environment of a vectorised env, and `n_envs` copies are
    concatenated, so a single network learns from every robot's experience.

    Preconditions:
        - total_steps >= 1
        - n_envs >= 1
    """
    import supersuit
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    env = MultiAgentWarehouseEnv(shaping=True)
    vec_env = supersuit.pettingzoo_env_to_vec_env_v1(env)
    vec_env = supersuit.concat_vec_envs_v1(
        vec_env, n_envs, num_cpus=1, base_class="stable_baselines3"
    )

    # PPO(seed=...) calls vec_env.seed(), which SuperSuit's ConcatVecEnv does not implement, so seed
    # Python/NumPy/torch here and seed the environments through ConcatVecEnv.reset instead.
    set_random_seed(seed)
    vec_env.venv.reset(seed=seed)

    model = PPO("MlpPolicy", vec_env, verbose=0)
    model.learn(total_steps)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    vec_env.close()
    return path.with_suffix(".zip")


def main() -> None:
    """Run training from the command line."""
    parser = argparse.ArgumentParser(description="Train a shared PPO policy.")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/ppo_multi_agent_warehouse")
    args = parser.parse_args()
    print(f"saved {train(args.steps, args.envs, args.seed, args.out)}")


if __name__ == "__main__":
    main()
