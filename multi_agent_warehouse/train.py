"""Train one shared PPO policy for all robots (parameter sharing through SuperSuit).

Stable-Baselines3, SuperSuit and torch come from the optional `rl` extra, so they are imported
inside functions and importing this module does not need them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multi_agent_warehouse.env import MultiAgentWarehouseEnv
from multi_agent_warehouse.seeds import draw_seeds

ALGOS = ("ppo",)


@dataclass(frozen=True)
class TrainConfig:
    """Everything that decides one training run, so a run can be saved and repeated.

    Attributes:
        total_steps: The number of environment steps (summed over sub-environments) to train for.
        n_envs: The number of warehouse copies trained side by side.
        seed: The seed for Python, NumPy, torch and the environments.
        n_steps: The rollout length per sub-environment.
        batch_size: The PPO minibatch size.
        learning_rate: The Adam learning rate.
        gamma: The discount factor.
        ent_coef: The entropy bonus coefficient.
        net_arch: The hidden layer widths of both the policy and the value network.
        algo: The training algorithm; only "ppo" for now.
        bootstrap_truncation: Whether time-limit truncations are bootstrapped from the value
            function instead of being treated as terminal.
        robot_counts: The robot counts cycled over the warehouse copies; copy i has
            `robot_counts[i % len(robot_counts)]` robots.
        env_kwargs: Extra keyword arguments for `MultiAgentWarehouseEnv`.
        checkpoint_steps: The timesteps after which a checkpoint is saved.

    Representation Invariants:
        - self.total_steps >= 1
        - self.n_envs >= 1
        - self.n_steps >= 1
        - self.batch_size >= 1
        - self.algo in ALGOS
        - len(self.net_arch) >= 1 and all(width >= 1 for width in self.net_arch)
        - len(self.robot_counts) >= 1 and all(count >= 1 for count in self.robot_counts)
        - all(step >= 1 for step in self.checkpoint_steps)
    """

    total_steps: int = 300_000
    n_envs: int = 8
    seed: int = 0
    n_steps: int = 128
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ent_coef: float = 0.0
    net_arch: tuple[int, ...] = (64, 64)
    algo: str = "ppo"
    bootstrap_truncation: bool = True
    robot_counts: tuple[int, ...] = (3,)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Reject configurations that cannot be trained."""
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}")
        for name in ("total_steps", "n_envs", "n_steps", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not self.net_arch or min(self.net_arch) < 1:
            raise ValueError(f"net_arch must be non-empty positive widths, got {self.net_arch}")
        if not self.robot_counts or min(self.robot_counts) < 1:
            raise ValueError(
                f"robot_counts must be non-empty positive counts, got {self.robot_counts}"
            )
        if any(step < 1 for step in self.checkpoint_steps):
            raise ValueError(f"checkpoint_steps must be positive, got {self.checkpoint_steps}")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON-serialisable values."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Rebuild a config from `to_dict` output, also after a JSON roundtrip turned tuples into
        lists."""
        values = dict(data)
        for name in ("net_arch", "robot_counts", "checkpoint_steps"):
            if name in values:
                values[name] = tuple(values[name])
        if "env_kwargs" in values:
            values["env_kwargs"] = dict(values["env_kwargs"])
        return cls(**values)


def _make_vec_env(cfg: TrainConfig) -> Any:
    """Build the seeded, monitored SB3 vec env for `cfg`, one sub-environment per robot.

    Copy i of the `cfg.n_envs` warehouses has `cfg.robot_counts[i % len(cfg.robot_counts)]`
    robots; the observation and action spaces do not depend on the robot count, so SuperSuit's
    `ConcatVecEnv` can join copies of different sizes.
    """
    import supersuit
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecMonitor
    from supersuit.vector import ConcatVecEnv
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

    def factory(n_robots: int) -> Any:
        """Return a zero-argument maker of one warehouse with `n_robots` robots as a vec env."""

        def make() -> Any:
            """Build the warehouse and turn each of its robots into a sub-environment."""
            env = MultiAgentWarehouseEnv(n_robots=n_robots, shaping=True, **cfg.env_kwargs)
            return supersuit.pettingzoo_env_to_vec_env_v1(env)

        return make

    counts = [cfg.robot_counts[i % len(cfg.robot_counts)] for i in range(cfg.n_envs)]
    vec_env = SB3VecEnvWrapper(ConcatVecEnv([factory(n) for n in counts]))

    # PPO(seed=...) calls vec_env.seed(), which SuperSuit's ConcatVecEnv does not implement, so seed
    # Python/NumPy/torch here and seed the environments through ConcatVecEnv.reset instead. This
    # must happen before VecMonitor, which hides `.venv`.
    set_random_seed(cfg.seed)
    vec_env.venv.reset(seed=cfg.seed)

    if not cfg.bootstrap_truncation:
        vec_env = _drop_truncation_flag(vec_env)

    # ConcatVecEnv has no `get_attr`, so SB3 warns that render_mode is unset; training never
    # renders, so None is right.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The `render_mode` attribute is not defined")
        return VecMonitor(vec_env)


def _drop_truncation_flag(vec_env: Any) -> Any:
    """Wrap `vec_env` so its infos lack "TimeLimit.truncated" and SB3 treats time limits as
    terminal."""
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class _DropTruncation(VecEnvWrapper):
        """Remove the time-limit flag from every step's infos."""

        def reset(self) -> Any:
            """Reset the wrapped env and return its observations."""
            return self.venv.reset()

        def step_wait(self) -> Any:
            """Step the wrapped env and strip "TimeLimit.truncated" from its infos."""
            observations, rewards, dones, infos = self.venv.step_wait()
            for info in infos:
                info.pop("TimeLimit.truncated", None)
            return observations, rewards, dones, infos

    return _DropTruncation(vec_env)


def _checkpoint_callback(steps: tuple[int, ...], directory: Path) -> Any:
    """Return a `CheckpointAtSteps` callback, defined lazily because it subclasses an SB3 class.

    Preconditions:
        - all(step >= 1 for step in steps)
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class CheckpointAtSteps(BaseCallback):
        """Save the model at the first rollout end at or after each requested timestep.

        A checkpoint is named by the real timestep count, `step_<timesteps:09d>.zip`, and several
        requested steps passed in one rollout share one checkpoint.

        Attributes:
            pending: The requested steps not reached yet, in ascending order.
            directory: Where the checkpoints are written.

        Representation Invariants:
            - self.pending == sorted(self.pending)
        """

        pending: list[int]
        directory: Path

        def __init__(self) -> None:
            """Start with every requested step pending."""
            super().__init__()
            self.pending = sorted(set(steps))
            self.directory = directory

        def _on_step(self) -> bool:
            """Keep training; checkpoints are only taken at rollout ends."""
            return True

        def _on_rollout_end(self) -> None:
            """Save a checkpoint if a requested step has been reached."""
            if not self.pending or self.num_timesteps < self.pending[0]:
                return
            self.pending = [step for step in self.pending if step > self.num_timesteps]
            self.directory.mkdir(parents=True, exist_ok=True)
            self.model.save(self.directory / f"step_{self.num_timesteps:09d}")

    return CheckpointAtSteps()


def fit(cfg: TrainConfig, out: Path, log_dir: Path | None = None, verbose: int = 0) -> Path:
    """Train an agent as `cfg` describes, save it to `out` and return the saved zip's path.

    With `log_dir`, also write `progress.csv`, `config.json` and `checkpoints/step_<n:09d>.zip`
    there.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    out = Path(out)
    vec_env = _make_vec_env(cfg)
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        device="cpu",
        verbose=verbose,
    )

    callback = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        model.set_logger(configure(str(log_dir), ["csv"]))
        if cfg.checkpoint_steps:
            callback = _checkpoint_callback(cfg.checkpoint_steps, log_dir / "checkpoints")

    model.learn(cfg.total_steps, callback=callback)

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    vec_env.close()
    return out.with_suffix(".zip")


def train(
    total_steps: int = 300_000,
    n_envs: int = 8,
    seed: int = 0,
    out: str = "models/ppo_multi_agent_warehouse",
    n_steps: int = 128,
    batch_size: int = 256,
    verbose: int = 0,
) -> Path:
    """Train a shared-weights PPO agent and return the path of the saved zip.

    Each robot becomes one sub-environment of a vectorised env, and `n_envs` copies are
    concatenated, so a single network learns from every robot's experience. SuperSuit makes each
    robot its own sub-environment, so one rollout holds `n_steps * n_robots * n_envs` samples.

    Preconditions:
        - total_steps >= 1
        - n_envs >= 1
        - n_steps >= 1
        - batch_size >= 1
    """
    cfg = TrainConfig(
        total_steps=total_steps,
        n_envs=n_envs,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    return fit(cfg, Path(out), verbose=verbose)


def main(argv: list[str] | None = None) -> None:
    """Run training from the command line; `argv` defaults to `sys.argv[1:]`."""
    parser = argparse.ArgumentParser(description="Train a shared PPO policy.")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="random in [0, 1000) if omitted")
    parser.add_argument("--out", default="models/ppo_multi_agent_warehouse")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    seed = draw_seeds(1)[0] if args.seed is None else args.seed
    print(f"seed {seed}")
    cfg = TrainConfig(
        total_steps=args.steps,
        n_envs=args.envs,
        seed=seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        net_arch=tuple(args.net_arch),
    )
    path = fit(cfg, Path(args.out), log_dir=args.log_dir, verbose=args.verbose)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
