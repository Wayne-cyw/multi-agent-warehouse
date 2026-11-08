"""Multi-seed experiments: train each variant on several seeds and evaluate every checkpoint.

`python -m multi_agent_warehouse.experiment run --name N --variants legacy,base --seeds 5` writes
`runs/experiments/N/` with `manifest.json`, `eval_curve.csv`, `baselines.csv` and one directory per
run, `<variant>/seed_<seed>/`. Rerunning the same name resumes: finished runs are kept, unfinished
ones are retrained, and evaluations already in `eval_curve.csv` are not repeated.

Stable-Baselines3 is only imported when a model is trained or loaded, so this module imports
without the `rl` extra.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from multi_agent_warehouse.agents import GreedyPolicy, RandomPolicy, RLPolicy
from multi_agent_warehouse.config import DEFAULT_N_ROBOTS
from multi_agent_warehouse.evaluate import evaluate_episodes, summarise
from multi_agent_warehouse.seeds import draw_eval_base, draw_seeds
from multi_agent_warehouse.train import TrainConfig, fit

VARIANTS: dict[str, TrainConfig] = {
    "legacy": TrainConfig(
        total_steps=300_000,
        bootstrap_truncation=False,
        checkpoint_steps=(50_000, 100_000, 150_000, 200_000, 250_000, 300_000),
    ),
    "base": TrainConfig(
        total_steps=3_000_000,
        bootstrap_truncation=True,
        checkpoint_steps=(
            50_000,
            100_000,
            200_000,
            300_000,
            500_000,
            1_000_000,
            2_000_000,
            3_000_000,
        ),
    ),
}

MODES = ("sampled", "deterministic")
CURVE_COLUMNS = (
    "variant",
    "seed",
    "timesteps",
    "mode",
    "mean_deliveries",
    "ci_deliveries",
    "mean_collisions",
    "ci_collisions",
    "n_episodes",
)
BASELINE_COLUMNS = (
    "policy",
    "mean_deliveries",
    "ci_deliveries",
    "mean_collisions",
    "ci_collisions",
    "n_episodes",
)
BASELINES = {"random": RandomPolicy, "greedy": GreedyPolicy}
PACKAGES = ("numpy", "torch", "stable_baselines3", "supersuit", "gymnasium", "pettingzoo")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _stat_cells(stats: dict[str, tuple[float, float]]) -> dict[str, str]:
    """The four metric cells of a CSV row, as `repr` so they read back as the same floats."""
    return {
        "mean_deliveries": repr(stats["deliveries"][0]),
        "ci_deliveries": repr(stats["deliveries"][1]),
        "mean_collisions": repr(stats["collisions"][0]),
        "ci_collisions": repr(stats["collisions"][1]),
    }


def _checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """`(timesteps, path)` of every checkpoint and `final.zip` in `run_dir`, oldest first.

    `final.zip` is labelled with its model's real timestep count and dropped when a checkpoint
    already has that count, so each policy is listed once.
    """
    from stable_baselines3 import PPO

    found = {
        int(path.stem.removeprefix("step_")): path
        for path in (run_dir / "checkpoints").glob("step_*.zip")
    }
    final = run_dir / "final.zip"
    found.setdefault(int(PPO.load(final, device="cpu").num_timesteps), final)
    return sorted(found.items())


def _iter_eval_rows(
    run_dir: Path,
    cfg: TrainConfig,
    eval_seeds: Sequence[int],
    modes: Sequence[str],
    skip: frozenset[tuple[int, str]] = frozenset(),
) -> Iterator[dict[str, Any]]:
    """Yield one row per checkpoint and mode of `run_dir`, leaving out `(timesteps, mode)` in
    `skip`; see `evaluate_run`.
    """
    for timesteps, path in _checkpoints(run_dir):
        for mode in modes:
            if (timesteps, mode) in skip:
                continue
            # One model load per checkpoint and mode; reset(seed) reseeds it for every episode.
            policy = RLPolicy(path, deterministic=mode == "deterministic", algo=cfg.algo)
            results = evaluate_episodes(
                lambda policy=policy: policy,
                eval_seeds,
                n_robots=DEFAULT_N_ROBOTS,
                env_kwargs=cfg.env_kwargs,
            )
            yield {
                "seed": cfg.seed,
                "timesteps": timesteps,
                "mode": mode,
                **_stat_cells(summarise(results)),
                "n_episodes": len(results),
            }


def evaluate_run(
    run_dir: Path, cfg: TrainConfig, eval_seeds: Sequence[int], modes: Sequence[str] = MODES
) -> list[dict]:
    """Evaluate every checkpoint and `final.zip` of the finished run in `run_dir`.

    Returns one row per checkpoint and mode with the `eval_curve.csv` columns except `variant`;
    the seed is `cfg.seed`. Shaping is off, as in all evaluation.

    Preconditions:
        - (run_dir / "final.zip").exists()
        - len(eval_seeds) >= 1
        - all(mode in MODES for mode in modes)
    """
    return list(_iter_eval_rows(Path(run_dir), cfg, eval_seeds, modes))


def _append_row(path: Path, columns: Sequence[str], row: dict[str, Any]) -> None:
    """Append `row` to the CSV at `path`, writing the header first if the file is new.

    The file is flushed after each row, so an interrupted run leaves only whole rows behind.
    """
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _read_rows(path: Path) -> list[dict[str, str]]:
    """The rows of the CSV at `path`, or none if it does not exist."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _drop_rows(path: Path, variant: str, seed: int) -> None:
    """Remove the `eval_curve.csv` rows of one run, whose model is about to be retrained."""
    rows = _read_rows(path)
    kept = [r for r in rows if (r["variant"], r["seed"]) != (variant, str(seed))]
    if len(kept) == len(rows):
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(tmp, path)


def _versions() -> dict[str, str]:
    """Python and key package versions, "not installed" for a missing package."""
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package.replace("_", "-"))
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _git_commit() -> str:
    """The repository's HEAD commit, or "unknown" outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _variant_configs(
    variants: Sequence[str], total_steps: int | None, n_envs: int | None
) -> dict[str, TrainConfig]:
    """The config of each named variant with the step and env-count overrides applied.

    A checkpoint past an overridden `total_steps` is dropped; if none is left, the run is
    checkpointed halfway. Raises ValueError for an unknown or repeated variant.
    """
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known variants: {sorted(VARIANTS)}")
    if len(set(variants)) != len(variants):
        raise ValueError(f"variants must not repeat, got {list(variants)}")
    configs = {}
    for name in variants:
        cfg = VARIANTS[name]
        if n_envs is not None:
            cfg = dataclasses.replace(cfg, n_envs=n_envs)
        if total_steps is not None:
            steps = tuple(s for s in cfg.checkpoint_steps if s <= total_steps)
            steps = steps or (max(1, total_steps // 2),)
            cfg = dataclasses.replace(cfg, total_steps=total_steps, checkpoint_steps=steps)
        configs[name] = cfg
    return configs


def _resolve_manifest(
    path: Path,
    name: str,
    configs: dict[str, TrainConfig],
    overridden: bool,
    n_seeds: int,
    train_seeds: Sequence[int] | None,
    eval_seed_base: int | None,
    n_eval_episodes: int,
) -> dict[str, Any]:
    """Load the manifest at `path`, or draw the seeds and write a new one there.

    An existing manifest wins over values left at None; an explicit value that disagrees with it
    raises ValueError naming the field. `overridden` says whether the step or env counts were
    given, and so whether `configs` must match the manifest's variants.
    """
    if path.exists():
        manifest = json.loads(path.read_text())
        stored = {key: TrainConfig.from_dict(value) for key, value in manifest["variants"].items()}
        conflicts = {
            "train_seeds": train_seeds is not None and list(train_seeds) != manifest["train_seeds"],
            "n_seeds": n_seeds != len(manifest["train_seeds"]),
            "eval_seed_base": eval_seed_base is not None
            and eval_seed_base != manifest["eval_seed_base"],
            "n_eval_episodes": n_eval_episodes != manifest["n_eval_episodes"],
            "variants": list(stored) != list(configs) or (overridden and stored != configs),
        }
        for field_name, conflict in conflicts.items():
            if conflict:
                raise ValueError(f"{field_name} disagrees with the manifest at {path}")
        return manifest

    seeds = draw_seeds(n_seeds) if train_seeds is None else [int(s) for s in train_seeds]
    base = draw_eval_base(n_eval_episodes) if eval_seed_base is None else int(eval_seed_base)
    if len(seeds) != n_seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"train_seeds must be {n_seeds} distinct seeds, got {seeds}")
    if base < 0 or set(seeds) & set(range(base, base + n_eval_episodes)):
        raise ValueError(f"train_seeds {seeds} overlap the evaluation block from {base}")
    manifest = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "train_seeds": seeds,
        "eval_seed_base": base,
        "n_eval_episodes": n_eval_episodes,
        "variants": {key: cfg.to_dict() for key, cfg in configs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)
    return manifest


def run_experiment(
    name: str,
    variants: Sequence[str],
    n_seeds: int = 5,
    total_steps: int | None = None,
    train_seeds: Sequence[int] | None = None,
    eval_seed_base: int | None = None,
    n_eval_episodes: int = 100,
    out_root: Path = Path("runs/experiments"),
    n_envs: int | None = None,
) -> Path:
    """Train and evaluate every variant on every training seed; return the experiment directory.

    Seeds are drawn once and written to `manifest.json` before any training; a rerun with the same
    `name` reads them back and resumes. `total_steps` and `n_envs` override every variant's own.

    Preconditions:
        - n_seeds >= 1
        - n_eval_episodes >= 1
        - total_steps is None or total_steps >= 1
        - n_envs is None or n_envs >= 1
    """
    configs = _variant_configs(variants, total_steps, n_envs)
    directory = Path(out_root) / name
    overridden = total_steps is not None or n_envs is not None
    manifest = _resolve_manifest(
        directory / "manifest.json",
        name,
        configs,
        overridden,
        n_seeds,
        train_seeds,
        eval_seed_base,
        n_eval_episodes,
    )
    configs = {key: TrainConfig.from_dict(value) for key, value in manifest["variants"].items()}
    base = manifest["eval_seed_base"]
    eval_seeds = range(base, base + manifest["n_eval_episodes"])

    curve_csv = directory / "eval_curve.csv"
    for variant, template in configs.items():
        for seed in manifest["train_seeds"]:
            cfg = dataclasses.replace(template, seed=seed)
            run_dir = directory / variant / f"seed_{seed}"
            if not (run_dir / "final.zip").exists():
                # A run without final.zip was interrupted: start it again from scratch.
                shutil.rmtree(run_dir, ignore_errors=True)
                _drop_rows(curve_csv, variant, seed)
                fit(cfg, out=run_dir / "final", log_dir=run_dir)
            done = frozenset(
                (int(row["timesteps"]), row["mode"])
                for row in _read_rows(curve_csv)
                if (row["variant"], row["seed"]) == (variant, str(seed))
            )
            for row in _iter_eval_rows(run_dir, cfg, eval_seeds, MODES, skip=done):
                _append_row(curve_csv, CURVE_COLUMNS, {"variant": variant, **row})

    baselines_csv = directory / "baselines.csv"
    done_baselines = {row["policy"] for row in _read_rows(baselines_csv)}
    for policy_name, policy_class in BASELINES.items():
        if policy_name not in done_baselines:
            results = evaluate_episodes(policy_class, eval_seeds, n_robots=DEFAULT_N_ROBOTS)
            row = {"policy": policy_name, **_stat_cells(summarise(results))}
            _append_row(baselines_csv, BASELINE_COLUMNS, {**row, "n_episodes": len(results)})
    return directory


def _int_list(text: str) -> list[int]:
    """Parse a comma-separated list of integers such as "1,2,3"."""
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    """Run an experiment from the command line; `argv` defaults to `sys.argv[1:]`."""
    parser = argparse.ArgumentParser(description="Multi-seed training experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train and evaluate variants on several seeds")
    run.add_argument("--name", required=True)
    run.add_argument("--variants", required=True, help=f"comma-separated, from {sorted(VARIANTS)}")
    run.add_argument("--seeds", type=int, required=True, help="number of training seeds")
    run.add_argument("--steps", type=int, default=None, help="override every variant's steps")
    run.add_argument("--train-seeds", type=_int_list, default=None, help="e.g. 1,2; else random")
    run.add_argument("--eval-seed-base", type=int, default=None, help="random if omitted")
    args = parser.parse_args(argv)

    if args.command == "run":
        directory = run_experiment(
            name=args.name,
            variants=[v.strip() for v in args.variants.split(",") if v.strip()],
            n_seeds=args.seeds,
            total_steps=args.steps,
            train_seeds=args.train_seeds,
            eval_seed_base=args.eval_seed_base,
        )
        print(f"results in {directory}")


if __name__ == "__main__":
    main()
