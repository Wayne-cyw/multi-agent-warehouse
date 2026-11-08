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
import statistics
import subprocess
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from multi_agent_warehouse.agents import (
    GreedyPolicy,
    RandomPolicy,
    RLPolicy,
    policy_algo,
    policy_env_kwargs,
)
from multi_agent_warehouse.config import DEFAULT_N_ROBOTS, TRAIN_LAYOUTS
from multi_agent_warehouse.evaluate import evaluate_episodes, summarise
from multi_agent_warehouse.seeds import draw_eval_base, draw_seeds
from multi_agent_warehouse.train import TrainConfig, fit, model_class

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
_BASE = VARIANTS["base"]
VARIANTS.update(
    {
        "ent_0.01": dataclasses.replace(_BASE, ent_coef=0.01),
        "shape_0": dataclasses.replace(_BASE, env_kwargs={"shaping_scale": 0.0}),
        "shape_0.05": dataclasses.replace(_BASE, env_kwargs={"shaping_scale": 0.05}),
        "shape_0.1": dataclasses.replace(_BASE, env_kwargs={"shaping_scale": 0.1}),
        "gamma_0.95": dataclasses.replace(_BASE, gamma=0.95),
        "gamma_0.995": dataclasses.replace(_BASE, gamma=0.995),
        "net_128": dataclasses.replace(_BASE, net_arch=(128, 128)),
        "coll_0.3": dataclasses.replace(_BASE, env_kwargs={"collision_penalty": -0.3}),
        "coll_1.0": dataclasses.replace(_BASE, env_kwargs={"collision_penalty": -1.0}),
        "rich_obs": dataclasses.replace(_BASE, env_kwargs={"observation": "rich"}),
        "fixed": _BASE,
        "rand": dataclasses.replace(
            _BASE, env_kwargs={"layouts": TRAIN_LAYOUTS}, robot_counts=(2, 3, 4, 5)
        ),
        "lstm": dataclasses.replace(_BASE, algo="recurrent_ppo"),
        "rich_lstm": dataclasses.replace(
            _BASE, algo="recurrent_ppo", env_kwargs={"observation": "rich"}
        ),
    }
)
VARIANTS["tuned"] = dataclasses.replace(
    VARIANTS["rich_lstm"],
    ent_coef=VARIANTS["ent_0.01"].ent_coef,
    env_kwargs={**VARIANTS["rich_lstm"].env_kwargs, **VARIANTS["coll_0.3"].env_kwargs},
)

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
GENERALISATION_COLUMNS = (
    "variant",
    "seed",
    "layout",
    "n_robots",
    "mean_deliveries",
    "ci_deliveries",
    "mean_collisions",
    "ci_collisions",
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


def _checkpoints(run_dir: Path, algo: str) -> list[tuple[int, Path]]:
    """`(timesteps, path)` of every checkpoint and `final.zip` in `run_dir`, oldest first.

    `final.zip` is loaded as an `algo` model, labelled with its real timestep count and dropped
    when a checkpoint already has that count, so each policy is listed once.
    """
    found = {
        int(path.stem.removeprefix("step_")): path
        for path in (run_dir / "checkpoints").glob("step_*.zip")
    }
    final = run_dir / "final.zip"
    timesteps = model_class(algo).load(final, device="cpu").num_timesteps
    found.setdefault(int(timesteps), final)
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
    for timesteps, path in _checkpoints(run_dir, cfg.algo):
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


def evaluate_generalisation(
    exp_dir: Path,
    variant: str,
    layouts: Sequence[str],
    robot_counts: Sequence[int],
    n_episodes: int,
) -> list[dict]:
    """Evaluate every finished seed of `variant` on each layout and robot count.

    Each cell uses the `n_episodes` evaluation seeds from the manifest's `eval_seed_base`, sampled
    actions and no shaping; the variant's other `env_kwargs` apply but its `layouts` do not. The
    variant's rows in `generalisation.csv` are replaced, other variants' are kept. Returns the new
    rows with the `GENERALISATION_COLUMNS` keys.

    Preconditions:
        - n_episodes >= 1
        - len(layouts) >= 1 and len(robot_counts) >= 1
    """
    exp_dir = Path(exp_dir)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    template = TrainConfig.from_dict(manifest["variants"][variant])
    base = manifest["eval_seed_base"]
    eval_seeds = range(base, base + n_episodes)
    rows: list[dict] = []
    for seed in manifest["train_seeds"]:
        final = exp_dir / variant / f"seed_{seed}" / "final.zip"
        policy = RLPolicy(final, deterministic=False, algo=template.algo)
        for layout in layouts:
            kwargs = {**template.env_kwargs, "layouts": (layout,)}
            for n_robots in robot_counts:
                results = evaluate_episodes(
                    lambda policy=policy: policy, eval_seeds, n_robots=n_robots, env_kwargs=kwargs
                )
                cells = _stat_cells(summarise(results))
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "layout": layout,
                        "n_robots": n_robots,
                        **{key: float(value) for key, value in cells.items()},
                    }
                )
    path = exp_dir / "generalisation.csv"
    kept = [r for r in _read_rows(path) if r["variant"] != variant]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GENERALISATION_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows({**row, **_stat_cells_of(row)} for row in rows)
    os.replace(tmp, path)
    return rows


def _stat_cells_of(row: dict) -> dict[str, str]:
    """The four metric cells of `row` as `repr` strings, so a CSV reads back the same floats."""
    keys = ("mean_deliveries", "ci_deliveries", "mean_collisions", "ci_collisions")
    return {key: repr(row[key]) for key in keys}


FINAL_COLUMNS = (
    "variant",
    "seed",
    "selected",
    "validation_deliveries",
    "test_deliveries",
    "test_ci_deliveries",
    "test_collisions",
)


def _draw_blocks(n_episodes: int, avoid: Sequence[range]) -> tuple[int, int]:
    """Draw the first seeds of two disjoint random blocks of `n_episodes` that avoid `avoid`."""
    blocks: list[range] = list(avoid)
    bases = []
    while len(bases) < 2:
        base = draw_eval_base(n_episodes)
        block = range(base, base + n_episodes)
        if all(block.start >= other.stop or other.start >= block.stop for other in blocks):
            blocks.append(block)
            bases.append(base)
    return bases[0], bases[1]


def _score(model_path: Path, algo: str, env_kwargs: dict, seeds: range) -> dict[str, float]:
    """Mean and CI half-width of deliveries and the mean collisions of a sampled model."""
    policy = RLPolicy(model_path, deterministic=False, algo=algo)
    stats = summarise(
        evaluate_episodes(lambda: policy, seeds, n_robots=DEFAULT_N_ROBOTS, env_kwargs=env_kwargs)
    )
    return {
        "deliveries": stats["deliveries"][0],
        "ci_deliveries": stats["deliveries"][1],
        "collisions": stats["collisions"][0],
    }


def select_and_test(
    exp_dir: Path,
    variant: str,
    baseline: str | None = "base",
    committed: Path | None = None,
    n_episodes: int = 100,
) -> dict[str, Any]:
    """Pick the best seed of `variant` on a validation block and score it on a fresh test block.

    The validation and test blocks are random, `n_episodes` long, disjoint from each other, from
    the manifest's evaluation block and from every training seed. The seed with the most
    validation deliveries is selected; its test-block score is the honest estimate, because no
    choice was made on that block. Every seed of `variant` and `baseline`, and the `committed`
    model if given, is also scored on the test block, and `final.csv` in `exp_dir` lists them all.

    The result has `selected_seed`, `final_path`, the block bases, `test` (the selected seed's
    scores), `seed_spread` (std over the variant's seeds' test deliveries), `baseline_mean`,
    `committed` and `replace`: whether the selected seed beats both the baseline's mean and the
    committed model by more than the seed spread (False when either is missing).

    Preconditions:
        - n_episodes >= 1
    """
    exp_dir = Path(exp_dir)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    eval_base = manifest["eval_seed_base"]
    avoid = [
        range(eval_base, eval_base + manifest["n_eval_episodes"]),
        range(0, 1000),
        *(range(int(s), int(s) + 1) for s in manifest["train_seeds"]),
    ]
    val_base, test_base = _draw_blocks(n_episodes, avoid)
    val_seeds = range(val_base, val_base + n_episodes)
    test_seeds = range(test_base, test_base + n_episodes)

    rows: list[dict[str, Any]] = []
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for name in [variant, *([baseline] if baseline else [])]:
        cfg = TrainConfig.from_dict(manifest["variants"][name])
        scores[name] = {}
        for seed in manifest["train_seeds"]:
            final = exp_dir / name / f"seed_{seed}" / "final.zip"
            val = _score(final, cfg.algo, cfg.env_kwargs, val_seeds)
            test = _score(final, cfg.algo, cfg.env_kwargs, test_seeds)
            scores[name][seed] = {"validation": val["deliveries"], **test}
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "selected": False,
                    "validation_deliveries": repr(val["deliveries"]),
                    "test_deliveries": repr(test["deliveries"]),
                    "test_ci_deliveries": repr(test["ci_deliveries"]),
                    "test_collisions": repr(test["collisions"]),
                }
            )
    own = scores[variant]
    selected = max(own, key=lambda seed: own[seed]["validation"])
    for row in rows:
        row["selected"] = (row["variant"], row["seed"]) == (variant, selected)

    committed_score = None
    if committed is not None:
        committed_score = _score(
            Path(committed), policy_algo(committed), policy_env_kwargs(committed), test_seeds
        )
        rows.append(
            {
                "variant": "committed",
                "seed": "",
                "selected": False,
                "validation_deliveries": "",
                "test_deliveries": repr(committed_score["deliveries"]),
                "test_ci_deliveries": repr(committed_score["ci_deliveries"]),
                "test_collisions": repr(committed_score["collisions"]),
            }
        )
    with open(exp_dir / "final.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    spread = statistics.stdev(v["deliveries"] for v in own.values()) if len(own) > 1 else 0.0
    chosen = own[selected]["deliveries"]
    baseline_mean = (
        statistics.fmean(v["deliveries"] for v in scores[baseline].values()) if baseline else None
    )
    replace = (
        baseline_mean is not None
        and committed_score is not None
        and chosen - baseline_mean > spread
        and chosen - committed_score["deliveries"] > spread
    )
    return {
        "variant": variant,
        "selected_seed": selected,
        "final_path": exp_dir / variant / f"seed_{selected}" / "final.zip",
        "validation_base": val_base,
        "test_base": test_base,
        "n_episodes": n_episodes,
        "test": own[selected],
        "seed_spread": spread,
        "baseline_mean": baseline_mean,
        "committed": committed_score,
        "replace": bool(replace),
    }


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
    gen = commands.add_parser("generalise", help="evaluate variants across layouts and counts")
    gen.add_argument("--name", required=True)
    gen.add_argument("--variants", required=True, help="comma-separated")
    gen.add_argument("--layouts", default=None, help="comma-separated; default every layout")
    gen.add_argument("--robot-counts", type=_int_list, default=[2, 3, 4, 5, 6])
    gen.add_argument("--episodes", type=int, default=100)
    gen.add_argument("--out-root", type=Path, default=Path("runs/experiments"))
    sel = commands.add_parser("select", help="pick a seed on validation, score it on a test block")
    sel.add_argument("--name", required=True)
    sel.add_argument("--variant", required=True)
    sel.add_argument("--baseline", default="base")
    sel.add_argument("--committed", type=Path, default=None, help="model zip to compare against")
    sel.add_argument("--out-root", type=Path, default=Path("runs/experiments"))
    plot = commands.add_parser("plot", help="write figures and report.md for an experiment")
    plot.add_argument("--name", required=True)
    plot.add_argument("--out-root", type=Path, default=Path("runs/experiments"))
    args = parser.parse_args(argv)

    if args.command == "generalise":
        from multi_agent_warehouse.config import LAYOUTS

        layouts = list(LAYOUTS) if args.layouts is None else args.layouts.split(",")
        for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
            evaluate_generalisation(
                args.out_root / args.name, variant, layouts, args.robot_counts, args.episodes
            )
            print(f"evaluated {variant}")
    elif args.command == "select":
        result = select_and_test(
            args.out_root / args.name, args.variant, args.baseline, args.committed
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "plot":
        from multi_agent_warehouse.plots import plot_all

        for path in plot_all(args.out_root / args.name):
            print(path)
    elif args.command == "run":
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
