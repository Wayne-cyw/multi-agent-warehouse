"""Tests for the multi-seed experiment runner, trained on tiny budgets."""

import csv
import dataclasses
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from multi_agent_warehouse import experiment
from multi_agent_warehouse.agents import GreedyPolicy
from multi_agent_warehouse.evaluate import evaluate_episodes, summarise

pytest.importorskip("stable_baselines3")
pytest.importorskip("supersuit")

TINY = {"n_seeds": 2, "total_steps": 3072, "n_envs": 2, "n_eval_episodes": 2}
CURVE_COLUMNS = [
    "variant",
    "seed",
    "timesteps",
    "mode",
    "mean_deliveries",
    "ci_deliveries",
    "mean_collisions",
    "ci_collisions",
    "n_episodes",
]
MANIFEST_KEYS = {
    "name",
    "created",
    "git_commit",
    "versions",
    "train_seeds",
    "eval_seed_base",
    "n_eval_episodes",
    "variants",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _curve_keys(directory: Path) -> list[tuple[str, str, str, str]]:
    rows = _rows(directory / "eval_curve.csv")
    return [(r["variant"], r["seed"], r["timesteps"], r["mode"]) for r in rows]


def _curve_by_key(directory: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = _rows(directory / "eval_curve.csv")
    return {(r["variant"], r["seed"], r["timesteps"], r["mode"]): r for r in rows}


def _write_curve(directory: Path, rows: list[dict[str, str]]) -> None:
    with open(directory / "eval_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class _Stop(Exception):
    pass


def _stopped_run(monkeypatch, tmp_path, name, variants, **kwargs):
    """Start a run that stops at its first training, so only its manifest is written.

    Returns the experiment directory and the seeds `fit` was called with.
    """
    seeds = []

    def stop(cfg, out, log_dir=None, verbose=0):
        seeds.append(cfg.seed)
        raise _Stop

    monkeypatch.setattr(experiment, "fit", stop)
    with pytest.raises(_Stop):
        experiment.run_experiment(name, variants, out_root=tmp_path, **{**TINY, **kwargs})
    return tmp_path / name, seeds


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    # One shared run keeps the file fast (contract checking makes every episode slow); tests that
    # change it work on a copy.
    root = tmp_path_factory.mktemp("experiments")
    return experiment.run_experiment("tiny", ["legacy"], out_root=root, **TINY)


def test_run_experiment_writes_manifest_curve_and_baselines(finished):
    assert finished.name == "tiny"
    assert set(_manifest(finished)) == MANIFEST_KEYS
    with open(finished / "eval_curve.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == CURVE_COLUMNS
    baselines = _rows(finished / "baselines.csv")
    assert [row["policy"] for row in baselines] == ["random", "greedy"]
    for seed in _manifest(finished)["train_seeds"]:
        run_dir = finished / "legacy" / f"seed_{seed}"
        assert (run_dir / "final.zip").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "progress.csv").exists()


def test_train_seeds_are_random_unless_given(finished, tmp_path, monkeypatch):
    again, _ = _stopped_run(monkeypatch, tmp_path, "again", ["legacy"])
    given, fitted = _stopped_run(monkeypatch, tmp_path, "given", ["legacy"], train_seeds=[11, 12])

    assert _manifest(again)["train_seeds"] != _manifest(finished)["train_seeds"]
    assert _manifest(given)["train_seeds"] == [11, 12]
    assert fitted == [11]


def test_train_seeds_avoid_the_eval_block(finished):
    manifest = _manifest(finished)
    base = manifest["eval_seed_base"]
    eval_seeds = range(base, base + manifest["n_eval_episodes"])

    assert manifest["n_eval_episodes"] == 2
    assert not set(manifest["train_seeds"]) & set(eval_seeds)


def test_manifest_records_commit_and_versions(finished, tmp_path, monkeypatch):
    manifest = _manifest(finished)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(experiment.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert manifest["git_commit"] == head
    assert set(manifest["versions"]) == {
        "python",
        "numpy",
        "torch",
        "stable_baselines3",
        "supersuit",
        "gymnasium",
        "pettingzoo",
    }
    assert manifest["versions"]["stable_baselines3"] != "not installed"
    legacy = manifest["variants"]["legacy"]
    assert legacy["total_steps"] == 3072
    assert legacy["n_envs"] == 2
    assert legacy["checkpoint_steps"] == [1536]
    assert legacy["bootstrap_truncation"] is False
    both, _ = _stopped_run(monkeypatch, tmp_path, "both", ["legacy", "base"])
    base = _manifest(both)["variants"]["base"]
    assert base["bootstrap_truncation"] is True
    assert base["total_steps"] == 3072
    assert base["checkpoint_steps"] == [1536]


def test_eval_curve_has_a_row_per_checkpoint_seed_and_mode(finished):
    seeds = [str(seed) for seed in _manifest(finished)["train_seeds"]]
    keys = _curve_keys(finished)
    expected = {
        (variant, seed, timesteps, mode)
        for variant in ("legacy",)
        for seed in seeds
        for timesteps in ("1536", "3072")
        for mode in ("sampled", "deterministic")
    }

    assert len(keys) == len(set(keys))
    assert set(keys) == expected
    assert all(row["n_episodes"] == "2" for row in _rows(finished / "eval_curve.csv"))


def test_baselines_use_the_same_eval_seeds(finished):
    manifest = _manifest(finished)
    base = manifest["eval_seed_base"]
    eval_seeds = range(base, base + manifest["n_eval_episodes"])
    stats = summarise(evaluate_episodes(GreedyPolicy, eval_seeds))
    greedy = next(row for row in _rows(finished / "baselines.csv") if row["policy"] == "greedy")

    assert float(greedy["mean_deliveries"]) == stats["deliveries"][0]
    assert float(greedy["ci_deliveries"]) == stats["deliveries"][1]
    assert float(greedy["mean_collisions"]) == stats["collisions"][0]
    assert float(greedy["ci_collisions"]) == stats["collisions"][1]
    assert greedy["n_episodes"] == "2"


def test_interrupted_run_is_completed_on_rerun(finished, tmp_path, monkeypatch):
    shutil.copytree(finished, tmp_path / "tiny")
    lost = _manifest(finished)["train_seeds"][1]
    # The lost run stopped after its checkpoint but before final.zip, and its old rows are stale:
    # the rerun must wipe both instead of keeping them.
    partial = tmp_path / "tiny" / "legacy" / f"seed_{lost}"
    (partial / "final.zip").unlink()
    (partial / "stale.txt").write_text("partial")
    assert any((partial / "checkpoints").iterdir())
    rows = _rows(finished / "eval_curve.csv")
    for row in rows:
        if row["seed"] == str(lost):
            row["mean_deliveries"] = "999.0"
    _write_curve(tmp_path / "tiny", rows)
    trained = []

    # Restoring the finished run stands in for retraining, which the shared run already covers.
    def counting_fit(cfg, out, log_dir=None, verbose=0):
        trained.append(cfg.seed)
        assert not log_dir.exists()
        shutil.copytree(finished / "legacy" / f"seed_{cfg.seed}", log_dir)
        return out.with_suffix(".zip")

    monkeypatch.setattr(experiment, "fit", counting_fit)

    directory = experiment.run_experiment("tiny", ["legacy"], out_root=tmp_path, **TINY)

    assert trained == [lost]
    assert not (partial / "stale.txt").exists()
    keys = _curve_keys(directory)
    assert Counter(keys).most_common(1)[0][1] == 1
    assert _curve_by_key(directory) == _curve_by_key(finished)
    assert len(_rows(directory / "baselines.csv")) == 2

    trained.clear()
    experiment.run_experiment("tiny", ["legacy"], out_root=tmp_path, **TINY)

    assert trained == []
    assert len(_curve_keys(directory)) == len(keys)


def test_partial_evaluation_is_completed_on_rerun(finished, tmp_path, monkeypatch):
    shutil.copytree(finished, tmp_path / "tiny")
    seed = str(_manifest(finished)["train_seeds"][0])
    rows = _rows(finished / "eval_curve.csv")
    missing = {("legacy", seed, "1536", "deterministic"), ("legacy", seed, "3072", "sampled")}
    kept = [r for r in rows if (r["variant"], r["seed"], r["timesteps"], r["mode"]) not in missing]
    assert len(kept) == len(rows) - 2
    _write_curve(tmp_path / "tiny", kept)

    def no_fit(cfg, out, log_dir=None, verbose=0):
        raise AssertionError("a finished run must not be retrained")

    monkeypatch.setattr(experiment, "fit", no_fit)

    directory = experiment.run_experiment("tiny", ["legacy"], out_root=tmp_path, **TINY)

    resumed = _rows(directory / "eval_curve.csv")
    assert resumed[: len(kept)] == kept
    added = {(r["variant"], r["seed"], r["timesteps"], r["mode"]) for r in resumed[len(kept) :]}
    assert added == missing
    assert _curve_by_key(directory) == _curve_by_key(finished)
    assert len(_curve_keys(directory)) == len(rows)


def test_rerun_with_conflicting_seeds_is_rejected(finished, tmp_path):
    shutil.copytree(finished, tmp_path / "tiny")
    seeds = _manifest(finished)["train_seeds"]

    with pytest.raises(ValueError, match="train_seeds"):
        experiment.run_experiment(
            "tiny",
            ["legacy"],
            out_root=tmp_path,
            train_seeds=[seed + 1 for seed in seeds],
            **TINY,
        )


def test_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        experiment.run_experiment("bad", ["nope"], out_root=tmp_path, **TINY)
    assert not (tmp_path / "bad").exists()


SWEEP_CHANGES = {
    "ent_0.01": ("ent_coef", 0.01),
    "shape_0": ("env_kwargs.shaping_scale", 0.0),
    "shape_0.05": ("env_kwargs.shaping_scale", 0.05),
    "shape_0.1": ("env_kwargs.shaping_scale", 0.1),
    "gamma_0.95": ("gamma", 0.95),
    "gamma_0.995": ("gamma", 0.995),
    "net_128": ("net_arch", (128, 128)),
    "coll_0.3": ("env_kwargs.collision_penalty", -0.3),
    "coll_1.0": ("env_kwargs.collision_penalty", -1.0),
    "rich_obs": ("env_kwargs.observation", "rich"),
}


def _flatten(config):
    flat = {}
    for key, value in config.to_dict().items():
        if key == "env_kwargs":
            flat.update({f"env_kwargs.{k}": v for k, v in value.items()})
        else:
            flat[key] = tuple(value) if isinstance(value, list) else value
    return flat


def test_sweep_variants_differ_from_base_in_exactly_one_field():
    base = _flatten(experiment.VARIANTS["base"])
    for name, (key, value) in SWEEP_CHANGES.items():
        assert name in experiment.VARIANTS
        flat = _flatten(experiment.VARIANTS[name])
        differing = {
            k
            for k in base.keys() | flat.keys()
            if base.get(k, "<absent>") != flat.get(k, "<absent>")
        }
        assert differing == {key}, name
        assert flat[key] == value, name
    with pytest.raises(ValueError, match="unknown"):
        experiment.run_experiment("bad", ["nope"], out_root=Path("/nonexistent"), **TINY)


def test_rich_obs_variant_sets_observation_kwarg():
    rich = experiment.VARIANTS["rich_obs"]
    base = experiment.VARIANTS["base"]
    assert rich.env_kwargs == {"observation": "rich"}
    assert dataclasses.replace(rich, env_kwargs=base.env_kwargs) == base


RECURRENT_CHANGES = {
    "lstm": {"algo": "recurrent_ppo"},
    "rich_lstm": {"algo": "recurrent_ppo", "env_kwargs.observation": "rich"},
}


def test_recurrent_variants_differ_from_base_as_specified():
    base = _flatten(experiment.VARIANTS["base"])
    for name, changes in RECURRENT_CHANGES.items():
        flat = _flatten(experiment.VARIANTS[name])
        differing = {
            k
            for k in base.keys() | flat.keys()
            if base.get(k, "<absent>") != flat.get(k, "<absent>")
        }
        assert differing == set(changes), name
        assert {k: flat[k] for k in changes} == changes, name


def test_evaluate_run_loads_a_recurrent_run(tmp_path):
    pytest.importorskip("sb3_contrib")
    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(
        algo="recurrent_ppo", n_envs=2, total_steps=1536, checkpoint_steps=(768,), seed=3
    )
    fit(cfg, out=tmp_path / "final", log_dir=tmp_path)

    rows = experiment.evaluate_run(tmp_path, cfg, eval_seeds=[5000], modes=("sampled",))

    assert [row["timesteps"] for row in rows] == [768, 1536]
    assert all(row["n_episodes"] == 1 for row in rows)


def test_cli_run_passes_its_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "run_experiment", lambda **kwargs: calls.append(kwargs))

    experiment.main(
        [
            "run",
            "--name",
            "n",
            "--variants",
            "legacy,base",
            "--seeds",
            "2",
            "--steps",
            "3072",
            "--train-seeds",
            "1,2",
            "--eval-seed-base",
            "20000",
        ]
    )

    assert calls == [
        {
            "name": "n",
            "variants": ["legacy", "base"],
            "n_seeds": 2,
            "total_steps": 3072,
            "train_seeds": [1, 2],
            "eval_seed_base": 20000,
        }
    ]


def test_generalisation_rows_cover_layout_and_count_grid(tmp_path):
    directory = experiment.run_experiment(
        "gen", ["legacy"], out_root=tmp_path, **{**TINY, "n_seeds": 2}
    )

    rows = experiment.evaluate_generalisation(
        directory, "legacy", ["default", "wide_gap"], [2, 3], n_episodes=2
    )

    seeds = _manifest(directory)["train_seeds"]
    assert Counter((r["seed"], r["layout"], r["n_robots"]) for r in rows) == dict.fromkeys(
        [(s, layout, n) for s in seeds for layout in ("default", "wide_gap") for n in (2, 3)], 1
    )
    written = _rows(directory / "generalisation.csv")
    assert list(written[0]) == list(experiment.GENERALISATION_COLUMNS)
    assert len(written) == len(rows)


def test_fixed_and_rand_variants():
    from multi_agent_warehouse.config import TRAIN_LAYOUTS

    assert experiment.VARIANTS["fixed"] == experiment.VARIANTS["base"]
    rand = experiment.VARIANTS["rand"]
    assert rand.env_kwargs == {"layouts": TRAIN_LAYOUTS}
    assert rand.robot_counts == (2, 3, 4, 5)
