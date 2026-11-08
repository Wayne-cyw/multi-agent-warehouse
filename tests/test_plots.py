"""Tests for experiment figures and the report, on synthetic experiment directories."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from multi_agent_warehouse import experiment, plots

CURVE_FIELDS = experiment.CURVE_COLUMNS
STEPS = (50_000, 100_000, 200_000)


def make_experiment(
    root: Path,
    variants=("base",),
    seeds=(1, 2),
    modes=("sampled", "deterministic"),
    baselines=True,
    progress=True,
    jitter=0,
) -> Path:
    """Write a synthetic experiment directory; `jitter` offsets timesteps per seed."""
    exp = root / "exp"
    exp.mkdir(parents=True)
    manifest = {
        "name": "exp",
        "git_commit": "abc123",
        "train_seeds": list(seeds),
        "eval_seed_base": 5000,
        "n_eval_episodes": 20,
    }
    (exp / "manifest.json").write_text(json.dumps(manifest))
    with open(exp / "eval_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for vi, variant in enumerate(variants):
            for seed in seeds:
                for ti, steps in enumerate(STEPS):
                    for mode in modes:
                        writer.writerow(
                            {
                                "variant": variant,
                                "seed": seed,
                                "timesteps": steps + seed * jitter,
                                "mode": mode,
                                "mean_deliveries": 2.0 + ti + vi + seed * 0.1,
                                "ci_deliveries": 0.3,
                                "mean_collisions": 5.0 - ti + seed * 0.1,
                                "ci_collisions": 0.2,
                                "n_episodes": 20,
                            }
                        )
    if baselines:
        with open(exp / "baselines.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=experiment.BASELINE_COLUMNS)
            writer.writeheader()
            for policy, d, c in (("random", 0.5, 9.0), ("greedy", 3.0, 4.0)):
                writer.writerow(
                    {
                        "policy": policy,
                        "mean_deliveries": d,
                        "ci_deliveries": 0.1,
                        "mean_collisions": c,
                        "ci_collisions": 0.1,
                        "n_episodes": 20,
                    }
                )
    if progress:
        for variant in variants:
            for seed in seeds:
                run = exp / variant / f"seed_{seed}"
                run.mkdir(parents=True)
                with open(run / "progress.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([c for c, _ in plots.DIAGNOSTICS] + ["time/total_timesteps"])
                    writer.writerow(["", "", "", "", 1000])  # first rollout: all blank
                    writer.writerow(["0.5", "-1.0", "0.3", "0.1", 2000])
                    writer.writerow(["1.5", "-0.8", "0.2", "0.4", 3000 + seed])
    return exp


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_seed_band_values():
    mean, lower, upper = plots.seed_band([1, 2, 3])
    half = 4.303 / math.sqrt(3)
    assert mean == pytest.approx(2)
    assert upper - mean == pytest.approx(half)
    assert mean - lower == pytest.approx(half)
    assert half == pytest.approx(2.484, abs=1e-3)


def test_seed_band_single_value_has_no_nan():
    assert plots.seed_band([4.5]) == (4.5, 4.5, 4.5)


def test_seed_band_uses_normal_critical_value_beyond_ten_df():
    values = list(range(12))
    mean, _, upper = plots.seed_band(values)
    std = float(np.std(values, ddof=1))
    assert upper - mean == pytest.approx(1.96 * std / math.sqrt(12))


def test_aggregate_curve_groups_by_checkpoint_index_with_unequal_timesteps():
    series = {1: [(3000, 1.0), (6000, 3.0)], 2: [(3072, 3.0), (6144, 5.0)]}
    agg = plots.aggregate_curve(series)
    assert [point[0] for point in agg] == [3036.0, 6072.0]
    assert [point[1] for point in agg] == [2.0, 4.0]
    assert agg[0][2] < 2.0 < agg[0][3]


def test_aggregate_curve_uneven_seed_lengths():
    agg = plots.aggregate_curve({1: [(10, 1.0), (20, 2.0)], 2: [(10, 3.0)]})
    assert agg[1] == (20.0, 2.0, 2.0, 2.0)


def test_aggregate_progress_treats_blanks_as_missing_not_zero():
    grid, mean = plots.aggregate_progress([([1, 2, 3], [math.nan, 4.0, 6.0])])
    assert list(grid) == [2.0, 3.0]
    assert list(mean) == [4.0, 6.0]


def test_aggregate_progress_averages_seeds_on_shared_grid():
    grid, mean = plots.aggregate_progress([([1, 2, 3], [0, 2, 4]), ([1, 3], [2, 4])])
    assert list(grid) == [1.0, 2.0, 3.0]
    assert list(mean) == [1.0, 2.5, 4.0]


def test_learning_curve_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("legacy", "base"), seeds=(1, 2, 3))
    out = plots.plot_learning_curve(exp, "deliveries", tmp_path / "lc.png")
    assert nonempty(out)
    assert nonempty(
        plots.plot_learning_curve(exp, "collisions", tmp_path / "lc2.png", "deterministic")
    )


def test_learning_curve_rejects_unknown_metric(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    with pytest.raises(ValueError, match="metric"):
        plots.plot_learning_curve(exp, "reward", tmp_path / "x.png")


def test_learning_curve_with_one_seed_and_one_checkpoint(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(7,))
    lines = [r for r in (exp / "eval_curve.csv").read_text().splitlines()]
    (exp / "eval_curve.csv").write_text("\n".join(lines[:3]) + "\n")  # header + one checkpoint
    assert nonempty(plots.plot_learning_curve(exp, "deliveries", tmp_path / "one.png"))


def test_learning_curve_with_unequal_checkpoint_timesteps(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(1, 2, 3), jitter=72)
    assert nonempty(plots.plot_learning_curve(exp, "deliveries", tmp_path / "j.png"))


def test_learning_curve_without_baselines_file(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, baselines=False)
    assert nonempty(plots.plot_learning_curve(exp, "deliveries", tmp_path / "nb.png"))
    assert nonempty(plots.plot_tradeoff(exp, tmp_path / "nb2.png"))


def test_sampled_vs_deterministic_names_missing_mode(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, modes=("sampled",))
    with pytest.raises(ValueError, match="deterministic"):
        plots.plot_sampled_vs_deterministic(exp, "base", tmp_path / "x.png")


def test_sampled_vs_deterministic_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    assert nonempty(plots.plot_sampled_vs_deterministic(exp, "base", tmp_path / "sd.png"))


def test_training_diagnostics_with_blank_cells(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    assert nonempty(plots.plot_training_diagnostics(exp, "base", tmp_path / "d.png"))


def test_training_diagnostics_without_progress_raises(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, progress=False)
    with pytest.raises(FileNotFoundError):
        plots.plot_training_diagnostics(exp, "base", tmp_path / "d.png")


def test_variant_comparison_orders_by_score(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("low", "mid", "high"), seeds=(1, 2, 3))
    # make "mid" the best variant by editing its rows
    rows = plots.load_curve(exp)
    for row in rows:
        row["mean_deliveries"] += 10 if row["variant"] == "mid" else 0
    ranked = plots.rank_variants(rows)
    assert [name for name, *_ in ranked][0] == "mid"
    assert [name for name, *_ in ranked] == ["mid", "high", "low"]
    assert ranked[0][3] == 3
    assert ranked[0][2] == pytest.approx(0.1)  # std of seeds 1..3 at +0.1 each
    assert nonempty(plots.plot_variant_comparison(exp, tmp_path / "vc.png"))


def test_variant_comparison_single_variant_single_seed(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(1,))
    assert plots.rank_variants(plots.load_curve(exp))[0][2] == 0.0
    assert nonempty(plots.plot_variant_comparison(exp, tmp_path / "vc.png"))


def test_final_checkpoint_is_largest_timesteps(tmp_path):
    exp = make_experiment(tmp_path, seeds=(1,))
    final = plots.final_results(plots.load_curve(exp), "sampled")
    assert final[("base", 1)]["timesteps"] == max(STEPS)


def test_make_report_links_every_figure_it_made(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("legacy", "base"))
    written = plots.plot_all(exp)
    report = exp / "report.md"
    assert written[-1] == report
    text = report.read_text()
    pngs = list((exp / "plots").glob("*.png"))
    assert len(pngs) >= 6
    for png in pngs:
        assert f"](plots/{png.name})" in text
    assert "abc123" in text and "5000" in text and "Train seeds: 1, 2" in text
    assert "### Sampled policy" in text and "### Deterministic policy" in text
    assert "| random |" in text and "| greedy |" in text
    assert "± " in text


def test_make_report_without_figures_or_baselines(tmp_path):
    exp = make_experiment(tmp_path, baselines=False)
    text = plots.make_report(exp).read_text()
    assert "Figures" not in text and "Baselines" not in text


def test_plot_all_without_progress_skips_diagnostics(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, progress=False)
    names = {p.name for p in plots.plot_all(exp)}
    assert "diagnostics_base.png" not in names
    assert {"learning_curve_deliveries.png", "tradeoff.png", "report.md"} <= names


def test_plot_all_skips_missing_mode(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, modes=("sampled",))
    names = {p.name for p in plots.plot_all(exp)}
    assert "sampled_vs_deterministic_base.png" not in names


def test_promote_copies_named_figures(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    plots.plot_all(exp)
    dest = tmp_path / "docs" / "figs"
    copied = plots.promote(exp, ["tradeoff", "variant_comparison"], dest)
    assert [p.name for p in copied] == ["tradeoff.png", "variant_comparison.png"]
    assert all(nonempty(p) for p in copied)


def test_promote_names_missing_figure(tmp_path):
    exp = make_experiment(tmp_path)
    with pytest.raises(FileNotFoundError, match="nope.png"):
        plots.promote(exp, ["nope"], tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_plot_subcommand_writes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    experiment.main(["plot", "--name", "exp", "--out-root", str(tmp_path)])
    assert nonempty(exp / "plots" / "learning_curve_deliveries.png")
    assert nonempty(exp / "report.md")
