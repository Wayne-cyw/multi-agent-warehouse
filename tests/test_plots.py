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
                    # The first rollout row is all blank.
                    writer.writerow(["", "", "", "", 1000])
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
    # Header plus one checkpoint row.
    (exp / "eval_curve.csv").write_text("\n".join(lines[:3]) + "\n")
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
    # Make "mid" the best variant by editing its rows.
    rows = plots.load_curve(exp)
    for row in rows:
        row["mean_deliveries"] += 10 if row["variant"] == "mid" else 0
    ranked = plots.rank_variants(rows)
    assert [name for name, *_ in ranked][0] == "mid"
    assert [name for name, *_ in ranked] == ["mid", "high", "low"]
    assert ranked[0][3] == 3
    # Seeds 1..3 differ by 0.1 each, so the std is 0.1.
    assert ranked[0][2] == pytest.approx(0.1)
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


def dashed(ax):
    """The dashed (baseline) lines of `ax`."""
    return [line for line in ax.get_lines() if line.get_linestyle() == "--"]


def close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_learning_curve_figure_content(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("legacy", "base"), seeds=(1, 2, 3))
    fig = plots.figure_learning_curve(exp, "deliveries")
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(dashed(ax)) == 2
    # 6 thin seed lines + 2 bold means + 2 baselines
    assert len(ax.get_lines()) == 10
    # One band per variant.
    assert len(ax.collections) == 2
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [
        "legacy (mean of 3)",
        "base (mean of 3)",
        "random baseline (0.50)",
        "greedy baseline (3.00)",
    ]
    assert "deliveries per episode" in ax.get_ylabel()
    assert "sampled" in ax.get_title() and "exp" in ax.get_title()
    close(fig)


def test_learning_curve_figure_without_baselines_has_no_dashed_lines(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, baselines=False)
    fig = plots.figure_learning_curve(exp, "collisions")
    ax = fig.axes[0]
    assert dashed(ax) == []
    # Two thin seed lines and one mean.
    assert len(ax.get_lines()) == 2 + 1
    assert "collisions per episode" in ax.get_ylabel()
    close(fig)


def test_learning_curve_figure_single_seed_has_degenerate_band(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(7,))
    fig = plots.figure_learning_curve(exp, "deliveries")
    ax = fig.axes[0]
    mean_line = [ln for ln in ax.get_lines() if ln.get_linewidth() > 2][0]
    assert list(mean_line.get_xdata()) == list(STEPS)
    assert len(ax.collections) == 1
    close(fig)


def test_learning_curve_figure_unequal_timesteps_plots_mean_x(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(1, 2), jitter=100)
    fig = plots.figure_learning_curve(exp, "deliveries")
    mean_line = [ln for ln in fig.axes[0].get_lines() if ln.get_linewidth() > 2][0]
    assert list(mean_line.get_xdata()) == [s + 150 for s in STEPS]
    close(fig)


def test_sampled_vs_deterministic_figure_content(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    fig = plots.figure_sampled_vs_deterministic(exp, "base")
    ax = fig.axes[0]
    assert len(ax.collections) == 2
    assert len(dashed(ax)) == 2
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:2] == ["sampled (mean of 2)", "deterministic (mean of 2)"]
    close(fig)


def test_training_diagnostics_figure_content(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)
    fig = plots.figure_training_diagnostics(exp, "base")
    axes = fig.axes
    assert len(axes) == 4
    assert [ax.get_title() for ax in axes] == [title for _, title in plots.DIAGNOSTICS]
    for ax in axes:
        # Two light seed lines and one bold mean; the blank first row is missing, not zero.
        assert len(ax.get_lines()) == 3
        for line in ax.get_lines():
            assert not any(y == 0 for y in line.get_ydata())
            assert not any(math.isnan(y) for y in line.get_ydata()[1:])
        assert ax.get_legend().get_texts()[0].get_text() == "mean of 2 seeds"
    close(fig)


def test_training_diagnostics_absent_column_is_marked_not_logged(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(1,))
    path = exp / "base" / "seed_1" / "progress.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(path, "w", newline="") as f:
        fields = [c for c in rows[0] if c != "train/value_loss"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    fig = plots.figure_training_diagnostics(exp, "base")
    value_ax = fig.axes[2]
    assert value_ax.get_legend() is None
    assert [t.get_text() for t in value_ax.texts] == ["not logged"]
    close(fig)


def test_training_diagnostics_missing_timesteps_column_raises_clear_error(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, seeds=(1,))
    path = exp / "base" / "seed_1" / "progress.csv"
    path.write_text("rollout/ep_rew_mean\n1.0\n")
    with pytest.raises(ValueError, match="time/total_timesteps"):
        plots.plot_training_diagnostics(exp, "base", tmp_path / "d.png")


def test_variant_comparison_figure_content(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("low", "high"), seeds=(1, 2, 3))
    fig = plots.figure_variant_comparison(exp)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["high", "low"]
    heights = [bar.get_height() for bar in ax.patches]
    assert heights[0] > heights[1]
    # The bar container plus its error bars.
    assert len(ax.containers) == 2
    assert any("n=3 seeds" in t.get_text() for t in ax.texts)
    close(fig)


def test_variant_comparison_colours_match_learning_curve(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("low", "high"), seeds=(1, 2))
    bars = plots.figure_variant_comparison(exp).axes[0].patches
    curve = plots.figure_learning_curve(exp, "deliveries").axes[0]
    bold = {
        ln.get_label().split()[0]: ln.get_color()
        for ln in curve.get_lines()
        if ln.get_linewidth() > 2
    }
    from matplotlib.colors import to_rgb

    # "high" ranks first and was listed second, so its bar has the second palette colour.
    assert to_rgb(bars[0].get_facecolor()) == pytest.approx(to_rgb(bold["high"]), abs=1e-6)
    assert to_rgb(bars[1].get_facecolor()) == pytest.approx(to_rgb(bold["low"]), abs=1e-6)


def test_tradeoff_figure_content(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, variants=("legacy", "base"), seeds=(1, 2, 3))
    fig = plots.figure_tradeoff(exp)
    ax = fig.axes[0]
    sizes = [len(c.get_offsets()) for c in ax.collections]
    # A dot per seed of each variant, then random and greedy.
    assert sizes == [3, 3, 1, 1]
    assert "collisions" in ax.get_xlabel() and "deliveries" in ax.get_ylabel()
    random_offset = ax.collections[2].get_offsets()[0]
    assert list(random_offset) == [9.0, 0.5]
    close(fig)


def test_tradeoff_without_baselines_has_only_seed_dots(tmp_path):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path, baselines=False)
    ax = plots.figure_tradeoff(exp).axes[0]
    assert [len(c.get_offsets()) for c in ax.collections] == [2]


def test_baseline_colours_agree_between_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgb

    exp = make_experiment(tmp_path)
    curve = plots.figure_learning_curve(exp, "deliveries").axes[0]
    trade = plots.figure_tradeoff(exp).axes[0]
    curve_colours = [to_rgb(ln.get_color()) for ln in dashed(curve)]
    trade_colours = [to_rgb(c.get_facecolor()[0][:3]) for c in trade.collections[1:]]
    assert [tuple(round(v, 4) for v in c) for c in curve_colours] == [
        tuple(round(v, 4) for v in c) for c in trade_colours
    ]


def test_report_tables_show_seed_counts_and_best_worst(tmp_path):
    exp = make_experiment(tmp_path, variants=("base",), seeds=(1, 2, 3))
    text = plots.make_report(exp).read_text()
    # Final deliveries are 2 + 2 + 0.1 * seed, so seed 3 is best and seed 1 worst.
    row = [ln for ln in text.splitlines() if ln.startswith("| base |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:2] == ["base", "3"]
    assert cells[2] == "4.20 ± 0.10"
    assert cells[3] == "3.20 ± 0.10"
    assert cells[4] == "3 (4.30)"
    assert cells[5] == "1 (4.10)"


def test_plot_all_propagates_real_drawing_errors(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    exp = make_experiment(tmp_path)

    def broken(*_args, **_kwargs):
        raise ValueError("genuine bug")

    monkeypatch.setattr(plots, "plot_training_diagnostics", broken)
    with pytest.raises(ValueError, match="genuine bug"):
        plots.plot_all(exp)


def many(tmp_path, n):
    """An experiment with `n` variants named v00, v01, ... and two seeds."""
    return make_experiment(tmp_path, variants=tuple(f"v{i:02d}" for i in range(n)), seeds=(1, 2))


def mean_lines(ax):
    """The bold, non-dashed mean lines of `ax`."""
    return [ln for ln in ax.get_lines() if ln.get_linestyle() != "--" and ln.get_linewidth() > 2]


def test_variant_style_is_distinct_for_up_to_thirty_variants():
    names = [f"v{i}" for i in range(30)]
    styles = [plots.variant_style(names, n) for n in names]
    assert len({(s["color"], s["linestyle"]) for s in styles}) == 30
    assert len({(s["color"], s["marker"]) for s in styles}) == 30
    assert all(s["linestyle"] != "--" for s in styles)
    assert plots.variant_style(names, "v3") == plots.variant_style(names, "v3")


def test_many_variants_mean_lines_are_distinct_without_seed_lines(tmp_path):
    pytest.importorskip("matplotlib")
    exp = many(tmp_path, 10)
    fig = plots.figure_learning_curve(exp, "deliveries")
    ax = fig.axes[0]
    pairs = {(ln.get_color(), ln.get_linestyle()) for ln in mean_lines(ax)}
    assert len(pairs) == 10
    # No thin per-seed lines: only the 10 means and the 2 baselines.
    assert len(ax.get_lines()) == 12
    # More than eight variants also drops the shaded bands.
    assert len(ax.collections) == 0
    close(fig)


def test_crowded_but_not_huge_keeps_light_bands(tmp_path):
    pytest.importorskip("matplotlib")
    ax = plots.figure_learning_curve(many(tmp_path, 6), "deliveries").axes[0]
    assert len(ax.get_lines()) == 6 + 2
    assert len(ax.collections) == 6
    assert all(c.get_alpha() < 0.18 for c in ax.collections)


def test_few_variants_keep_seed_lines_and_inside_legend(tmp_path):
    pytest.importorskip("matplotlib")
    fig = plots.figure_learning_curve(many(tmp_path, 4), "deliveries")
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 4 * 2 + 4 + 2
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 < ax.get_window_extent().x1
    close(fig)


def test_legend_is_outside_the_axes_for_many_variants(tmp_path):
    pytest.importorskip("matplotlib")
    exp = many(tmp_path, 10)
    for fig in (plots.figure_learning_curve(exp, "deliveries"), plots.figure_tradeoff(exp)):
        fig.canvas.draw()
        ax = fig.axes[0]
        assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
        close(fig)


def test_tradeoff_legend_inside_for_six_variants_or_fewer(tmp_path):
    pytest.importorskip("matplotlib")
    fig = plots.figure_tradeoff(many(tmp_path, 6))
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_legend().get_window_extent().x0 < ax.get_window_extent().x1
    close(fig)


def test_tradeoff_variant_markers_differ_and_baselines_stay_distinct(tmp_path):
    pytest.importorskip("matplotlib")
    ax = plots.figure_tradeoff(many(tmp_path, 10)).axes[0]
    paths = [tuple(map(tuple, c.get_paths()[0].vertices.round(3).tolist())) for c in ax.collections]
    variant_markers, baseline_markers = paths[:10], paths[10:]
    assert len(set(variant_markers)) == 10
    assert not set(variant_markers) & set(baseline_markers)
    assert len(set(baseline_markers)) == 2


def test_comparison_bars_use_the_shared_variant_colours(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgb

    exp = many(tmp_path, 10)
    names = [f"v{i:02d}" for i in range(10)]
    ax = plots.figure_variant_comparison(exp).axes[0]
    # Higher index scores higher, so bars run v09 .. v00.
    for bar, name in zip(ax.patches, reversed(names), strict=True):
        expected = to_rgb(plots.variant_style(names, name)["color"])
        assert to_rgb(bar.get_facecolor()) == pytest.approx(expected, abs=1e-6)


def _write_generalisation(exp: Path, variants=("fixed", "rand")) -> None:
    with open(exp / "generalisation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=experiment.GENERALISATION_COLUMNS)
        writer.writeheader()
        for vi, variant in enumerate(variants):
            for seed in (1, 2):
                for layout in ("default", "wide_gap"):
                    for n in (2, 3):
                        writer.writerow(
                            {
                                "variant": variant,
                                "seed": seed,
                                "layout": layout,
                                "n_robots": n,
                                "mean_deliveries": 3.0 + vi + n * 0.5 + seed * 0.1,
                                "ci_deliveries": 0.2,
                                "mean_collisions": 1.0,
                                "ci_collisions": 0.1,
                            }
                        )


def test_plot_generalisation_writes_png(tmp_path):
    from multi_agent_warehouse.plots_generalisation import (
        generalisation_grid,
        plot_generalisation,
    )

    exp = make_experiment(tmp_path)
    _write_generalisation(exp)

    out = plot_generalisation(exp, tmp_path / "g.png")

    assert nonempty(out)
    layouts, counts, grid = generalisation_grid(
        plots._read_csv(exp / "generalisation.csv"), "fixed"
    )
    assert layouts == ["default", "wide_gap"] and counts == [2, 3]
    assert grid[0, 0] == pytest.approx(3.0 + 1.0 + 0.15)


def test_plot_all_includes_generalisation_when_present(tmp_path):
    exp = make_experiment(tmp_path)
    _write_generalisation(exp)

    written = plots.plot_all(exp)

    assert exp / "plots" / "generalisation.png" in written
