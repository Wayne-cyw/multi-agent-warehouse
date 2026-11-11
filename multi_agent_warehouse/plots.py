"""Figures and a markdown report for an experiment directory written by `experiment.py`.

`plot_all` reads `eval_curve.csv`, `baselines.csv`, `manifest.json` and the per-run
`progress.csv` files of one experiment and writes PNGs into `<exp_dir>/plots/` plus `report.md`.
Aggregation helpers (`seed_band`, `aggregate_curve`, `final_results`, ...) are plain `csv` and
`numpy`; matplotlib is imported inside the drawing functions only, so this module imports without
the optional `plot` extra.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Two-sided 95% Student-t critical values for df = 1..10; 1.96 beyond that.
T_CRITICAL = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)
METRICS = {
    "deliveries": "deliveries per episode",
    "collisions": "collisions per episode",
}
BASELINE_COLOURS = {"random": "#888888", "greedy": "#444444"}
# Twelve distinct colours (colour-blind-safe core plus extras) for variants.
PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#7B3F98",
    "#56B4E9",
    "#CC79A7",
    "#8C564B",
    "#8A9A00",
    "#1B2A6B",
    "#17BECF",
    "#222222",
)
# Past the palette, line styles cycle (never "--", which marks the baselines).
LINESTYLES = ("-", "-.", ":")
# Variant scatter markers; the baselines use "X" and "P", which are left out here.
MARKERS = ("o", "s", "^", "D", "v", "<", ">", "h", "p", "*")
# Beyond this many variants the figures drop per-seed lines and move the legend outside.
CROWDED = 4
# Beyond this many variants the learning curves also drop the shaded bands.
VERY_CROWDED = 8
STEPS_COLUMN = "time/total_timesteps"
DIAGNOSTICS = (
    ("rollout/ep_rew_mean", "episode reward (training mean)"),
    ("train/entropy_loss", "entropy loss"),
    ("train/value_loss", "value loss"),
    ("train/explained_variance", "explained variance"),
)
# (timesteps, mean, lower, upper) of one aggregated curve point.
CurvePoint = tuple[float, float, float, float]


def seed_band(values: Sequence[float]) -> tuple[float, float, float]:
    """The `(mean, lower, upper)` of a 95% t-interval over `values`; `(v, v, v)` for one value.

    Preconditions:
        - len(values) >= 1
    """
    n = len(values)
    mean = float(np.mean(values))
    if n == 1:
        return (mean, mean, mean)
    crit = T_CRITICAL[n - 2] if n - 1 <= len(T_CRITICAL) else 1.96
    half = crit * float(np.std(values, ddof=1)) / math.sqrt(n)
    return (mean, mean - half, mean + half)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """The rows of the CSV at `path`, or none if it does not exist."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(cell: str | None) -> float:
    """`cell` as a float; a blank, missing or unparsable cell is NaN (missing, not zero)."""
    try:
        return float(cell) if cell not in (None, "") else math.nan
    except ValueError:
        return math.nan


def load_curve(exp_dir: Path) -> list[dict[str, Any]]:
    """The rows of `eval_curve.csv` with seed, timesteps and the metric cells as numbers."""
    rows = []
    for raw in _read_csv(Path(exp_dir) / "eval_curve.csv"):
        row: dict[str, Any] = dict(raw)
        row["seed"] = int(raw["seed"])
        row["timesteps"] = int(raw["timesteps"])
        for key in ("mean_deliveries", "mean_collisions"):
            row[key] = float(raw[key])
        rows.append(row)
    return rows


def load_baselines(exp_dir: Path) -> dict[str, dict[str, float]]:
    """`{policy: {"deliveries": mean, "collisions": mean}}` from `baselines.csv`, maybe empty."""
    return {
        row["policy"]: {
            "deliveries": float(row["mean_deliveries"]),
            "collisions": float(row["mean_collisions"]),
        }
        for row in _read_csv(Path(exp_dir) / "baselines.csv")
    }


def variants_of(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The variant names in `rows`, in order of first appearance."""
    return list(dict.fromkeys(row["variant"] for row in rows))


def seed_series(
    rows: Sequence[dict[str, Any]], variant: str, mode: str, metric: str
) -> dict[int, list[tuple[int, float]]]:
    """`{seed: [(timesteps, value), ...]}` of one variant and mode, each list oldest first."""
    series: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        if row["variant"] == variant and row["mode"] == mode:
            series.setdefault(row["seed"], []).append((row["timesteps"], row[f"mean_{metric}"]))
    return {seed: sorted(points) for seed, points in sorted(series.items())}


def aggregate_curve(series: dict[int, list[tuple[int, float]]]) -> list[CurvePoint]:
    """Mean and 95% seed band at each checkpoint, as `(timesteps, mean, lower, upper)` points.

    Checkpoints are grouped by their index within a seed (the k-th checkpoint of every seed), not
    by exact timestep, because checkpoints are saved at rollout ends and so differ slightly
    between seeds (e.g. 3072 vs 3000). The point's x is the mean timestep of its group, and a
    group holds only the seeds that reached that index.
    """
    longest = max((len(points) for points in series.values()), default=0)
    aggregated = []
    for k in range(longest):
        group = [points[k] for points in series.values() if len(points) > k]
        mean, lower, upper = seed_band([value for _, value in group])
        aggregated.append((float(np.mean([t for t, _ in group])), mean, lower, upper))
    return aggregated


def final_results(rows: Sequence[dict[str, Any]], mode: str) -> dict[tuple[str, int], dict]:
    """The row with the largest `timesteps` for each `(variant, seed)` in `mode`."""
    final: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["variant"], row["seed"])
        if row["mode"] == mode and (key not in final or row["timesteps"] > final[key]["timesteps"]):
            final[key] = row
    return final


def rank_variants(
    rows: Sequence[dict[str, Any]], mode: str = "sampled"
) -> list[tuple[str, float, float, int]]:
    """`(variant, mean deliveries, seed std, n seeds)` at the final checkpoint, best first.

    The std uses ddof=1 and is 0 for a single seed.
    """
    by_variant: dict[str, list[float]] = {}
    for (variant, _), row in final_results(rows, mode).items():
        by_variant.setdefault(variant, []).append(row["mean_deliveries"])
    ranked = [
        (v, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))
        for v, vals in by_variant.items()
    ]
    return sorted(ranked, key=lambda item: -item[1])


def aggregate_progress(
    runs: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds of `(timesteps, values)` series on a shared timestep grid.

    The grid is the longest series' timesteps; each other series is linearly interpolated onto it
    and is missing (NaN) beyond its own range. NaN values are dropped before interpolating, so
    blank cells count as missing, not zero. Grid points where every seed is missing are NaN.
    """
    cleaned = []
    for steps, values in runs:
        x, y = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
        keep = ~np.isnan(x) & ~np.isnan(y)
        if keep.any():
            cleaned.append((x[keep], y[keep]))
    if not cleaned:
        return np.array([]), np.array([])
    grid = max(cleaned, key=lambda xy: len(xy[0]))[0]
    stacked = np.vstack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in cleaned])
    with warnings.catch_warnings():
        # Grid points where every seed is missing stay NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        return grid, np.nanmean(stacked, axis=0)


def load_progress(exp_dir: Path, variant: str) -> dict[int, dict[str, list[float]]]:
    """`{seed: {column: values}}` from each `<variant>/seed_<s>/progress.csv`; blanks are NaN."""
    runs: dict[int, dict[str, list[float]]] = {}
    for path in sorted((Path(exp_dir) / variant).glob("seed_*/progress.csv")):
        rows = _read_csv(path)
        if not rows:
            continue
        seed = int(path.parent.name.removeprefix("seed_"))
        runs[seed] = {column: [_float(row.get(column)) for row in rows] for column in rows[0]}
    return runs


def _pyplot() -> Any:
    """`matplotlib.pyplot` with the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def variant_style(variants: Sequence[str], variant: str) -> dict[str, str]:
    """The `color`, `linestyle` and `marker` of `variant`, fixed by its place in `variants`.

    Every figure of one experiment passes the same `variants` (from `variants_of`), so a variant
    keeps its look everywhere. Colours cycle through `PALETTE`; line style changes each time the
    palette wraps and the marker cycles independently, so no two of ~36 variants look identical.

    Preconditions:
        - variant in variants
    """
    i = list(variants).index(variant)
    return {
        "color": PALETTE[i % len(PALETTE)],
        "linestyle": LINESTYLES[(i // len(PALETTE)) % len(LINESTYLES)],
        "marker": MARKERS[i % len(MARKERS)],
    }


def _legend(ax: Any, outside: bool) -> None:
    """Draw the legend, to the right of the axes when `outside`."""
    if outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)
    else:
        ax.legend(loc="best", fontsize=8)


def _finish(fig: Any, out: Path) -> Path:
    """Save `fig` to `out` (creating the folder), close it and return `out`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _pyplot().close(fig)
    return out


def _set_timestep_axis(ax: Any, xs: Sequence[float]) -> None:
    """Log-scale the x axis when every x is positive, and label it."""
    if len(xs) and min(xs) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("training timesteps")
    ax.grid(alpha=0.25)


def _draw_baselines(ax: Any, baselines: dict[str, dict[str, float]], metric: str) -> None:
    """Dashed horizontal lines for the random and greedy baselines, labelled with their values."""
    for policy, stats in baselines.items():
        ax.axhline(
            stats[metric],
            linestyle="--",
            linewidth=1.2,
            color=BASELINE_COLOURS.get(policy, "#666666"),
            label=f"{policy} baseline ({stats[metric]:.2f})",
        )


def figure_learning_curve(exp_dir: Path, metric: str, mode: str = "sampled") -> Any:
    """Evaluation metric against timesteps: a line per seed, the variant mean and its 95% band.

    Seeds are aggregated per checkpoint index (see `aggregate_curve`). Dashed lines mark the
    random and greedy baselines when `baselines.csv` exists. Raises ValueError for an unknown
    metric or when no row of `mode` exists.

    Preconditions:
        - mode in ("sampled", "deterministic")
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
    rows = load_curve(exp_dir)
    variants = variants_of(rows)
    if not any(row["mode"] == mode for row in rows):
        raise ValueError(f"no {mode} rows in eval_curve.csv of {exp_dir}")
    crowded = len(variants) > CROWDED
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10 if crowded else 8, 5.5 if crowded else 5))
    xs: list[float] = []
    for variant in variants:
        series = seed_series(rows, variant, mode, metric)
        style = variant_style(variants, variant)
        for points in series.values():
            xs += [t for t, _ in points]
            if not crowded:
                ax.plot(
                    *zip(*points, strict=False),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=0.3,
                    linewidth=0.8,
                )
        agg = aggregate_curve(series)
        if agg:
            t, mean, lower, upper = (np.array(col) for col in zip(*agg, strict=False))
            ax.plot(
                t,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2 if crowded else 2.5,
                label=f"{variant} (mean of {len(series)})",
            )
            if len(variants) <= VERY_CROWDED:
                alpha = 0.10 if crowded else 0.18
                ax.fill_between(t, lower, upper, color=style["color"], alpha=alpha, linewidth=0)
    _draw_baselines(ax, load_baselines(exp_dir), metric)
    _set_timestep_axis(ax, xs)
    ax.set_ylabel(METRICS[metric])
    ax.set_title(f"{Path(exp_dir).name}: {METRICS[metric]} ({mode} policy)")
    _legend(ax, crowded)
    return fig


def plot_learning_curve(exp_dir: Path, metric: str, out: Path, mode: str = "sampled") -> Path:
    """Draw `figure_learning_curve` and save it to `out`; see that function."""
    return _finish(figure_learning_curve(exp_dir, metric, mode), out)


def figure_sampled_vs_deterministic(exp_dir: Path, variant: str) -> Any:
    """Deliveries of one variant for the sampled and deterministic policy, mean and 95% band.

    Raises ValueError naming the missing mode when `eval_curve.csv` lacks one.
    """
    rows = load_curve(exp_dir)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    xs: list[float] = []
    for mode, colour in (("sampled", PALETTE[0]), ("deterministic", PALETTE[1])):
        series = seed_series(rows, variant, mode, "deliveries")
        if not series:
            plt.close(fig)
            raise ValueError(f"no {mode} rows for variant {variant!r} in {exp_dir}")
        t, mean, lower, upper = (
            np.array(col) for col in zip(*aggregate_curve(series), strict=False)
        )
        xs += list(t)
        ax.plot(
            t,
            mean,
            color=colour,
            linewidth=2.5,
            marker="o",
            label=f"{mode} (mean of {len(series)})",
        )
        ax.fill_between(t, lower, upper, color=colour, alpha=0.18, linewidth=0)
    _draw_baselines(ax, load_baselines(exp_dir), "deliveries")
    _set_timestep_axis(ax, xs)
    ax.set_ylabel(METRICS["deliveries"])
    ax.set_title(f"{Path(exp_dir).name}: {variant}, sampled vs deterministic policy")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_sampled_vs_deterministic(exp_dir: Path, variant: str, out: Path) -> Path:
    """Draw `figure_sampled_vs_deterministic` and save it to `out`; see that function."""
    return _finish(figure_sampled_vs_deterministic(exp_dir, variant), out)


def figure_training_diagnostics(exp_dir: Path, variant: str) -> Any:
    """A 2x2 grid of training reward, entropy loss, value loss and explained variance.

    Each panel has a light line per seed and the mean over seeds in bold. Raises
    FileNotFoundError when the variant has no `progress.csv`.
    """
    runs = load_progress(exp_dir, variant)
    if not runs:
        raise FileNotFoundError(f"no progress.csv for variant {variant!r} in {exp_dir}")
    if any(STEPS_COLUMN not in run for run in runs.values()):
        raise ValueError(f"progress.csv of variant {variant!r} lacks the {STEPS_COLUMN} column")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (column, title) in zip(axes.flat, DIAGNOSTICS, strict=False):
        series = [(r[STEPS_COLUMN], r[column]) for r in runs.values() if column in r]
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        if not series:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes)
            continue
        for steps, values in series:
            ax.plot(steps, values, color=PALETTE[0], alpha=0.25, linewidth=0.8)
        grid, mean = aggregate_progress(series)
        ax.plot(grid, mean, color=PALETTE[0], linewidth=2.2, label=f"mean of {len(series)} seeds")
        ax.legend(loc="best", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("training timesteps")
    fig.suptitle(f"{Path(exp_dir).name}: {variant} training diagnostics")
    fig.tight_layout()
    return fig


def plot_training_diagnostics(exp_dir: Path, variant: str, out: Path) -> Path:
    """Draw `figure_training_diagnostics` and save it to `out`; see that function."""
    return _finish(figure_training_diagnostics(exp_dir, variant), out)


def figure_variant_comparison(exp_dir: Path) -> Any:
    """Final-checkpoint sampled deliveries per variant, best first, error bars = seed std."""
    ranked = rank_variants(load_curve(exp_dir), "sampled")
    if not ranked:
        raise ValueError(f"no sampled rows in eval_curve.csv of {exp_dir}")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(4.0, 1.6 * len(ranked) + 2.5), 5))
    names = [name for name, *_ in ranked]
    # Colour by variant order, as in the other figures.
    order = variants_of(load_curve(exp_dir))
    colours = [variant_style(order, name)["color"] for name in names]
    means, stds = [m for _, m, _, _ in ranked], [s for _, _, s, _ in ranked]
    ax.bar(names, means, yerr=stds, capsize=6, color=colours, alpha=0.85)
    top = max(m + s for m, s in zip(means, stds, strict=False))
    for i, (_, mean, std, n) in enumerate(ranked):
        ax.text(i, mean + std + 0.02 * top, f"{mean:.2f}\nn={n} seeds", ha="center", fontsize=9)
    ax.set_ylim(0, top * 1.15 if top > 0 else 1)
    ax.set_ylabel(METRICS["deliveries"] + " (final checkpoint)")
    ax.set_title(f"{Path(exp_dir).name}: final deliveries by variant (mean, error bar = seed std)")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_variant_comparison(exp_dir: Path, out: Path) -> Path:
    """Draw `figure_variant_comparison` and save it to `out`; see that function."""
    return _finish(figure_variant_comparison(exp_dir), out)


def figure_tradeoff(exp_dir: Path) -> Any:
    """Final deliveries (y) against collisions (x): a dot per variant and seed, plus baselines."""
    rows = load_curve(exp_dir)
    variants = variants_of(rows)
    final = final_results(rows, "sampled")
    if not final:
        raise ValueError(f"no sampled rows in eval_curve.csv of {exp_dir}")
    plt = _pyplot()
    outside = len(variants) > 6
    fig, ax = plt.subplots(figsize=(9 if outside else 7, 5.5))
    for variant in variants:
        points = [r for (v, _), r in final.items() if v == variant]
        style = variant_style(variants, variant)
        ax.scatter(
            [r["mean_collisions"] for r in points],
            [r["mean_deliveries"] for r in points],
            color=style["color"],
            marker=style["marker"],
            s=55,
            alpha=0.85,
            label=f"{variant} (n={len(points)})",
        )
    for (policy, stats), marker in zip(load_baselines(exp_dir).items(), ("X", "P"), strict=False):
        ax.scatter(
            stats["collisions"],
            stats["deliveries"],
            marker=marker,
            s=110,
            color=BASELINE_COLOURS.get(policy, "#666666"),
            label=f"{policy} baseline",
        )
    ax.set_xlabel(METRICS["collisions"])
    ax.set_ylabel(METRICS["deliveries"])
    ax.set_title(f"{Path(exp_dir).name}: deliveries vs collisions (final checkpoint)")
    ax.grid(alpha=0.25)
    _legend(ax, outside)
    return fig


def plot_tradeoff(exp_dir: Path, out: Path) -> Path:
    """Draw `figure_tradeoff` and save it to `out`; see that function."""
    return _finish(figure_tradeoff(exp_dir), out)


def _mean_std(values: Sequence[float]) -> str:
    """`values` as "mean ± std" (std is ddof=1, 0 for one value)."""
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{np.mean(values):.2f} ± {std:.2f}"


def _mode_table(rows: Sequence[dict[str, Any]], mode: str) -> list[str]:
    """Markdown lines of the final-checkpoint table for `mode`, empty if it has no rows."""
    final = final_results(rows, mode)
    if not final:
        return []
    lines = [
        f"### {mode.capitalize()} policy",
        "",
        "| variant | seeds | deliveries (mean ± std) | collisions (mean ± std) "
        "| best seed | worst seed |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants_of(rows):
        runs = sorted((r for (v, _), r in final.items() if v == variant), key=lambda r: r["seed"])
        if not runs:
            continue
        best = max(runs, key=lambda r: r["mean_deliveries"])
        worst = min(runs, key=lambda r: r["mean_deliveries"])
        lines.append(
            f"| {variant} | {len(runs)} | {_mean_std([r['mean_deliveries'] for r in runs])} "
            f"| {_mean_std([r['mean_collisions'] for r in runs])} "
            f"| {best['seed']} ({best['mean_deliveries']:.2f}) "
            f"| {worst['seed']} ({worst['mean_deliveries']:.2f}) |"
        )
    return [*lines, ""]


def make_report(exp_dir: Path) -> Path:
    """Write `<exp_dir>/report.md`: manifest facts, final-checkpoint tables and every figure.

    Every PNG in `<exp_dir>/plots/` is linked with a relative path. Missing inputs leave their
    section out.
    """
    exp_dir = Path(exp_dir)
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rows = load_curve(exp_dir)
    lines = [f"# Experiment {manifest.get('name', exp_dir.name)}", ""]
    if manifest:
        lines += [
            f"- Train seeds: {', '.join(str(s) for s in manifest.get('train_seeds', []))}",
            f"- Evaluation seed base: {manifest.get('eval_seed_base')}",
            f"- Evaluation episodes: {manifest.get('n_eval_episodes')}",
            f"- Git commit: {manifest.get('git_commit')}",
            "",
        ]
    lines += ["## Final-checkpoint results", ""]
    for mode in ("sampled", "deterministic"):
        lines += _mode_table(rows, mode)
    baselines = _read_csv(exp_dir / "baselines.csv")
    if baselines:
        lines += [
            "## Baselines",
            "",
            "| policy | deliveries | collisions | episodes |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {b['policy']} | {float(b['mean_deliveries']):.2f} "
            f"| {float(b['mean_collisions']):.2f} | {b['n_episodes']} |"
            for b in baselines
        ]
        lines.append("")
    pngs = sorted((exp_dir / "plots").glob("*.png"))
    if pngs:
        lines += ["## Figures", ""]
        for png in pngs:
            lines += [f"![{png.stem.replace('_', ' ')}](plots/{png.name})", ""]
    report = exp_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


def plot_all(exp_dir: Path) -> list[Path]:
    """Write every applicable figure into `<exp_dir>/plots/`, then the report.

    Figures whose input is absent (diagnostics without `progress.csv`, a variant without both
    modes) are skipped by checking the inputs first; any error while drawing propagates.
    Returns the figure paths followed by the report path.
    Raises FileNotFoundError when `eval_curve.csv` has no rows.
    """
    exp_dir = Path(exp_dir)
    rows = load_curve(exp_dir)
    if not rows:
        raise FileNotFoundError(f"no rows in {exp_dir / 'eval_curve.csv'}")
    plots = exp_dir / "plots"
    written = [
        plot_learning_curve(exp_dir, "deliveries", plots / "learning_curve_deliveries.png"),
        plot_learning_curve(exp_dir, "collisions", plots / "learning_curve_collisions.png"),
    ]
    for variant in variants_of(rows):
        modes = {row["mode"] for row in rows if row["variant"] == variant}
        if len(modes) == 2:
            written.append(
                plot_sampled_vs_deterministic(
                    exp_dir, variant, plots / f"sampled_vs_deterministic_{variant}.png"
                )
            )
        if load_progress(exp_dir, variant):
            written.append(
                plot_training_diagnostics(exp_dir, variant, plots / f"diagnostics_{variant}.png")
            )
    written.append(plot_variant_comparison(exp_dir, plots / "variant_comparison.png"))
    written.append(plot_tradeoff(exp_dir, plots / "tradeoff.png"))
    if (exp_dir / "generalisation.csv").exists():
        from multi_agent_warehouse.plots_generalisation import plot_generalisation

        written.append(plot_generalisation(exp_dir, plots / "generalisation.png"))
    return [*written, make_report(exp_dir)]


def promote(
    exp_dir: Path, names: Sequence[str], dest: Path = Path("docs/evidence")
) -> list[Path]:
    """Copy `<exp_dir>/plots/<name>.png` for each name into `dest`; return the new paths.

    Raises FileNotFoundError naming the first missing figure, before anything is copied.
    """
    sources = [Path(exp_dir) / "plots" / f"{name}.png" for name in names]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"figure {source.name} not found in {source.parent}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy(source, dest / source.name)) for source in sources]
