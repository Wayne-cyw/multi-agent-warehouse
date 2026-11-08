"""The generalisation heatmap: deliveries by layout and robot count, one panel per variant.

It reads `generalisation.csv` as written by `experiment.evaluate_generalisation`. Held-out layouts
(never trained on) are marked in the row labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from multi_agent_warehouse.config import HELDOUT_LAYOUTS
from multi_agent_warehouse.plots import _finish, _float, _pyplot, _read_csv


def generalisation_grid(
    rows: list[dict[str, Any]], variant: str
) -> tuple[list[str], list[int], np.ndarray]:
    """Layouts, robot counts and the seed-mean deliveries of `variant` as a (layout, count) grid.

    Layouts keep their order of first appearance and counts are ascending. A cell with no row is
    NaN.
    """
    mine = [r for r in rows if r["variant"] == variant]
    layouts = list(dict.fromkeys(r["layout"] for r in mine))
    counts = sorted({int(r["n_robots"]) for r in mine})
    grid = np.full((len(layouts), len(counts)), np.nan)
    for i, layout in enumerate(layouts):
        for j, count in enumerate(counts):
            values = [
                _float(r["mean_deliveries"])
                for r in mine
                if r["layout"] == layout and int(r["n_robots"]) == count
            ]
            if values:
                grid[i, j] = float(np.mean(values))
    return layouts, counts, grid


def plot_generalisation(exp_dir: Path, out: Path) -> Path:
    """Write a heatmap per variant (rows layouts, columns robot counts) to `out`.

    All panels share one colour scale, so variants compare directly. Raises FileNotFoundError when
    `generalisation.csv` is missing or empty.
    """
    path = Path(exp_dir) / "generalisation.csv"
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"no rows in {path}")
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(variants), figsize=(4.6 * len(variants) + 1.2, 4.2), squeeze=False, sharey=True
    )
    grids = {v: generalisation_grid(rows, v) for v in variants}
    top = max(float(np.nanmax(g)) for _, _, g in grids.values())
    image = None
    for ax, variant in zip(axes[0], variants, strict=True):
        layouts, counts, grid = grids[variant]
        image = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=top, aspect="auto")
        ax.set_xticks(range(len(counts)), [str(c) for c in counts])
        ax.set_yticks(
            range(len(layouts)),
            [f"{name} (held out)" if name in HELDOUT_LAYOUTS else name for name in layouts],
        )
        ax.set_xlabel("robots")
        ax.set_title(variant)
        for i in range(len(layouts)):
            for j in range(len(counts)):
                colour = "white" if grid[i, j] < 0.6 * top else "black"
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color=colour)
    fig.colorbar(image, ax=list(axes[0]), label="deliveries per episode", shrink=0.9)
    return _finish(fig, out)
