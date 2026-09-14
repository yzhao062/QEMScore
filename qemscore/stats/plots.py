"""Headless critical-difference diagrams."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from qemscore.stats.power import critical_difference


def critical_difference_diagram(
    average_ranks: Mapping[str, float],
    n_cells: int,
    path: str | Path,
    *,
    q_alpha: float | None = None,
) -> Path:
    """Render a compact Demsar-style mean-rank axis and CD bar to PDF."""

    if len(average_ranks) < 2:
        raise ValueError("a critical-difference diagram requires at least two methods")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cd = critical_difference(n_cells, len(average_ranks), q_alpha=q_alpha)
    ordered = sorted(average_ranks.items(), key=lambda pair: pair[1])
    figure, axis = plt.subplots(figsize=(7.0, 1.3 + 0.28 * len(ordered)))
    axis.set_xlim(len(ordered) + 0.25, 0.75)
    axis.set_ylim(-0.4, len(ordered) + 1.2)
    axis.xaxis.tick_top()
    axis.set_xticks(range(1, len(ordered) + 1))
    axis.set_yticks([])
    axis.set_xlabel("Mean rank (lower is better)")
    axis.xaxis.set_label_position("top")
    for index, (method, rank) in enumerate(ordered):
        y = len(ordered) - index - 0.2
        axis.plot([rank, rank], [0, y], color="0.75", linewidth=0.8)
        axis.plot(rank, y, "o", color="black", markersize=3)
        axis.text(rank, y + 0.12, f"{method} ({rank:.2f})", ha="center", fontsize=8)
    left = 1.0
    axis.plot([left, left + cd], [len(ordered) + 0.6] * 2, color="black", linewidth=1.5)
    axis.plot([left, left], [len(ordered) + 0.48, len(ordered) + 0.72], color="black")
    axis.plot([left + cd, left + cd], [len(ordered) + 0.48, len(ordered) + 0.72], color="black")
    axis.text(left + cd / 2, len(ordered) + 0.78, f"CD = {cd:.2f}", ha="center", fontsize=8)
    for spine in ("left", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output
