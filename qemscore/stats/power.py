"""Critical-difference resolution and planned Wilcoxon-Holm power analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from qemscore.stats.inference import PlannedComparisonFamily

# Demsar 2006, Table 5(a), alpha=0.05, two-tailed Nemenyi critical values.
NEMENYI_Q_05 = {
    2: 1.960,
    3: 2.344,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


@dataclass(frozen=True)
class CriticalDifferenceResolution:
    n_methods: int
    rank_separation: float
    q_alpha: float
    minimum_paired_cells: int
    label: str = "critical-difference resolution"


@dataclass(frozen=True)
class WilcoxonHolmPower:
    comparison_family: str
    planned_comparisons: int
    standardized_shift: float
    alpha_familywise: float
    target_power: float
    minimum_paired_blocks: int
    estimated_power: float
    simulations: int
    seed: int


def critical_difference(n_cells: int, n_methods: int, *, q_alpha: float | None = None) -> float:
    """Return the Nemenyi critical difference, which is a resolution measure."""

    if n_cells <= 0 or n_methods < 2:
        raise ValueError("n_cells must be positive and n_methods must be at least two")
    if q_alpha is None:
        try:
            q_alpha = NEMENYI_Q_05[n_methods]
        except KeyError as exc:
            raise ValueError("q_alpha is required outside the tabulated k=2..10 range") from exc
    if not np.isfinite(q_alpha) or q_alpha <= 0:
        raise ValueError("q_alpha must be finite and positive")
    return float(q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6 * n_cells)))


def critical_difference_resolution(
    n_methods: int,
    *,
    rank_separation: float = 1.0,
    q_alpha: float | None = None,
) -> CriticalDifferenceResolution:
    """Return cells needed for a requested mean-rank separation, not 80% power."""

    if rank_separation <= 0:
        raise ValueError("rank_separation must be positive")
    if q_alpha is None:
        try:
            q_alpha = NEMENYI_Q_05[n_methods]
        except KeyError as exc:
            raise ValueError("q_alpha is required outside the tabulated k=2..10 range") from exc
    minimum = math.ceil(
        q_alpha**2 * n_methods * (n_methods + 1) / (6 * rank_separation**2)
    )
    return CriticalDifferenceResolution(
        n_methods=n_methods,
        rank_separation=float(rank_separation),
        q_alpha=float(q_alpha),
        minimum_paired_cells=minimum,
    )


def wilcoxon_holm_power_analysis(
    family: PlannedComparisonFamily,
    *,
    standardized_shift: float = 0.5,
    alpha_familywise: float = 0.05,
    target_power: float = 0.80,
    simulations: int = 10_000,
    seed: int = 123,
) -> WilcoxonHolmPower:
    """Estimate blocks for a normal location shift under the frozen Holm family.

    The most stringent Holm threshold, alpha divided by the number of planned
    comparisons, defines the conservative per-comparison target. Roster size is
    deliberately absent from this API.
    """

    if standardized_shift <= 0 or not 0 < alpha_familywise < 1 or not 0 < target_power < 1:
        raise ValueError("shift must be positive and alpha/power must lie in (0, 1)")
    if simulations < 1_000:
        raise ValueError("at least 1,000 simulations are required")
    comparisons = len(family.comparisons)
    local_alpha = alpha_familywise / comparisons
    # The t-test approximation localizes the search. Exact signed-rank p-values
    # determine the returned integer.
    rough = math.ceil(
        (
            (
                stats.norm.ppf(1 - local_alpha / 2)
                + stats.norm.ppf(target_power)
            )
            / standardized_shift
        )
        ** 2
    )
    start = max(5, rough - 5)
    stop = rough + 12
    chosen = stop
    chosen_power = 0.0
    for n_blocks in range(start, stop + 1):
        rng = np.random.default_rng(seed)
        differences = rng.normal(
            loc=-standardized_shift, scale=1.0, size=(simulations, n_blocks)
        )
        p_values = stats.wilcoxon(
            differences, axis=1, alternative="two-sided", method="exact"
        ).pvalue
        estimated = float(np.mean(p_values < local_alpha))
        if estimated >= target_power:
            chosen = n_blocks
            chosen_power = estimated
            break
    return WilcoxonHolmPower(
        comparison_family=family.name,
        planned_comparisons=comparisons,
        standardized_shift=float(standardized_shift),
        alpha_familywise=float(alpha_familywise),
        target_power=float(target_power),
        minimum_paired_blocks=chosen,
        estimated_power=chosen_power,
        simulations=simulations,
        seed=seed,
    )
