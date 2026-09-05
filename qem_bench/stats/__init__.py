"""Cell-level descriptive, inferential, robustness, and resolution statistics."""

from qem_bench.stats.bootstrap import BootstrapInterval, circuit_blocked_bootstrap
from qem_bench.stats.descriptive import macro_mean_iqr
from qem_bench.stats.incremental_value import evaluate_incremental_value
from qem_bench.stats.inference import (
    PlannedComparisonFamily,
    bias_by_stratum,
    friedman_rank_test,
    holm_adjust,
    mean_ranks,
    paired_wilcoxon_holm,
    wilcoxon_rank_sums,
)
from qem_bench.stats.ood import (
    method_s0_relative_degradation_rejected,
    raw_normalized_ood_degradation,
)
from qem_bench.stats.power import (
    critical_difference,
    critical_difference_resolution,
    wilcoxon_holm_power_analysis,
)

__all__ = [
    "BootstrapInterval",
    "PlannedComparisonFamily",
    "bias_by_stratum",
    "circuit_blocked_bootstrap",
    "critical_difference",
    "critical_difference_resolution",
    "evaluate_incremental_value",
    "friedman_rank_test",
    "holm_adjust",
    "mean_ranks",
    "macro_mean_iqr",
    "method_s0_relative_degradation_rejected",
    "paired_wilcoxon_holm",
    "raw_normalized_ood_degradation",
    "wilcoxon_holm_power_analysis",
    "wilcoxon_rank_sums",
]
