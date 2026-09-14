"""Campaign design constants and the analysis that produces the primary endpoint."""

from qemscore.campaign.analysis import (
    assert_campaign_structure,
    build_campaign_tables,
    build_setting_record,
    evaluate_campaign,
)
from qemscore.campaign.design import (
    ARMS,
    BOOTSTRAP_RESAMPLES,
    CONFIDENCE,
    PRIMARY_SIZE,
    REGIMES,
    ROOT_SEED,
    SEEDS,
    SIZES,
    campaign_setting_keys,
    campaign_split_spec,
    expected_circuits,
    is_frozen_setting,
    role_counts,
    setting_key,
)

__all__ = [
    "ARMS",
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "PRIMARY_SIZE",
    "REGIMES",
    "ROOT_SEED",
    "SEEDS",
    "SIZES",
    "assert_campaign_structure",
    "build_campaign_tables",
    "build_setting_record",
    "campaign_setting_keys",
    "campaign_split_spec",
    "evaluate_campaign",
    "expected_circuits",
    "is_frozen_setting",
    "role_counts",
    "setting_key",
]
