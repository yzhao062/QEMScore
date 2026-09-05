"""The frozen campaign design, written as code rather than as a table.

Plan review round 1 asked for an executable freeze. A table in a planning
document cannot be compared against a realized dataset, so the numbers that
decide what the campaign measures live here and every consumer reads them from
this module: the split specification the generator receives, the per-family
circuit counts the structural assertions require, and the setting keys the
analysis expects to find. A setting that is absent is a failed setting, which is
why the expected keys are enumerated here instead of discovered on disk.
"""

from __future__ import annotations

from collections.abc import Mapping

from qem_bench.datasets.splits import SplitSpec

# Two evolution-step regimes. The families use different shipped values because
# their rotation-angle conventions differ; the large endpoint is common.
REGIMES: dict[str, dict[str, float]] = {
    "shipped": {"tfi": 0.2, "heisenberg": 0.15},
    "large": {"tfi": 0.6, "heisenberg": 0.6},
}
BASELINE_REGIME = "shipped"
CONTRAST_REGIME = "large"

SEEDS: tuple[int, ...] = (101, 211, 307)
# Training circuits per family. The larger size carries the primary hypothesis;
# the smaller one is a secondary analysis and shares its training prefix.
SIZES: tuple[int, ...] = (160, 640)
PRIMARY_SIZE = 640

VALIDATION_CIRCUITS_PER_FAMILY = 320
TEST_CIRCUITS_PER_FAMILY = 160
FAMILIES: tuple[str, ...] = ("tfi", "heisenberg")

N_QUBITS = 10
FAMILY_NATIVE_STEPS = 3
NOISE_FAMILY = "depolarizing_readout"
SEVERITIES: tuple[str, ...] = ("L1", "L3")
OBSERVABLES: tuple[str, ...] = ("z_mid", "zz_mid")
SHOTS = 2048

ROOT_SEED = 20260904
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95
RATIO_MARGIN = 1.05
# The gate's two-family requirement is met inside each setting, so a seed that
# succeeds in one family only is not a replication.
REQUIRED_FAMILIES = 2

TRAINING_SHUFFLE_SEED = 1234
FIXED_MODEL_SHUFFLE_SEEDS: tuple[int, ...] = tuple(range(20))

# The primary arm is affine; the capacity-matched arm answers the objection that
# an affine control cannot isolate measurement benefit for a nonlinear model.
ARMS: dict[str, dict[str, str]] = {
    "primary": {"full": "ridge", "control": "feat-only"},
    "capacity_matched": {"full": "liao", "control": "liao-feat-only"},
}
PRIMARY_ARM = "primary"
DROPPED_FEATURE = "noisy_expectation"
TRAINING_SHUFFLE_ARMS: dict[str, str] = {
    "ridge": "ridge-training-shuffle",
    "liao": "liao-training-shuffle",
}


def setting_key(regime: str, seed: int, size: int) -> str:
    """Name a setting. Used for files and for record keys.

    A rehearsal runs on a separate seed and a smaller shape, so the key admits
    both; whether a setting belongs to the frozen campaign is a separate
    question, and `is_frozen_setting` is the one that answers it.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    return f"{regime}-s{seed}-n{size}"


def is_frozen_setting(seed: int, size: int) -> bool:
    """Whether this seed and size belong to the campaign the plan froze."""
    return seed in SEEDS and size in SIZES


def campaign_setting_keys() -> tuple[str, ...]:
    """Every setting the campaign must produce, in a deterministic order."""
    return tuple(
        setting_key(regime, seed, size)
        for regime in REGIMES
        for seed in SEEDS
        for size in SIZES
    )


def expected_circuits(size: int) -> dict[str, int]:
    """Physical circuits per family and role at one training size."""
    if size not in SIZES:
        raise ValueError(f"size {size!r} is not one of the campaign sizes")
    return {
        "train": int(size),
        "validation": VALIDATION_CIRCUITS_PER_FAMILY,
        "test": TEST_CIRCUITS_PER_FAMILY,
    }


def role_counts(size: int) -> dict[str, int]:
    """CLI role counts, which are totals across families rather than per family.

    Severity and observable multiplicity do not divide the role count again, so
    each entry is the per-family count times the number of families.
    """
    return {
        role: value * len(FAMILIES)
        for role, value in expected_circuits(size).items()
    }


def campaign_split_spec(
    regime: str, size: int, *, counts: Mapping[str, int] | None = None,
) -> SplitSpec:
    """The S0 split specification for one regime and training size.

    The generator seed is not part of the specification; it is passed separately
    as the master seed, which is what makes two sizes at one seed share their
    parameter and sampler streams. ``counts`` overrides the per-family circuit
    counts, which only a rehearsal does; every axis except the role counts stays
    at its frozen value, so a rehearsal exercises the same shape at less cost.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    dt = REGIMES[regime]
    per_family = expected_circuits(size) if counts is None else {
        role: int(counts[role]) for role in ("train", "validation", "test")
    }
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": [NOISE_FAMILY],
            "noise_strength": list(SEVERITIES),
            "circuit_family": list(FAMILIES),
            "family_native_depth": [FAMILY_NATIVE_STEPS],
            "observable_class": list(OBSERVABLES),
            "shots": [SHOTS],
        },
        n_qubits=[N_QUBITS],
        role_counts={role: value * len(FAMILIES)
                     for role, value in per_family.items()},
        family_parameters={family: {"dt": dt[family]} for family in FAMILIES},
        budget_tier="H",
    )


def declared_design() -> dict[str, object]:
    """The frozen values, for recording beside the numbers they produced."""
    return {
        "regimes": {name: dict(values) for name, values in REGIMES.items()},
        "baseline_regime": BASELINE_REGIME,
        "contrast_regime": CONTRAST_REGIME,
        "seeds": list(SEEDS),
        "sizes": list(SIZES),
        "primary_size": PRIMARY_SIZE,
        "families": list(FAMILIES),
        "required_families": REQUIRED_FAMILIES,
        "circuits_per_family": {
            str(size): expected_circuits(size) for size in SIZES
        },
        "n_qubits": N_QUBITS,
        "family_native_steps": FAMILY_NATIVE_STEPS,
        "noise_family": NOISE_FAMILY,
        "severities": list(SEVERITIES),
        "observables": list(OBSERVABLES),
        "shots": SHOTS,
        "root_seed": ROOT_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "confidence": CONFIDENCE,
        "ratio_margin": RATIO_MARGIN,
        "arms": {name: dict(value) for name, value in ARMS.items()},
        "primary_arm": PRIMARY_ARM,
        "dropped_feature": DROPPED_FEATURE,
        "training_shuffle_seed": TRAINING_SHUFFLE_SEED,
        "fixed_model_shuffle_seeds": list(FIXED_MODEL_SHUFFLE_SEEDS),
    }


def resolve_regime(dt_by_family: Mapping[str, float]) -> str | None:
    """Name the regime a dataset's family parameters belong to, if any."""
    for name, values in REGIMES.items():
        if all(
            family in dt_by_family and float(dt_by_family[family]) == value
            for family, value in values.items()
        ):
            return name
    return None
