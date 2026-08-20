"""Reproducibility metadata for the frozen CI environment."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

CI_LOCK_SHA256 = "20ce2461bc2e6f70e14eeb0feaf3fe7e3cb9e9af75031da04671eab99c2bfd70"
CI_LOCK_SHA256_ENV = "QEM_BENCH_CI_LOCK_SHA256"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def environment_contract(observed_lock_sha256: str | None = None) -> dict[str, object]:
    observed = observed_lock_sha256
    if observed is None:
        observed = os.environ.get(CI_LOCK_SHA256_ENV)
    if observed is not None:
        observed = observed.strip().lower()
        if _SHA256.fullmatch(observed) is None:
            raise ValueError(f"invalid observed CI lock SHA-256: {observed!r}")
    return {
        "expected_ci_lock_sha256": CI_LOCK_SHA256,
        "observed_ci_lock_sha256": observed,
        "lock_verified": observed == CI_LOCK_SHA256,
    }


def validate_environment_contract(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("environment contract must be an object")
    observed = value.get("observed_ci_lock_sha256")
    if observed is not None and (
        not isinstance(observed, str) or _SHA256.fullmatch(observed) is None
    ):
        raise ValueError("invalid observed CI lock SHA-256")
    expected = {
        "expected_ci_lock_sha256": CI_LOCK_SHA256,
        "observed_ci_lock_sha256": observed,
        "lock_verified": observed == CI_LOCK_SHA256,
    }
    if dict(value) != expected:
        raise ValueError("invalid environment contract")
    return expected


__all__ = [
    "CI_LOCK_SHA256",
    "CI_LOCK_SHA256_ENV",
    "environment_contract",
    "validate_environment_contract",
]
