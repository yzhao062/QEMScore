"""Unit tests for the shift-transfer tool.

Governing rule: docs/frozen-rules/2026-10-06-shift-transfer.md.
Tests the S4 and S2 specifications against the campaign's S0 specification,
the source-identity and target-structure checks on synthetic rows, and the
N3 coupling-noise table for target circuits.
"""

from __future__ import annotations

import numpy as np
import pytest

from qemscore.campaign.design import campaign_split_spec
from qemscore.datasets.splits import resolve_split_spec
from tools import descriptor_ladder as ladder
from tools.shift_transfer import (
    SHIFT_CODE,
    shift_split_spec,
    source_identity,
    target_noise_table,
    target_structure,
)


@pytest.mark.parametrize("shift,axis,source,target", [
    ("S4", "family_native_depth", [3], [6]),
    ("S2", "noise_strength", ["L1", "L3"], ["L4"]),
])
def test_shift_spec_moves_one_axis(shift, axis, source, target):
    spec = shift_split_spec(shift).to_dict()
    base = campaign_split_spec("shipped", 640).to_dict()
    assert spec["split_id"] == shift
    assert spec["source_domain"] == {axis: source}
    assert spec["target_domain"] == {axis: target}
    assert axis not in spec["fixed_axes"]
    assert {k: v for k, v in base["fixed_axes"].items() if k != axis} == spec["fixed_axes"]
    for field in ("n_qubits", "role_counts", "family_parameters", "budget_tier",
                  "partition_id", "replicates"):
        assert spec[field] == base[field]
    resolve_split_spec(shift_split_spec(shift))  # passes the split contract


def _row(family, split, instance, severity="L1", observable="z_mid", **extra):
    row = {"family": family, "split": split, "domain": "source" if split != "test" else "target",
           "instance": instance, "severity": severity, "observable": observable,
           "shots": 2048, "n_qubits": 10, "circuit_id": f"c-{family}-{split}-{instance}",
           "circuit_hash": "h", "circuit_sidecar": "s", "steps": 3, "dt": 0.2,
           "j": 0.5, "h": 0.7, "jx": 0.0, "jy": 0.0, "jz": 0.0,
           "two_qubit_gates": 54, "transpiled_depth": 40, "obs_locality": 1,
           "ideal_expectation": 0.1, "noisy_expectation": 0.08, "noisy_stderr": 0.02,
           "counts_hash": "k", "item_id": f"i-{family}-{split}-{instance}-{severity}-{observable}"}
    row.update(extra)
    return row


def test_source_identity_detects_a_changed_field():
    s0 = {"train": [_row("tfi", "train", 0), _row("tfi", "train", 1)],
          "validation": [_row("tfi", "validation", 0)]}
    same = [dict(r) for r in s0["train"] + s0["validation"]]
    assert source_identity(same, s0)["identical"] is True
    changed = [dict(r) for r in same]
    changed[1]["noisy_expectation"] = 0.081
    report = source_identity(changed, s0)
    assert report["identical"] is False
    assert report["train"]["mismatches_by_field"] == {"noisy_expectation": 1}
    missing = same[1:]
    assert source_identity(missing, s0)["identical"] is False


def test_target_structure():
    rows = [_row(f, "test", i, severity=s, steps=6)
            for f in ("tfi", "heisenberg") for i in range(160) for s in ("L1", "L3")]
    assert target_structure("S4", rows)["passed"] is True
    assert target_structure("S2", rows)["passed"] is False
    l4 = [_row(f, "test", i, severity="L4") for f in ("tfi", "heisenberg") for i in range(160)]
    assert target_structure("S2", l4)["passed"] is True
    assert target_structure("S2", l4[:-2])["passed"] is False


def test_target_noise_table_keeps_s0_draws():
    s0 = {"seed": 101,
          "train": [_row("tfi", "train", 0), _row("heisenberg", "train", 0)],
          "validation": [_row("tfi", "validation", 0)],
          "test": [_row("tfi", "test", 9)]}
    reference = ladder.coupling_noise_table(s0)
    # S2 reuses an S0 test circuit under a new L4 item; S4 brings new circuits.
    reused = _row("tfi", "test", 9, severity="L4", item_id="l4-item")
    new = [_row("tfi", "test", 0, severity="L4", circuit_id="t-0", item_id="ti-0"),
           _row("heisenberg", "test", 1, severity="L4", circuit_id="t-1", item_id="ti-1")]
    table = target_noise_table(s0, [reused, *new], 101, "S2")
    for item, z in reference["z_by_item"].items():
        assert table["z_by_item"][item] == z
    assert table["z_by_item"]["l4-item"] == reference["z_by_item"][s0["test"][0]["item_id"]]
    index = sorted(["t-0", "t-1", s0["test"][0]["circuit_id"]]).index("t-0")
    rng = np.random.default_rng(np.random.SeedSequence(
        [20261002, 101, index], spawn_key=(1000 + SHIFT_CODE["S2"],)))
    assert table["z_by_item"]["ti-0"] == tuple(float(v) for v in rng.standard_normal(2))
    assert len(table["z_by_item"]["ti-1"]) == 3
    other = target_noise_table(s0, new, 101, "S4")
    assert other["z_by_item"]["ti-0"] != table["z_by_item"]["ti-0"]
    clash = [_row("tfi", "test", 0, circuit_id=s0["train"][0]["circuit_id"], item_id="x")]
    with pytest.raises(SystemExit):
        target_noise_table(s0, clash, 101, "S2")


@pytest.mark.parametrize("shift", ["S2", "S4"])
def test_new_target_streams_differ_from_s0_streams(shift):
    for index in range(4):
        new = np.random.default_rng(np.random.SeedSequence(
            [20261002, 101, index], spawn_key=(1000 + SHIFT_CODE[shift],))).standard_normal(3)
        for s0_index in (index, 1000 + SHIFT_CODE[shift]):
            s0 = np.random.default_rng(np.random.SeedSequence(
                [20261002, 101, s0_index])).standard_normal(3)
            assert not np.array_equal(new, s0)


def test_target_structure_checks_circuits_against_s0():
    s0 = {"train": [_row("tfi", "train", 0)], "validation": [_row("tfi", "validation", 0)],
          "test": [_row(f, "test", i) for f in ("tfi", "heisenberg") for i in range(160)]}
    l4 = [_row(f, "test", i, severity="L4") for f in ("tfi", "heisenberg") for i in range(160)]
    assert target_structure("S2", l4, s0)["passed"] is True
    fresh = [dict(r, circuit_id="new-" + r["circuit_id"]) for r in l4]
    assert target_structure("S2", fresh, s0)["passed"] is False
    deep = [dict(r, steps=6, severity=s, circuit_id="d-" + r["circuit_id"])
            for r in l4 for s in ("L1", "L3")]
    assert target_structure("S4", deep, s0)["passed"] is True
    reused = [dict(r, steps=6, severity=s) for r in l4 for s in ("L1", "L3")]
    assert target_structure("S4", reused, s0)["passed"] is False
