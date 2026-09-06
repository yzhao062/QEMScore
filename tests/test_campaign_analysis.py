"""Campaign entry point: pre-fit structure, frozen records, and the endpoint.

The expensive half is exercised once, on a deliberately small generated split,
because what is under test is the wiring rather than the estimators: that
selection sees source validation only, that a test label cannot reach it, and
that the record carries everything the analysis needs. Everything else runs on
prescribed predictions, so a test states the outcome it expects instead of
deriving one from a fit.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from qem_bench.campaign.analysis import (
    RECORD_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    TEST_ROW_FIELDS,
    assert_campaign_structure,
    build_campaign_tables,
    build_setting_record,
    evaluate_campaign,
    evaluate_setting_share,
    regenerate_setting_diagnostics,
)
from qem_bench.campaign.design import (
    ARMS,
    campaign_setting_keys,
    campaign_split_spec,
    expected_circuits,
    primary_share_keys,
    role_counts,
    setting_key,
)
from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import SplitSpec
from qem_bench.stats.improvement_share import draw_matrix
from qem_bench.validation import validate_split_artifact

SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
FAMILIES = ("heisenberg", "tfi")
REHEARSAL = {"train": 6, "validation": 4, "test": 4}


# --------------------------------------------------------------------------
# Structural rows, built to a stated shape
# --------------------------------------------------------------------------


def _rows(regime, seed, size, *, counts=None, circuit=None):
    """Build one setting's rows with the identities the assertions read.

    ``circuit`` names the physical circuit of an (family, role, instance), so a
    test can make two sizes share a training prefix or break it.
    """
    counts = counts or expected_circuits(size)
    if circuit is None:
        def circuit(family, role, instance):
            role_tag = "" if role == "train" else role
            return f"{regime}-{seed}-{family}-{role_tag}-c{instance}"
    rows = []
    for family in FAMILIES:
        for role, total in counts.items():
            for instance in range(total):
                for severity in SEVERITIES:
                    for observable in OBSERVABLES:
                        rows.append({
                            "item_id": f"{regime}{seed}{size}{family}{role}"
                                       f"{instance}{severity}{observable}",
                            "dataset_schema_version": "split-v2",
                            "split_id": "S0",
                            "split": role,
                            "family": family,
                            "stratum": "continuous_regression",
                            "circuit_id": circuit(family, role, instance),
                            "instance": instance,
                            "noise_family": "depolarizing_readout",
                            "severity": severity,
                            "observable": observable,
                            "ideal_expectation": 0.0,
                        })
    return rows


def _setting(regime, seed, size, **kwargs):
    return {
        "regime": regime, "seed": seed, "size": size,
        "items": _rows(regime, seed, size, **kwargs),
    }


def test_the_realized_counts_and_identities_of_the_frozen_design_pass():
    settings = [
        _setting(regime, 101, size)
        for regime in ("shipped", "large")
        for size in (160, 640)
    ]
    report = assert_campaign_structure(settings)

    assert report["design_frozen"] is True
    assert report["settings"]["shipped-s101-n640"]["tfi"]["train"] == {
        "circuits": 640, "items": 2560}
    assert report["settings"]["shipped-s101-n160"]["heisenberg"]["validation"] == {
        "circuits": 320, "items": 1280}
    # Two sizes per regime and two regimes per size, each checked once.
    assert len(report["cross_size_checks"]) == 2
    assert len(report["cross_regime_checks"]) == 2
    assert all(check["training_prefix_nested"] for check in report["cross_size_checks"])
    assert all(check["circuits_disjoint"] for check in report["cross_regime_checks"])
    # Eight of the twelve settings were not supplied, and they are named.
    assert len(report["settings_not_supplied"]) == 8
    assert "large-s211-n160" in report["settings_not_supplied"]
    json.dumps(report, allow_nan=False)


def test_a_short_family_is_refused_before_anything_is_fitted():
    short = dict(expected_circuits(160), train=159)
    with pytest.raises(ValueError, match="realized 159 physical circuits, expected 160"):
        assert_campaign_structure([_setting("shipped", 101, 160, counts=short)])


def test_a_broken_training_prefix_is_refused():
    """N640 must extend N160's training pool rather than resample it."""
    def resampled(family, role, instance):
        tag = "resampled" if role == "train" else role
        return f"shipped-101-{family}-{tag}-c{instance}"

    with pytest.raises(ValueError, match="training instance 0 differs between the sizes"):
        assert_campaign_structure([
            _setting("shipped", 101, 160),
            _setting("shipped", 101, 640, circuit=resampled),
        ])


def test_validation_circuits_that_differ_between_sizes_are_refused():
    def moved(family, role, instance):
        suffix = "-moved" if role == "validation" else ""
        role_tag = "" if role == "train" else role
        return f"shipped-101-{family}-{role_tag}-c{instance}{suffix}"

    with pytest.raises(ValueError, match="validation circuits differ between n160"):
        assert_campaign_structure([
            _setting("shipped", 101, 160),
            _setting("shipped", 101, 640, circuit=moved),
        ])


def test_two_regimes_sharing_a_circuit_are_refused():
    """Changing dt redraws every pool, so a shared circuit means matched draws."""
    def shared(family, role, instance):
        role_tag = "" if role == "train" else role
        return f"shared-{family}-{role_tag}-c{instance}"

    with pytest.raises(ValueError, match="have to be independent samples"):
        assert_campaign_structure([
            _setting("shipped", 101, 160, circuit=shared),
            _setting("large", 101, 160, circuit=shared),
        ])


def test_two_regimes_sharing_a_circuit_across_different_roles_are_refused():
    """A role-by-role comparison would pass this and still promise disjointness."""
    def crossed(family, role, instance):
        if role == "test":
            return f"shipped-101-{family}--c{instance}"
        return f"large-101-{family}-{role}-c{instance}"

    with pytest.raises(ValueError, match="in any family or role"):
        assert_campaign_structure([
            _setting("shipped", 101, 160),
            _setting("large", 101, 160, circuit=crossed),
        ])


def test_a_circuit_shared_between_roles_is_refused():
    def leaking(family, role, instance):
        return f"shipped-101-{family}-c{instance}"

    with pytest.raises(ValueError, match="span more than one family or role"):
        assert_campaign_structure([_setting("shipped", 101, 160, circuit=leaking)])


def test_gapped_instance_numbering_is_refused():
    rows = _rows("shipped", 101, 160)
    for row in rows:
        if row["family"] == "tfi" and row["split"] == "test" and row["instance"] == 0:
            row["instance"] = 160
    with pytest.raises(ValueError, match="instance numbering is not 0 through 159"):
        assert_campaign_structure(
            [{"regime": "shipped", "seed": 101, "size": 160, "items": rows}])


def test_a_rehearsal_shape_needs_an_explicit_override():
    settings = [_setting("shipped", 101, 160, counts=REHEARSAL)]
    with pytest.raises(ValueError, match="realized 6 physical circuits, expected 160"):
        assert_campaign_structure(settings)

    report = assert_campaign_structure(settings, expected_by_size={160: REHEARSAL})
    assert report["design_frozen"] is False
    assert report["asserted_circuits"]["shipped-s101-n160"] == REHEARSAL


# --------------------------------------------------------------------------
# The endpoint, regenerated from records alone
# --------------------------------------------------------------------------


DIAGNOSTIC_METHODS = ("ridge", "feat-only", "noisy-only", "liao", "liao-feat-only",
                      "ridge-training-shuffle", "liao-training-shuffle")
SHUFFLE_SEEDS = ("0", "1")


def _validation_block(label):
    """A minimal but genuine source-validation block, so diagnostics regenerate.

    The gate needs two continuous families and at least two circuits per pool, so
    the block carries them; the numbers matter only in that they are consistent.
    """
    rows, predictions = [], {method: {} for method in DIAGNOSTIC_METHODS}
    for family in FAMILIES:
        for instance in range(4):
            for severity in SEVERITIES:
                for observable in OBSERVABLES:
                    item_id = f"{label}-v-{family}-{instance}-{severity}-{observable}"
                    rows.append({
                        "item_id": item_id,
                        "dataset_schema_version": "split-v2",
                        "split_id": "S0",
                        "split_axis": "circuit_instance",
                        "partition_id": f"{label}-p",
                        "split": "validation",
                        "domain": "source",
                        "family": family,
                        "stratum": "continuous_regression",
                        "circuit_id": f"{label}-v-{family}-c{instance}",
                        "circuit_pool_id": f"{label}-pool-{family}",
                        "instance": instance,
                        "noise_family": "depolarizing_readout",
                        "severity": severity,
                        "observable": observable,
                        "shots": 2048,
                        "noisy_expectation": 0.30,
                        "ideal_expectation": 0.0,
                    })
                    for index, method in enumerate(DIAGNOSTIC_METHODS):
                        predictions[method][item_id] = 0.05 + 0.01 * index
    order = [str(row["item_id"]) for row in rows]
    shuffle = {
        "seeds": [int(value) for value in SHUFFLE_SEEDS],
        "permutation_strata": ["family", "noise_family", "severity", "observable"],
        "scope": "fixture",
        "item_order": order,
        "predictions": {
            method: {seed: [0.09 + 0.01 * offset] * len(order)
                     for offset, seed in enumerate(SHUFFLE_SEEDS)}
            for method in ("ridge", "liao")
        },
    }
    return rows, predictions, shuffle


def _record(regime, seed, size, errors, *, frozen=True, label=None):
    """A record whose untouched-test absolute errors are exactly prescribed."""
    label = label or f"{regime}{seed}{size}"
    items, predictions = [], {method: {} for method in errors}
    for family in FAMILIES:
        for instance in range(8):
            for severity in SEVERITIES:
                for observable in OBSERVABLES:
                    item_id = f"{label}-{family}-{instance}-{severity}-{observable}"
                    items.append({
                        "item_id": item_id,
                        "dataset_schema_version": "split-v2",
                        "split": "test",
                        "family": family,
                        "stratum": "continuous_regression",
                        "circuit_id": f"{label}-{family}-c{instance}",
                        "instance": instance,
                        "noise_family": "depolarizing_readout",
                        "severity": severity,
                        "observable": observable,
                        # The unmitigated reference the improvement share
                        # scores beside the ladder. It is a model input rather
                        # than a label, and it is worse than every arm here, so
                        # a reported reference error names this column.
                        "noisy_expectation": 0.40,
                        "ideal_expectation": 0.0,
                    })
                    for method, error in errors.items():
                        predictions[method][item_id] = error
    validation_items, validation_predictions, shuffle = _validation_block(label)
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "setting": setting_key(regime, seed, size),
        "regime": regime, "seed": seed, "size": size,
        "data_path": f"/campaign/{label}",
        "dataset_hash": f"hash-{label}",
        "split_spec_hash": f"spec-{label}",
        "master_seed": seed,
        "code_revision": "abc123",
        "design_frozen": frozen,
        "n_resamples": 40,
        "confidence": 0.95,
        "gate_seed": 20260904,
        "test_items": items,
        "test_predictions": predictions,
        "validation_items": validation_items,
        "validation_predictions": validation_predictions,
        "fixed_model_shuffle": shuffle,
    }


def _campaign(seed_errors, *, size=640, frozen=True):
    """One record pair per seed, at the given size, for both arms."""
    records = []
    for seed, (shipped, large) in seed_errors.items():
        records.append(_record("shipped", seed, size, shipped, frozen=frozen))
        records.append(_record("large", seed, size, large, frozen=frozen))
    return records


def _errors(full, control):
    return {
        ARMS["primary"]["full"]: full,
        ARMS["primary"]["control"]: control,
        ARMS["capacity_matched"]["full"]: full,
        ARMS["capacity_matched"]["control"]: control,
    }


def _distinct_errors(ridge, feat_only, liao, liao_feat_only):
    """Four different errors, so a MAE names which predictions produced it."""
    return {
        ARMS["primary"]["full"]: ridge,
        ARMS["primary"]["control"]: feat_only,
        ARMS["capacity_matched"]["full"]: liao,
        ARMS["capacity_matched"]["control"]: liao_feat_only,
    }


RISING = (_errors(0.10, 0.10), _errors(0.08, 0.10))
FLAT = (_errors(0.10, 0.10), _errors(0.10, 0.10))
FALLING = (_errors(0.08, 0.10), _errors(0.10, 0.10))


def _audit(record, **overrides):
    return {
        "dataset_hash": record["dataset_hash"],
        "passed": True,
        "zne_replay": {
            "checked": True,
            "roster_artifact_id": _binding(record)["artifact_id"],
        },
    } | overrides


def _evaluate(records, **kwargs):
    """Evaluate at the frozen resample count, which is what a record can claim."""
    kwargs.setdefault(
        "audits", {record["setting"]: _audit(record) for record in records})
    return evaluate_campaign(records, **kwargs)


def test_the_report_regenerates_the_gate_and_both_shuffle_diagnostics():
    """Round 2 asked for every prediction map, so the diagnostics can replay."""
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))
    diagnostics = report["diagnostics"]["shipped-s101-n640"]

    assert set(diagnostics["gates"]) == {"ridge", "liao"}
    for gate in diagnostics["gates"].values():
        assert gate["status"] in {"passed", "failed"}, gate["reason"]
    assert set(diagnostics["training_shuffle_differences"]) == {"ridge", "liao"}
    assert set(diagnostics["fixed_model_shuffle_mae"]["ridge"]) == set(FAMILIES)
    assert len(diagnostics["fixed_model_shuffle_mae"]["liao"]["tfi"]["per_seed"]) == 2

    tables = build_campaign_tables(report)
    assert len(tables["gates"]) == 12
    assert len(tables["training_shuffle"]) == 24
    assert len(tables["fixed_model_shuffle"]) == 24
    assert {row["setting"] for row in tables["gates"]} == set(
        report["diagnostics"])
    json.dumps(tables, allow_nan=False)


def test_a_record_without_the_permutation_maps_cannot_be_evaluated():
    """A summary alone cannot be re-derived, so the maps are required."""
    records = _campaign({101: RISING})
    del records[0]["fixed_model_shuffle"]["predictions"]
    with pytest.raises(KeyError, match="predictions"):
        _evaluate(records)


def test_three_rising_seeds_promote_and_name_every_dataset_hash():
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))

    primary = report["primary"]
    assert primary["successes"] == 3
    assert primary["successful_seeds"] == ["101", "211", "307"]
    assert primary["promoted"] is True
    assert primary["publication_path"] == "promote_three_of_three"
    assert primary["full_method"] == "ridge"
    assert primary["control_method"] == "feat-only"
    # Both endpoint gains are 0.0 and 0.2, so the rise is real but the shipped
    # endpoint carries no benefit of its own.
    assert primary["positive_endpoint_benefit_supported"]["101"] == {
        "heisenberg": True, "tfi": True}
    assert report["code_revisions"] == ["abc123"]
    assert report["dataset_hashes"]["large-s307-n640"] == "hash-large307640"
    json.dumps(report, allow_nan=False)


def test_two_rising_seeds_and_one_reversal_promote_with_the_dissent_named():
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: FALLING}))

    primary = report["primary"]
    assert primary["successful_seeds"] == ["101", "211"]
    assert primary["reversed_seeds"] == ["307"]
    assert primary["promoted"] is True
    assert primary["publication_path"] == "promote_two_of_three_with_dissent"


def test_one_rising_seed_does_not_promote():
    report = _evaluate(_campaign({101: RISING, 211: FLAT, 307: FLAT}))

    primary = report["primary"]
    assert primary["successes"] == 1
    assert primary["promoted"] is False
    assert primary["publication_path"] == "no_headline_one_replication"


def test_three_falling_seeds_report_evidence_against_the_declared_direction():
    report = _evaluate(_campaign({101: FALLING, 211: FALLING, 307: FALLING}))

    primary = report["primary"]
    assert primary["successes"] == 0
    assert primary["reversed_seeds"] == ["101", "211", "307"]
    assert primary["publication_path"] == "evidence_against_predeclared_direction"


def test_two_reversals_carry_the_reversal_verdict_whatever_the_third_seed_says():
    """Two jointly reversed seeds and no success, mirroring the promotion rule.

    Requiring every evaluated seed to reverse would make an unavailable third
    seed strengthen the verdict that an inconclusive third seed weakens.
    """
    inconclusive = _evaluate(_campaign({101: FALLING, 211: FALLING, 307: FLAT}))
    assert inconclusive["primary"]["reversed_seeds"] == ["101", "211"]
    assert inconclusive["primary"]["evaluated_seeds"] == ["101", "211", "307"]
    assert inconclusive["primary"]["publication_path"] == (
        "evidence_against_predeclared_direction")

    absent = _evaluate([
        value for value in _campaign({101: FALLING, 211: FALLING, 307: FLAT})
        if value["seed"] != 307
    ])
    assert absent["primary"]["publication_path"] == inconclusive["primary"][
        "publication_path"]


def test_flat_seeds_are_inconclusive_rather_than_evidence_of_absence():
    report = _evaluate(_campaign({101: FLAT, 211: FLAT, 307: FLAT}))

    primary = report["primary"]
    assert primary["successes"] == 0
    assert primary["reversed_seeds"] == []
    assert primary["publication_path"] == "no_headline_inconclusive"


def test_a_seed_that_succeeds_in_one_family_only_is_not_a_replication():
    """The gate's two-family requirement is met inside a seed, not across seeds."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    for record in records:
        if record["regime"] == "large" and record["seed"] == 307:
            for item in record["test_items"]:
                if item["family"] == "tfi":
                    record["test_predictions"]["ridge"][item["item_id"]] = 0.10
    report = _evaluate(records)

    primary = report["primary"]
    assert primary["successful_seeds"] == ["101", "211"]
    assert report["contrasts"]["primary"]["640"]["307"]["families"]["tfi"][
        "contrast"]["excludes_zero_above"] is False


def test_a_family_one_regime_never_produced_survives_into_the_tables():
    """The estimator reports it; the tables have to print it rather than crash.

    Round 3's repair gave an unavailable regime a null endpoint summary, and the
    table builder dereferenced it. A test that stops at the estimator cannot see
    that, so this one goes through `evaluate_campaign` to the printed rows.
    """
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    for record in records:
        if record["regime"] == "large" and record["seed"] == 307:
            keep = [row for row in record["test_items"] if row["family"] != "tfi"]
            dropped = {row["item_id"] for row in record["test_items"]} - {
                row["item_id"] for row in keep}
            record["test_items"] = keep
            for predictions in record["test_predictions"].values():
                for item_id in dropped:
                    predictions.pop(item_id)

    report = _evaluate(records)
    contrast = report["contrasts"]["primary"]["640"]["307"]
    assert contrast["families"]["tfi"]["status"] == "not_evaluable"
    assert contrast["families"]["tfi"]["reason"] == "family_missing_from_a_regime"
    assert contrast["status"] == "not_evaluable"
    # A seed missing one family is not a replication.
    assert report["primary"]["successful_seeds"] == ["101", "211"]
    assert report["primary"]["failed_seeds"] == ["307"]

    tables = build_campaign_tables(report)
    rows = [row for row in tables["endpoints"]
            if row["seed"] == 307 and row["family"] == "tfi"
            and row["arm"] == "primary"]
    assert {row["regime"]: row["available"] for row in rows} == {
        "shipped": True, "large": False}
    unavailable = next(row for row in rows if row["regime"] == "large")
    assert unavailable["full_mae"] is None
    assert unavailable["gain_lower"] is None
    assert unavailable["reason"] == "family_missing_from_a_regime"
    # The regime that did produce the family keeps its numbers.
    available = next(row for row in rows if row["regime"] == "shipped")
    assert available["control_mae"] == pytest.approx(0.10)
    json.dumps(tables, allow_nan=False)


def test_a_missing_record_is_a_failed_setting_rather_than_a_dropped_one():
    records = [
        value for value in _campaign({101: RISING, 211: RISING, 307: RISING})
        if not (value["regime"] == "large" and value["seed"] == 307)
    ]
    report = _evaluate(records)

    assert "large-s307-n640" in report["failed_settings"]
    assert report["contrasts"]["primary"]["640"]["307"] == {
        "status": "failed_setting",
        "reason": "missing_record",
        "missing": ["large-s307-n640"],
    }
    primary = report["primary"]
    assert primary["failed_seeds"] == ["307"]
    assert primary["successes"] == 2
    assert primary["publication_path"] == "promote_two_of_three_with_dissent"


def test_every_unsupplied_setting_is_listed_as_failed():
    report = _evaluate(_campaign({101: RISING}))

    assert len(report["failed_settings"]) == len(campaign_setting_keys()) - 2
    assert sorted(report["failed_settings"])[:2] == [
        "large-s101-n160", "large-s211-n160"]
    assert report["primary"]["failed_seeds"] == ["211", "307"]


def test_a_rehearsal_record_can_never_reach_a_publication_path():
    report = _evaluate(
        _campaign({101: RISING, 211: RISING, 307: RISING}, frozen=False))

    assert report["design_frozen"] is False
    assert report["primary"]["successes"] == 3
    assert report["primary"]["promoted"] is False
    assert report["primary"]["publication_path"] == "rehearsal_not_publishable"
    paths = {row["publication_path"] for row in build_campaign_tables(report)[
        "replication"]}
    assert paths == {"rehearsal_not_publishable"}


def test_a_cheaper_resample_count_self_marks_as_a_rehearsal():
    """The plan froze 10,000 draws, so 200 of them cannot carry the headline."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    report = evaluate_campaign(records, n_resamples=200)

    assert report["design_frozen"] is False
    assert report["primary"]["successes"] == 3
    assert report["primary"]["publication_path"] == "rehearsal_not_publishable"


def _binding(record, **overrides):
    return {
        "setting": record["setting"],
        "artifact_id": f"run-{record['setting']}",
        "dataset_hash": record["dataset_hash"],
        "dataset_manifest_sha256": f"sha256:{record['setting']}",
        "methods": ["raw", "ridge", "zne", "liao", "feat-only", "noisy-only",
                    "shrinkage", "shuf-noisy"],
    } | overrides


def test_the_descriptive_roster_is_bound_to_the_dataset_it_scored():
    """The endpoint arms are fitted here; ZNE comes from the shipped roster."""
    records = _campaign({101: RISING})
    report = _evaluate(
        records, rosters={r["setting"]: _binding(r) for r in records})

    assert sorted(report["roster_artifacts"]) == [
        "large-s101-n640", "shipped-s101-n640"]
    assert "zne" in report["roster_artifacts"]["shipped-s101-n640"]["methods"]
    assert report["settings_without_a_roster"] == []


def test_a_setting_without_a_roster_is_named_rather_than_ignored():
    records = _campaign({101: RISING})
    report = _evaluate(
        records, rosters={records[0]["setting"]: _binding(records[0])})

    assert report["settings_without_a_roster"] == ["large-s101-n640"]
    assert sorted(report["roster_artifacts"]) == ["shipped-s101-n640"]


def test_a_roster_built_from_another_dataset_is_refused():
    records = _campaign({101: RISING})
    rosters = {r["setting"]: _binding(r) for r in records}
    rosters["large-s101-n640"]["dataset_hash"] = "hash-someone-elses-dataset"

    with pytest.raises(ValueError, match="the roster reports dataset"):
        _evaluate(records, rosters=rosters)


def test_the_two_seeds_draw_from_independent_recorded_streams():
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))

    mappings = [
        report["contrasts"]["primary"]["640"][seed]["stream_mapping"]
        for seed in ("101", "211", "307")
    ]
    seeds = [value for mapping in mappings for value in mapping.values()]
    assert len(set(seeds)) == len(seeds)


def test_a_record_whose_fields_disagree_with_its_key_is_refused():
    records = _campaign({101: RISING})
    records[0]["setting"] = setting_key("large", 101, 640)
    with pytest.raises(ValueError, match="disagrees with its fields"):
        _evaluate(records)


def test_records_from_another_schema_are_refused():
    records = _campaign({101: RISING})
    records[0]["schema_version"] = "qem-bench-campaign-record-v0"
    with pytest.raises(ValueError, match="schema_version"):
        _evaluate(records)


def test_the_same_setting_twice_is_refused():
    records = _campaign({101: RISING})
    with pytest.raises(ValueError, match="appears twice"):
        _evaluate(records + [copy.deepcopy(records[0])])


def test_the_tables_carry_both_endpoints_beside_every_contrast():
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))
    tables = build_campaign_tables(report)

    assert tables["schema_version"] == REPORT_SCHEMA_VERSION
    primary_rows = [row for row in tables["endpoints"]
                    if row["arm"] == "primary" and row["seed"] == 101
                    and row["family"] == "tfi"]
    assert {row["regime"] for row in primary_rows} == {"shipped", "large"}
    large = next(row for row in primary_rows if row["regime"] == "large")
    assert large["full_mae"] == pytest.approx(0.08)
    assert large["control_mae"] == pytest.approx(0.10)
    assert large["absolute_reduction"] == pytest.approx(0.02)
    assert large["gain"] == pytest.approx(0.2)
    contrast = next(row for row in tables["contrasts"]
                    if row["arm"] == "primary" and row["seed"] == 101
                    and row["family"] == "tfi" and row["size"] == 640)
    assert contrast["estimate"] == pytest.approx(0.2)
    assert contrast["excludes_zero_above"] is True
    json.dumps(tables, allow_nan=False)


# --------------------------------------------------------------------------
# The record itself, on a real fit
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rehearsal_setting(tmp_path_factory):
    """A deliberately small S0 split, generated the way the campaign generates."""
    root = tmp_path_factory.mktemp("campaign")
    spec = SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": ["depolarizing_readout"],
            "noise_strength": list(SEVERITIES),
            "circuit_family": ["tfi", "heisenberg"],
            "family_native_depth": [1],
            "observable_class": list(OBSERVABLES),
            "shots": [64],
        },
        n_qubits=[3],
        role_counts={role: value * 2 for role, value in REHEARSAL.items()},
        family_parameters={"tfi": {"dt": 0.2}, "heisenberg": {"dt": 0.15}},
        budget_tier="H",
    )
    generate_split(spec, root / "data", master_seed=101)
    rows, manifest = validate_split_artifact(root / "data")
    return rows, manifest


@pytest.fixture(scope="module")
def rehearsal_record(rehearsal_setting):
    rows, manifest = rehearsal_setting
    return build_setting_record(
        rows, manifest, regime="shipped", seed=101, size=160,
        data_path="/campaign/rehearsal", code_revision="rev-1",
        expected=REHEARSAL, n_resamples=40, fixed_model_shuffle_seeds=(0, 1),
    )


def test_the_record_carries_every_arm_and_says_what_it_dropped(rehearsal_record):
    record = rehearsal_record
    methods = set(record["test_predictions"])

    assert methods == {"ridge", "feat-only", "liao", "liao-feat-only"}
    # Validation carries the two scoring arms, both gate controls, and both
    # training-shuffle diagnostics; noisy-only exists for the gate alone.
    assert set(record["validation_predictions"]) == methods | {
        "noisy-only", "ridge-training-shuffle", "liao-training-shuffle"}
    assert record["selected_configs"]["liao-feat-only"]["dropped_features"] == [
        "noisy_expectation"]
    assert record["selected_configs"]["liao"]["dropped_features"] == []
    # The capacity-matched pair differs in the dropped column alone.
    full = record["selected_configs"]["liao"]
    control = record["selected_configs"]["liao-feat-only"]
    assert full["selection_rule"] == control["selection_rule"]
    assert record["selected_configs"]["ridge"]["best_alpha"] > 0
    assert record["design_frozen"] is False
    assert record["asserted_circuits"] == REHEARSAL
    json.dumps(record, allow_nan=False)


def test_the_record_retains_a_prediction_for_every_test_row(rehearsal_record):
    identities = {row["item_id"] for row in rehearsal_record["test_items"]}

    assert len(identities) == len(rehearsal_record["test_items"])
    for method, predictions in rehearsal_record["test_predictions"].items():
        assert set(predictions) == identities, method
    assert {row["split"] for row in rehearsal_record["test_items"]} == {"test"}
    # The retained rows are exactly the declared projection, and the unmitigated
    # reference is one of them. Without this column the improvement share has
    # nothing to score R against and could only report the ladder.
    for row in rehearsal_record["test_items"]:
        assert set(row) == set(TEST_ROW_FIELDS)
        assert isinstance(row["noisy_expectation"], float)


def test_the_record_reports_the_gate_and_both_shuffle_diagnostics(rehearsal_record):
    record = rehearsal_record

    assert set(record["gates"]) == {"ridge", "liao"}
    for name, gate in record["gates"].items():
        assert gate["evaluation_role"] == "source_validation"
        # A gate missing a control reports not_evaluable rather than failing, so
        # an unfitted control would leave an empty gate that still read as one.
        assert gate["status"] in {"passed", "failed"}, (name, gate["reason"])
        for control in gate["controls"]:
            assert control in record["validation_predictions"], control
    differences = record["training_shuffle"]["differences"]
    assert set(differences) == {"ridge", "liao"}
    for by_family in differences.values():
        assert set(by_family) == {"heisenberg", "tfi"}
        for interval in by_family.values():
            assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    shuffle = record["fixed_model_shuffle"]
    assert shuffle["seeds"] == [0, 1]
    assert set(shuffle["mae"]["ridge"]) == {"heisenberg", "tfi"}
    assert len(shuffle["mae"]["liao"]["tfi"]["per_seed"]) == 2


def test_changing_a_test_label_cannot_change_selection_or_predictions(
    rehearsal_setting, rehearsal_record,
):
    """Prediction rows omit the label, so the test roster stays untouched."""
    rows, manifest = rehearsal_setting
    poisoned = copy.deepcopy(rows)
    for row in poisoned:
        if row["split"] == "test":
            row["ideal_expectation"] = 0.0

    rebuilt = build_setting_record(
        poisoned, manifest, regime="shipped", seed=101, size=160,
        data_path="/campaign/rehearsal", code_revision="rev-1",
        expected=REHEARSAL, n_resamples=40, fixed_model_shuffle_seeds=(0, 1),
    )
    assert rebuilt["selected_configs"] == rehearsal_record["selected_configs"]
    assert rebuilt["test_predictions"] == rehearsal_record["test_predictions"]
    assert rebuilt["gates"] == rehearsal_record["gates"]
    # Only the labels the endpoint scores against moved.
    assert rebuilt["test_items"] != rehearsal_record["test_items"]


def test_the_diagnostics_replay_from_the_record_without_a_refit_or_the_dataset(
    rehearsal_record, tmp_path,
):
    """The record has to be sufficient on its own, so this reads only its bytes.

    Round trip through JSON on disk, with no dataset directory in reach and no
    model in memory. Equality with what the fit stored is what shows the retained
    rows and prediction maps carry the diagnostics rather than merely echo them.
    """
    path = tmp_path / "record.json"
    path.write_text(json.dumps(rehearsal_record, allow_nan=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert not Path(loaded["data_path"]).exists()

    replayed = regenerate_setting_diagnostics(loaded)

    assert replayed["gates"] == rehearsal_record["gates"]
    assert replayed["training_shuffle_differences"] == (
        rehearsal_record["training_shuffle"]["differences"])
    assert replayed["fixed_model_shuffle_mae"] == (
        rehearsal_record["fixed_model_shuffle"]["mae"])


def test_a_record_without_a_code_revision_is_refused(rehearsal_setting):
    rows, manifest = rehearsal_setting
    with pytest.raises(ValueError, match="name the code revision"):
        build_setting_record(
            rows, manifest, regime="shipped", seed=101, size=160,
            data_path="/campaign/rehearsal", code_revision="  ",
            expected=REHEARSAL, n_resamples=40, fixed_model_shuffle_seeds=(),
        )


def test_a_record_whose_master_seed_disagrees_with_the_setting_is_refused(
    rehearsal_setting,
):
    rows, manifest = rehearsal_setting
    with pytest.raises(ValueError, match="master seed 101, not 211"):
        build_setting_record(
            rows, manifest, regime="shipped", seed=211, size=160,
            data_path="/campaign/rehearsal", code_revision="rev-1",
            expected=REHEARSAL, n_resamples=40, fixed_model_shuffle_seeds=(),
        )


# --------------------------------------------------------------------------
# The generator behaviour the two-size and two-regime designs rest on
# --------------------------------------------------------------------------


def _small_spec(dt, *, train):
    """The campaign's shape at a size a test can generate in seconds."""
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": ["depolarizing_readout"],
            "noise_strength": list(SEVERITIES),
            "circuit_family": ["tfi", "heisenberg"],
            "family_native_depth": [1],
            "observable_class": list(OBSERVABLES),
            "shots": [64],
        },
        n_qubits=[3],
        role_counts={"train": train * 2, "validation": 4, "test": 4},
        family_parameters={family: {"dt": value} for family, value in dt.items()},
        budget_tier="H",
    )


@pytest.fixture(scope="module")
def generated_sizes_and_regimes(tmp_path_factory):
    """Two sizes at one seed, and two evolution steps at that same seed."""
    root = tmp_path_factory.mktemp("generator")
    shipped = {"tfi": 0.2, "heisenberg": 0.15}
    large = {"tfi": 0.6, "heisenberg": 0.6}
    built = {}
    for name, dt, train in (
        ("shipped-2", shipped, 2), ("shipped-5", shipped, 5), ("large-2", large, 2),
    ):
        generate_split(_small_spec(dt, train=train), root / name, master_seed=101)
        built[name] = validate_split_artifact(root / name)[0]
    return built


def test_the_larger_size_extends_the_smaller_pool_at_one_seed(
    generated_sizes_and_regimes,
):
    """Role counts sit outside the pool descriptor, so the draws recur."""
    built = generated_sizes_and_regimes
    report = assert_campaign_structure(
        [
            {"regime": "shipped", "seed": 101, "size": 2,
             "items": built["shipped-2"]},
            {"regime": "shipped", "seed": 101, "size": 5,
             "items": built["shipped-5"]},
        ],
        expected_by_size={
            2: {"train": 2, "validation": 2, "test": 2},
            5: {"train": 5, "validation": 2, "test": 2},
        },
    )

    assert report["cross_size_checks"] == [{
        "regime": "shipped", "seed": 101, "sizes": [2, 5],
        "training_prefix_nested": True, "validation_and_test_identical": True,
    }]
    assert report["design_frozen"] is False


def test_changing_the_evolution_step_redraws_every_pool(
    generated_sizes_and_regimes,
):
    """The two regimes are independent samples, so no circuit may recur."""
    built = generated_sizes_and_regimes
    counts = {2: {"train": 2, "validation": 2, "test": 2}}
    report = assert_campaign_structure(
        [
            {"regime": "shipped", "seed": 101, "size": 2,
             "items": built["shipped-2"]},
            {"regime": "large", "seed": 101, "size": 2, "items": built["large-2"]},
        ],
        expected_by_size=counts,
    )

    assert report["cross_regime_checks"] == [{
        "seed": 101, "size": 2, "regimes": ["large", "shipped"],
        "circuits_disjoint": True,
    }]
    shipped_ids = {row["circuit_id"] for row in built["shipped-2"]}
    large_ids = {row["circuit_id"] for row in built["large-2"]}
    assert shipped_ids and not shipped_ids & large_ids


# --------------------------------------------------------------------------
# The frozen design, read from code rather than from a table
# --------------------------------------------------------------------------


def test_the_driver_accepts_every_frozen_setting_and_nothing_else():
    import argparse

    from tools.run_campaign_analysis import _parse_setting

    for key in campaign_setting_keys():
        assert setting_key(*_parse_setting(key)) == key
    for unknown in ("shipped-s101-n320", "medium-s101-n160", "shipped-s999-n160"):
        with pytest.raises(argparse.ArgumentTypeError, match="unknown setting"):
            _parse_setting(unknown)


def test_the_frozen_design_resolves_the_counts_the_plan_declares():
    assert len(campaign_setting_keys()) == 12
    assert expected_circuits(640) == {
        "train": 640, "validation": 320, "test": 160}
    assert role_counts(160) == {"train": 320, "validation": 640, "test": 320}
    assert role_counts(640) == {"train": 1280, "validation": 640, "test": 320}


def test_the_two_regimes_differ_only_in_the_evolution_step():
    shipped = campaign_split_spec("shipped", 640).to_dict()
    large = campaign_split_spec("large", 640).to_dict()

    assert shipped["family_parameters"] == {
        "tfi": {"dt": 0.2}, "heisenberg": {"dt": 0.15}}
    assert large["family_parameters"] == {
        "tfi": {"dt": 0.6}, "heisenberg": {"dt": 0.6}}
    assert {key: value for key, value in shipped.items()
            if key != "family_parameters"} == {
        key: value for key, value in large.items() if key != "family_parameters"}


def test_the_two_sizes_differ_only_in_the_training_count():
    small = campaign_split_spec("shipped", 160).to_dict()
    large = campaign_split_spec("shipped", 640).to_dict()

    assert small["role_counts"]["train"] == 320
    assert large["role_counts"]["train"] == 1280
    assert {key: value for key, value in small.items() if key != "role_counts"} == {
        key: value for key, value in large.items() if key != "role_counts"}


def test_an_unaudited_setting_can_never_reach_a_publication_path():
    """The plan requires the re-derivation to pass before a number is printed."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    report = evaluate_campaign(records)

    assert report["design_frozen"] is True
    assert report["primary"]["successes"] == 3
    assert report["primary"]["promoted"] is False
    assert report["primary"]["publication_path"] == "unaudited_not_publishable"
    assert sorted(report["settings_without_an_audit"]) == sorted(
        record["setting"] for record in records)


def test_a_failing_audit_withholds_the_publication_path_and_is_named():
    """A re-derivation that found a mismatch is not a re-derivation that passed."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    audits = {record["setting"]: _audit(record) for record in records}
    audits["large-s211-n640"]["passed"] = False
    report = evaluate_campaign(records, audits=audits)

    assert report["settings_failing_audit"] == ["large-s211-n640"]
    assert report["audit_complete"] is False
    assert report["primary"]["promoted"] is False
    assert report["primary"]["publication_path"] == "unaudited_not_publishable"


def test_an_audit_of_a_different_dataset_is_refused():
    """A re-derivation of another dataset says nothing about this one."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    audits = {record["setting"]: _audit(record) for record in records}
    audits["shipped-s101-n640"]["dataset_hash"] = "sha256:somewhere-else"

    with pytest.raises(ValueError, match="re-derived dataset"):
        evaluate_campaign(records, audits=audits)


def test_an_audit_that_skipped_the_zero_noise_replay_does_not_count_as_passed():
    """`passed` means no mismatch among the checks that ran, and --skip-zne runs none."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    audits = {record["setting"]: _audit(record) for record in records}
    audits["large-s101-n640"]["zne_replay"] = {"checked": False}
    report = evaluate_campaign(
        records,
        rosters={record["setting"]: _binding(record) for record in records},
        audits=audits,
    )

    assert report["settings_without_a_zero_noise_replay"] == ["large-s101-n640"]
    assert report["audited_settings"]["large-s101-n640"]["zne_replayed"] is False
    assert report["primary"]["promoted"] is False
    assert report["primary"]["publication_path"] == "unaudited_not_publishable"


def _with_labels(record, by_observable):
    """Prescribe ideal labels per observable, keeping the prescribed errors.

    A record's predictions equal its absolute errors only while the label is
    zero. Shifting the prediction by the same label holds abs(prediction - label)
    at the value the fixture prescribed.
    """
    for item in record["test_items"]:
        center, half_width = by_observable[item["observable"]]
        label = center + (half_width if item["instance"] % 2 == 0 else -half_width)
        item["ideal_expectation"] = label
        for predictions in record["test_predictions"].values():
            predictions[item["item_id"]] += label
    return record


def test_the_tables_carry_the_ideal_label_spread_beside_every_endpoint():
    """The endpoint is scale invariant, so the label scale travels with it."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    for record in records:
        widths = ((0.5, 2.5) if record["regime"] == "large" else (0.1, 0.5))
        _with_labels(record, {"z_mid": (0.1, widths[0]),
                              "zz_mid": (0.9, widths[1])})

    report = _evaluate(records)
    tables = build_campaign_tables(report)
    rows = {(row["regime"], row["seed"], row["size"], row["family"]): row
            for row in tables["label_spread"]}
    assert len(rows) == len(tables["label_spread"]) == 12

    # Within a cell the circuits sit at the center plus or minus the half width,
    # so both statistics equal that half width and the macro mean over the four
    # cells is 0.3. Pooling the same rows would fold the gap between the two
    # observable centers into the number.
    shipped = rows[("shipped", 101, 640, "tfi")]
    endpoint = report["contrasts"]["primary"]["640"]["101"]["families"]["tfi"][
        "regimes"]["shipped"]
    assert (shipped["n_cells"], shipped["n_items"]) == (
        endpoint["n_cells"], endpoint["n_items"]) == (4, 32)
    assert shipped["macro_mean_absolute_deviation"] == pytest.approx(0.3)
    assert shipped["macro_std"] == pytest.approx(0.3)
    assert shipped["macro_std"] != pytest.approx(0.5385164807134504)
    assert rows[("large", 101, 640, "tfi")][
        "macro_mean_absolute_deviation"] == pytest.approx(1.5)

    # Every available endpoint joins to a spread row on the four columns it
    # already prints, and the endpoint itself is unchanged.
    large = next(row for row in tables["endpoints"]
                 if row["arm"] == "primary" and row["seed"] == 101
                 and row["family"] == "tfi" and row["regime"] == "large")
    assert large["control_mae"] == pytest.approx(0.10)
    assert large["gain"] == pytest.approx(0.2)
    for row in tables["endpoints"]:
        if row["available"]:
            assert (row["regime"], row["seed"], row["size"],
                    row["family"]) in rows
    json.dumps(tables, allow_nan=False)


def test_a_replay_of_a_replaced_roster_does_not_count_as_a_replay():
    """The roster can be regenerated, so agreeing dataset hashes prove nothing.

    `roster` writes a fresh binding for the same dataset, and the audit that ran
    against the previous one still reports `passed` with its own artifact named.
    Without the artifact comparison the stale audit authorizes the roster whose
    zero-noise numbers the paper would print.
    """
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    rosters = {record["setting"]: _binding(record) for record in records}
    audits = {record["setting"]: _audit(record) for record in records}
    rosters["large-s101-n640"]["artifact_id"] = "run-replacement-never-audited"
    report = evaluate_campaign(records, rosters=rosters, audits=audits)

    assert report["settings_without_a_zero_noise_replay"] == ["large-s101-n640"]
    assert report["audited_settings"]["large-s101-n640"]["zne_replayed"] is False
    assert report["audit_complete"] is False
    assert report["primary"]["promoted"] is False
    assert report["primary"]["publication_path"] == "unaudited_not_publishable"


def test_a_replay_naming_the_bound_roster_is_accepted():
    """The gate has to admit the case it exists to distinguish."""
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    report = evaluate_campaign(
        records,
        rosters={record["setting"]: _binding(record) for record in records},
        audits={record["setting"]: _audit(record) for record in records},
    )

    assert report["settings_without_a_zero_noise_replay"] == []
    assert report["audit_complete"] is True
    assert report["primary"]["publication_path"] == "promote_three_of_three"


def test_the_learner_fixed_contrast_is_reported_without_a_promotion_verdict():
    """Both promoted arms move learner and control at once.

    A divergence between them cannot be attributed to the control without a
    comparison that holds the learner fixed. The plan declares that comparison,
    so the driver has to compute it, and it has to stay out of the promotion
    machinery it was never meant to enter.

    The four methods carry four different errors, so the reported MAEs name which
    predictions the contrast consumed. Routing it through any other pair changes
    them.
    """
    pair = (_distinct_errors(0.10, 0.12, 0.03, 0.05),
            _distinct_errors(0.08, 0.12, 0.02, 0.05))
    report = _evaluate(_campaign({101: pair, 211: pair, 307: pair}))

    assert set(report["contrasts"]) == {"primary", "capacity_matched",
                                        "learner_fixed"}
    assert set(report["replication"]) == {"primary", "capacity_matched"}
    assert report["declared_design"]["learner_fixed_contrast"] == {
        "full": "liao", "control": "feat-only"}

    contrast = report["contrasts"]["learner_fixed"]["640"]["101"]
    assert contrast["full_method"] == "liao"
    assert contrast["control_method"] == "feat-only"

    regimes = contrast["families"]["tfi"]["regimes"]
    # liao against feat-only, so 0.03 against 0.12 and 0.02 against 0.12. The
    # primary arm would read 0.10 and the capacity-matched control 0.05.
    assert regimes["shipped"]["full_mae"] == pytest.approx(0.03)
    assert regimes["shipped"]["control_mae"] == pytest.approx(0.12)
    assert regimes["shipped"]["gain"] == pytest.approx(1 - 0.03 / 0.12)
    assert regimes["large"]["full_mae"] == pytest.approx(0.02)
    assert regimes["large"]["control_mae"] == pytest.approx(0.12)
    assert contrast["families"]["tfi"]["contrast"]["estimate"] == pytest.approx(
        (1 - 0.02 / 0.12) - (1 - 0.03 / 0.12))

    # The two promoted arms keep their own pairs, so the third one displaced
    # nothing.
    primary = report["contrasts"]["primary"]["640"]["101"]
    assert primary["full_method"] == "ridge"
    assert primary["families"]["tfi"]["regimes"]["shipped"][
        "full_mae"] == pytest.approx(0.10)

    tables = build_campaign_tables(report)
    assert [row["arm"] for row in tables["endpoints"]].count("learner_fixed") > 0
    assert all(row["arm"] != "learner_fixed" for row in tables["replication"])


def test_the_share_carries_the_unmitigated_reference_beside_the_decomposition():
    """The plan reports R beside A, C, F, so the report has to supply it.

    A signature that accepts `reference_methods` is not a caller that passes
    one, and a record whose retained rows drop `noisy_expectation` cannot build
    one. This test goes through `evaluate_campaign` to the reported number, so
    an unwired reference fails here rather than reading as wired.

    The four arms carry four different errors, so the ladder rungs name which
    predictions produced them: A is feat-only at 0.12, C is liao-feat-only at
    0.05 and F is liao at 0.03. R is the noisy expectation at 0.40, which is
    worse than every rung, so it cannot be confused with one.
    """
    pair = (_distinct_errors(0.10, 0.12, 0.03, 0.05),
            _distinct_errors(0.08, 0.12, 0.02, 0.05))
    records = _campaign({101: pair, 211: pair, 307: pair})
    assert all(row["noisy_expectation"] == 0.40
               for record in records for row in record["test_items"])

    report = _evaluate(records)
    share = report["shares"]["shipped-s101-n640"]

    assert share["reference_methods"] == ["unmitigated"]
    assert share["reference_method"] == "unmitigated"
    assert share["status"] == "estimated"
    family = share["families"]["tfi"]
    reference = family["reference_errors"]["unmitigated"]
    assert reference["estimate"] == pytest.approx(0.40, rel=1e-12)
    assert reference["lower"] == pytest.approx(0.40, rel=1e-12)
    assert reference["upper"] == pytest.approx(0.40, rel=1e-12)

    # The reference stays out of the ladder, so the decomposition is the one the
    # frozen constants declare and the share still divides T rather than R.
    assert set(family["errors"]) == {"A", "C", "F"}
    assert family["errors"]["A"]["estimate"] == pytest.approx(0.12, rel=1e-12)
    assert family["total"]["estimate"] == pytest.approx(0.09, rel=1e-12)
    assert family["share"]["point"] == pytest.approx(0.07 / 0.09, rel=1e-12)

    # R - A is a point difference beside the ladder, not a declared span.
    # A span endpoint has to be a ladder rung.
    assert share["span_labels"] == []
    assert family["spans"] == {}
    points = family["point_values"]
    assert points["reference_errors"]["unmitigated"] - points["errors"][
        "A"] == pytest.approx(0.40 - 0.12, rel=1e-12)

    # The scope travels with the number, so it is the instruction a reader
    # follows. It has to name the same orientation, or the reader subtracts
    # the reference from the rung and reads the improvement A makes as a loss.
    scope = share["reference_scope"]
    assert "R - A is not published as an interval" in scope
    assert ("point_values.reference_errors.unmitigated minus "
            "point_values.errors.A") in scope
    json.dumps(report, allow_nan=False)


def test_the_wrapper_key_drops_the_size_so_both_sizes_draw_one_resample():
    """The production key is the one this pins, not a key the caller supplies.

    The estimator-level size test hands `stream_components` in directly, so it
    says nothing about the tuple `evaluate_setting_share` builds. Mixing the
    size into that tuple leaves the rest of the suite passing and silently
    unpairs the two sizes whose comparison the report advertises as paired.

    The larger side's rows arrive reversed. Row order is an artifact of how a
    record was assembled and the table sorts the circuits it scores, so a key
    that depended on it would be a second way to lose the pairing.
    """
    errors = _distinct_errors(0.10, 0.12, 0.03, 0.05)
    small = _record("shipped", 101, 160, errors, label="one-pool")
    large = _record("shipped", 101, 640, errors, label="one-pool")
    large["test_items"] = list(reversed(large["test_items"]))

    left = evaluate_setting_share(small, n_resamples=40)
    right = evaluate_setting_share(large, n_resamples=40)
    assert (left["size"], right["size"]) == (160, 640)

    for family in FAMILIES:
        circuits = [
            sorted({row["circuit_id"] for row in record["test_items"]
                    if row["family"] == family})
            for record in (small, large)
        ]
        assert circuits[0] == circuits[1] != []

        one, other = left["families"][family], right["families"][family]
        assert one["status"] == other["status"] == "estimated"
        assert one["circuit_order_digest"] == other["circuit_order_digest"]
        assert one["n_circuits"] == other["n_circuits"] == len(circuits[0])
        assert one["stream_seed"] == other["stream_seed"]
        assert np.array_equal(
            draw_matrix(one["n_circuits"], seed=one["stream_seed"],
                        n_resamples=40),
            draw_matrix(other["n_circuits"], seed=other["stream_seed"],
                        n_resamples=40))


def test_two_sizes_that_scored_different_circuits_cannot_be_combined():
    """The pairing claim is checked on the records the report combines.

    `assert_campaign_structure` compares the two sizes' test circuits before
    anything is fitted, but only across the settings it was handed, and records
    are fitted one setting at a time. Without this check the report accepts two
    sizes that scored disjoint pools, publishes both shares as estimated, and
    carries the paired claim beside numbers that were never paired.

    A rehearsal resample count keeps the accepted half cheap; what it exercises
    is the combination rule, which runs before anything is estimated.
    """
    errors = _distinct_errors(0.10, 0.12, 0.03, 0.05)
    disjoint = [_record("shipped", 101, size, errors) for size in (160, 640)]
    with pytest.raises(ValueError, match="retained test circuits differ"):
        _evaluate(disjoint, n_resamples=40)

    paired = [_record("shipped", 101, size, errors, label="one-pool")
              for size in (160, 640)]
    report = _evaluate(paired, n_resamples=40)
    supplied = {record["setting"] for record in paired}
    assert set(report["shares"]) == set(campaign_setting_keys())
    assert all(report["shares"][key]["status"] == "estimated" for key in supplied)
    assert all(report["shares"][key]["status"] == "not_estimable"
               for key in set(report["shares"]) - supplied)


# --------------------------------------------------------------------------
# What a share says about its own eligibility, and which settings it reports
# --------------------------------------------------------------------------


def test_a_rehearsal_share_carries_the_rehearsal_restriction():
    """The rehearsal and audit restrictions reached `replication` alone.

    A rehearsal keeps held-out test rows, so its decomposition is arithmetically
    estimable and says so. Without a restriction of its own that estimated
    number travels with no trace of the prohibition the older primary result
    carries, and a consumer reading `shares` inherits none of it.
    """
    record = _record("shipped", 999, 8,
                     _distinct_errors(0.08, 0.10, 0.02, 0.025), frozen=False)
    report = evaluate_campaign([record], n_resamples=40)
    share = report["shares"][record["setting"]]

    assert report["primary"]["publication_path"] == "rehearsal_not_publishable"
    # A is 0.10, C is 0.025 and F is 0.02, so T is 0.08, K is 0.075, S is 0.9375.
    assert share["status"] == "estimated"
    assert share["families"]["tfi"]["share"]["point"] == pytest.approx(0.9375)
    assert share["publication_path"] == "rehearsal_not_publishable"
    assert share["design_frozen"] is False


def test_an_unaudited_share_carries_the_unaudited_restriction():
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    report = evaluate_campaign(records)

    assert report["design_frozen"] is True
    assert report["primary"]["publication_path"] == "unaudited_not_publishable"
    for share in report["shares"].values():
        assert share["publication_path"] == "unaudited_not_publishable"
        assert share["audit_complete"] is False


def test_a_frozen_audited_share_is_still_not_publishable():
    """The best case this module can reach is still not a publication license.

    The replication result here says `promote_three_of_three`, and the share
    deliberately does not follow it. `evaluate_campaign` reads records and never
    the datasets they name, so it cannot tell a complete retained projection
    from a filtered one, and it opens no manifest. Only a layer that has checked
    both may write a value that permits publication, and none exists yet.
    """
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    report = evaluate_campaign(
        records,
        rosters={record["setting"]: _binding(record) for record in records},
        audits={record["setting"]: _audit(record) for record in records},
    )

    assert report["primary"]["publication_path"] == "promote_three_of_three"
    for share in report["shares"].values():
        assert share["publication_path"] == "unverified_not_publishable"
        assert share["design_frozen"] is True
        assert share["audit_complete"] is True
    assert "eligible_for_publication" not in json.dumps(report, allow_nan=False)


def test_a_record_fitted_from_a_dirty_tree_can_never_be_published():
    """A revision nobody can check out again cannot support a published number.

    `_code_revision` appends `-dirty` when the working tree carried uncommitted
    edits at fit time, and until now nothing read that marker. This is the
    realistic single-author failure: fitting the campaign with edits in flight
    and then publishing the result. It outranks the other three restrictions
    because a fit that cannot be reproduced is not rescued by a passing audit.
    """
    records = _campaign({101: RISING, 211: RISING, 307: RISING})
    records[0] = dict(records[0], code_revision="abc123-dirty")
    report = evaluate_campaign(
        records,
        rosters={record["setting"]: _binding(record) for record in records},
        audits={record["setting"]: _audit(record) for record in records},
    )

    assert report["fitted_from_a_clean_tree"] is False
    for share in report["shares"].values():
        assert share["publication_path"] == "dirty_code_not_publishable"
        assert share["fitted_from_a_clean_tree"] is False
    assert "eligible_for_publication" not in json.dumps(report, allow_nan=False)


def test_every_expected_setting_appears_in_the_share_export():
    """A consumer iterating `shares` has to see the settings that produced nothing.

    Iterating the supplied records alone makes reading the map a selection:
    eleven settings would simply be absent, and a table writer built on it would
    print the twelfth as though it were the grid. An absent setting carries the
    estimator's own not-estimable result, so it answers the questions a supplied
    setting answers rather than carrying a shape invented beside it.
    """
    record = _record("shipped", 101, 640, _distinct_errors(0.10, 0.12, 0.03, 0.05))
    report = _evaluate([record], n_resamples=40)

    assert set(report["shares"]) == set(campaign_setting_keys())
    assert len(report["failed_settings"]) == 11
    supplied = report["shares"]["shipped-s101-n640"]
    assert supplied["record_present"] is True
    assert supplied["status"] == "estimated"

    absent = report["shares"]["large-s307-n160"]
    assert absent["record_present"] is False
    assert absent["status"] == "not_estimable"
    assert (absent["regime"], absent["seed"], absent["size"]) == ("large", 307, 160)
    assert set(absent["families"]) == set(supplied["families"]) == set(FAMILIES)
    for family in FAMILIES:
        entry = absent["families"][family]
        assert set(entry) == set(supplied["families"][family])
        assert entry["status"] == "not_estimable"
        assert entry["reasons"] == ["family_missing_from_the_setting"]
        assert entry["share"]["point"] is None
        assert entry["point_values"] is None
        assert entry["denominator_diagnostics"]["n_draws"] == 0
    # The ladder, the interpretation and the restriction travel with it, so an
    # absent entry cannot be mistaken for a differently defined quantity.
    assert absent["ladder"] == supplied["ladder"]
    assert absent["share_interpretation"] == supplied["share_interpretation"]
    assert absent["publication_path"] == supplied["publication_path"]
    json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------
# Which six of the grid the paper reports, and the grid it reports them in
# --------------------------------------------------------------------------


def test_only_the_six_committed_decompositions_are_labeled_primary():
    """Every entry of the grid used to arrive stamped primary.

    Twelve settings and two families is twenty-four decompositions, and the
    frozen commitment is six of them: the shipped regime at n640, on the three
    declared seeds, in both families. A grid labelled primary throughout leaves
    picking six out of twenty-four to whoever draws the table, which is the
    selection the roles exist to remove.
    """
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))

    roles = {}
    for share in report["shares"].values():
        for family in share["families"]:
            roles.setdefault(share["share_role"], set()).add(
                (share["regime"], share["seed"], share["size"], family))

    assert sum(len(keys) for keys in roles.values()) == 24
    assert sorted(roles["primary"]) == sorted(primary_share_keys())
    assert len(roles["secondary"]) == 18
    assert set(roles) == {"primary", "secondary"}
    # The smaller training size and the contrast regime are the secondary half,
    # and they stay in the report rather than being dropped for not being it.
    assert ("large", 101, 640, "tfi") in roles["secondary"]
    assert ("shipped", 101, 160, "tfi") in roles["secondary"]


def test_a_rehearsal_decomposition_is_labeled_a_rehearsal():
    """A rehearsal reached the report labelled primary alongside the campaign.

    `secondary` would have been wrong too: that is a reported result the paper
    does not headline, and a rehearsal is a result the paper cannot use at all.
    """
    record = _record("shipped", 999, 8,
                     _distinct_errors(0.08, 0.10, 0.02, 0.025), frozen=False)
    report = evaluate_campaign([record], n_resamples=40)

    assert report["shares"][record["setting"]]["share_role"] == "rehearsal"
    # It joins the grid rather than replacing part of it, so the six committed
    # keys are still exactly the six.
    assert len(report["shares"]) == len(campaign_setting_keys()) + 1
    row = next(entry for entry in build_campaign_tables(report)["shares"]
               if entry["setting"] == record["setting"])
    assert row["share_role"] == "rehearsal"
    assert row["publication_path"] == "rehearsal_not_publishable"


def test_the_older_evolution_step_hypothesis_is_labeled_secondary():
    """`report["primary"]` is an older key holding a demoted hypothesis.

    A reader who took the key at its word would read the evolution-step result
    as the campaign's primary one. The label says which it is; the six primary
    decompositions are the ones `shares` marks.
    """
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))

    assert report["primary"]["hypothesis_role"] == "secondary"
    assert "evolution step" in report["primary"]["hypothesis"]


def test_the_table_export_carries_the_whole_ordered_grid():
    """The tables emitted no share rows at all.

    The paper's result therefore had to be read out of `report` by hand, from a
    grid whose entries all read primary. Every decomposition is exported now,
    each row carrying the role, the reference, the ladder, the gaps, the total,
    the share, their intervals, the refusal reasons and the publication fields.
    """
    errors = _distinct_errors(0.10, 0.12, 0.03, 0.05)
    records = [_record(regime, seed, 640, errors)
               for regime in ("shipped", "large") for seed in (101, 211, 307)]
    report = _evaluate(records)
    tables = build_campaign_tables(report)
    rows = tables["shares"]

    assert len(rows) == 24
    assert [tuple(key) for key in tables["primary_share_keys"]] == list(
        primary_share_keys())
    # The order is the report's own, so an exporter cannot reorder the grid.
    assert [row["setting"] for row in rows[:2]] == ["large-s101-n160"] * 2
    assert [row["family"] for row in rows[:2]] == ["heisenberg", "tfi"]

    primary = [row for row in rows if row["share_role"] == "primary"]
    assert len(primary) == 6
    row = next(entry for entry in primary
               if entry["setting"] == "shipped-s101-n640"
               and entry["family"] == "tfi")
    assert row["status"] == "estimated"
    assert row["share_status"] == "estimated"
    assert row["reasons"] == row["share_reasons"] == []
    assert row["n_circuits"] == 8
    assert row["R"] == pytest.approx(0.40)
    assert (row["A"], row["C"], row["F"]) == pytest.approx((0.12, 0.05, 0.03))
    assert (row["K"], row["D"], row["T"]) == pytest.approx((0.07, 0.02, 0.09))
    assert row["S"] == pytest.approx(0.07 / 0.09)
    for label in ("R", "A", "C", "F", "K", "D", "T", "S"):
        assert row[f"{label}_lower"] <= row[label] <= row[f"{label}_upper"]
    assert row["publication_path"] == "unverified_not_publishable"
    assert (row["design_frozen"], row["audit_complete"],
            row["fitted_from_a_clean_tree"]) == (True, True, True)
    # Written by the driver, which is the layer that opens the freeze. This
    # report came from the library, so they are absent rather than claimed.
    assert row["freeze_verified"] is None
    assert row["fit_bindings_verified"] is None
    assert row["freeze_manifest_sha256"] is None
    json.dumps(rows, allow_nan=False)


def test_an_unavailable_decomposition_is_exported_as_a_placeholder():
    """A committed key that produced nothing is a row of nulls, not an absence.

    Exporting only what was estimated would make the table a selection of the
    settings that succeeded, and a small share, a negative gap and a withheld
    share are all outcomes the paper reports as they come.
    """
    rows = build_campaign_tables(evaluate_campaign([]))["shares"]

    assert len(rows) == 24
    primary = [row for row in rows if row["share_role"] == "primary"]
    assert sorted((row["regime"], row["seed"], row["size"], row["family"])
                  for row in primary) == sorted(primary_share_keys())
    for row in primary:
        assert row["record_present"] is False
        assert row["status"] == "not_estimable"
        assert row["reasons"] == ["family_missing_from_the_setting"]
        for label in ("R", "A", "C", "F", "K", "D", "T", "S"):
            assert row[label] is None
            assert row[f"{label}_lower"] is None
            assert row[f"{label}_upper"] is None
    json.dumps(rows, allow_nan=False)


def test_a_negative_gap_and_a_withheld_share_are_exported_as_they_come():
    """The reporting rule admits all three outcomes, so the export carries them.

    A negative gap is a number the grid prints; a withheld share is a row whose
    S is null beside the reason. Neither is an occasion to drop the row, and a
    row that vanished would leave the surviving rows reading as the grid.
    """
    # A = 0.05, C = 0.08, F = 0.03: K is negative and T is still positive.
    negative = _record("shipped", 101, 640,
                       _distinct_errors(0.03, 0.05, 0.03, 0.08))
    # A = 0.03, C = 0.04, F = 0.05: the improvement runs backwards, so the
    # denominator is not positive and the share is withheld.
    withheld = _record("shipped", 211, 640,
                       _distinct_errors(0.05, 0.03, 0.05, 0.04))
    rows = build_campaign_tables(
        _evaluate([negative, withheld], n_resamples=40))["shares"]

    first = next(row for row in rows if row["setting"] == "shipped-s101-n640"
                 and row["family"] == "tfi")
    assert first["share_role"] == "primary"
    assert first["K"] == pytest.approx(-0.03)
    assert first["T"] == pytest.approx(0.02)
    assert first["S"] == pytest.approx(-1.5)

    second = next(row for row in rows if row["setting"] == "shipped-s211-n640"
                  and row["family"] == "tfi")
    assert second["share_role"] == "primary"
    assert second["status"] == "estimated"
    assert second["T"] == pytest.approx(-0.02)
    assert second["share_status"] == "not_estimable"
    assert "nonpositive_point_total" in second["share_reasons"]
    assert second["S"] is None
    json.dumps(rows, allow_nan=False)


def test_a_grid_missing_a_committed_decomposition_is_refused():
    """The projection is counted rather than trusted to the entries' own labels.

    An entry carries its role and nothing else counts them, so a grid short of a
    committed key or one whose primary label spread again reads as correct from
    any single row.
    """
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))
    del report["shares"]["shipped-s101-n640"]

    with pytest.raises(ValueError, match="primary projection"):
        build_campaign_tables(report)


def test_a_decomposition_whose_role_disagrees_with_its_key_is_refused():
    """The role is derived from the key on both sides, so they cannot drift."""
    report = _evaluate(_campaign({101: RISING, 211: RISING, 307: RISING}))
    report["shares"]["large-s101-n640"]["share_role"] = "primary"

    with pytest.raises(ValueError, match="this family's key is 'secondary'"):
        build_campaign_tables(report)
