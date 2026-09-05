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

import pytest

from qem_bench.campaign.analysis import (
    RECORD_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    assert_campaign_structure,
    build_campaign_tables,
    build_setting_record,
    evaluate_campaign,
    regenerate_setting_diagnostics,
)
from qem_bench.campaign.design import (
    ARMS,
    campaign_setting_keys,
    campaign_split_spec,
    expected_circuits,
    role_counts,
    setting_key,
)
from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import SplitSpec
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


RISING = (_errors(0.10, 0.10), _errors(0.08, 0.10))
FLAT = (_errors(0.10, 0.10), _errors(0.10, 0.10))
FALLING = (_errors(0.08, 0.10), _errors(0.10, 0.10))


def _evaluate(records, **kwargs):
    """Evaluate at the frozen resample count, which is what a record can claim."""
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
