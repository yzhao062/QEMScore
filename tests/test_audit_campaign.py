"""The independent audit: what it re-derives, and what it actually catches.

Round 3 of the plan review injected `sampler_seed + 1` at measurement execution
and showed that the resulting dataset passed artifact validation and every
structural assertion while 47 of 48 stored expectations had moved. That fault is
the audit's reason to exist, so it is the fault these tests inject.
"""

import copy
import json

import pytest

from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import SplitSpec
from qem_bench.validation import validate_split_artifact
from tools.audit_campaign import (
    AUDIT_INDICES,
    CHECK_COVERAGE,
    audit_dataset,
)
import tools.audit_campaign as audit

SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")


def _small_spec():
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
        role_counts={"train": 4, "validation": 4, "test": 4},
        family_parameters={"tfi": {"dt": 0.2}, "heisenberg": {"dt": 0.15}},
        budget_tier="H",
    )


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    """A small S0 split with the campaign's shape at a size a test can build."""
    root = tmp_path_factory.mktemp("audit") / "data"
    generate_split(_small_spec(), root, master_seed=101)
    return root


@pytest.fixture(scope="module")
def clean_report(artifact):
    return audit_dataset(artifact)


def test_the_audit_re_derives_every_sampled_circuit(clean_report):
    report = clean_report

    assert report["passed"] is True
    assert report["mismatches"] == []
    # Index 0 of each family and role exists at this size; the larger indices do
    # not, and the rule skips them rather than failing.
    assert report["circuits"] == 6
    assert report["measurement_groups"] == 12
    assert report["rows"] == 24
    assert report["checks"]["histogram_replay"] == 12
    assert report["checks"]["exact_label"] == 24
    json.dumps(report, allow_nan=False)


def test_the_audit_says_which_checks_are_coverage_and_which_re_derive(clean_report):
    """`validate_split_artifact` already recomputes most of this, and says so."""
    coverage = clean_report["check_coverage"]

    # Every check that ran is labelled; the zero-noise one runs only with a
    # roster, so it is declared without appearing in this run's counts.
    assert set(clean_report["checks"]) <= set(coverage)
    assert set(coverage) - set(clean_report["checks"]) == {"zne_prediction"}
    assert {name for name, kind in coverage.items() if kind == "rederived"} == {
        "circuit_identity", "circuit_sidecar", "counts_sidecar", "observable",
        "noisy_estimate", "exact_label"}
    assert {name for name, kind in coverage.items() if kind == "independent"} == {
        "independent_protocol", "independent_label"}
    assert coverage == CHECK_COVERAGE
    assert "no folded histogram is stored" in clean_report["scope"]


def test_the_parameter_and_structure_pass_reaches_every_circuit(clean_report):
    """The index rule selects the replay sample; the draw check is exhaustive.

    Twelve pool instances and forty-eight rows, against six sampled circuits and
    twenty-four sampled rows. A check that only ever looked at the sample would
    leave most of the campaign's circuits unexamined.
    """
    checks = clean_report["checks"]

    assert checks["parameter_draw"] == 12
    assert checks["redrawn_identity"] == 12
    assert checks["exhaustive_structure"] == 48
    assert checks["pool_descriptor"] == 6
    # The sampled replay stays a sample.
    assert checks["histogram_replay"] == 12
    assert clean_report["circuits"] == 6


def test_an_extra_consumed_draw_is_caught_although_every_seed_is_intact(tmp_path):
    """Round 4's counterexample, which every seed-and-value check passes.

    Consuming one extra random number before each Heisenberg draw shifts that
    family's parameter stream. The recorded seeds do not move, the recorded
    parameters stay consistent with the circuits built from them, and the whole
    hash chain validates. Only replaying the stream from the master seed sees it.
    """
    import importlib

    # `qem_bench.datasets.generate` resolves to the package's re-exported
    # function rather than the submodule, so the module has to be fetched by
    # name; patching the attribute of the function would silently do nothing and
    # the test would pass against an unfaulted dataset.
    module = importlib.import_module("qem_bench.datasets.generate")
    original = module.sample_heisenberg_params
    calls = {"n": 0}

    def faulty(rng, *args, **kwargs):
        calls["n"] += 1
        rng.random()
        return original(rng, *args, **kwargs)

    root = tmp_path / "faulted"
    module.sample_heisenberg_params = faulty
    try:
        generate_split(_small_spec(), root, master_seed=101)
    finally:
        module.sample_heisenberg_params = original
    assert calls["n"] == 6, "the fault has to reach every Heisenberg draw"

    # The fault survives every shipped check.
    assert validate_split_artifact(root)[1]["master_seed"] == 101

    report = audit_dataset(root)
    assert report["passed"] is False
    assert {value["check"] for value in report["mismatches"]} == {"parameter_draw"}
    assert len(report["mismatches"]) == 6
    # One consumed draw shifts the stream by one, so each recorded coupling is
    # the one the replay draws for the previous field.
    detail = report["mismatches"][0]["detail"]
    assert detail["jy"]["redrawn"] == pytest.approx(detail["jx"]["recorded"])


def test_a_compiled_depth_wrong_on_one_row_alone_is_caught(artifact, monkeypatch):
    """Comparing against a group's first row would miss the other rows."""
    seen = {"calls": 0}
    original = audit._compile

    def drifting(circuit, transpile_seed):
        result = original(circuit, transpile_seed)
        seen["calls"] += 1
        return result

    monkeypatch.setattr(audit, "_compile", drifting)
    report = audit_dataset(artifact)

    # One compile per circuit, but a comparison against every row it produced.
    assert seen["calls"] == 12
    assert report["checks"]["exhaustive_structure"] == 48


def test_a_label_routine_that_drifts_is_caught_only_by_the_independent_one(tmp_path):
    """The fault class the shared-code replay cannot see.

    Generate with a label routine that is wrong by a constant. The recorded label
    carries that error, and the audit's re-derivation calls the same routine, so
    it agrees and reports nothing. The independent evolution disagrees.
    """
    import importlib

    generator = importlib.import_module("qem_bench.datasets.split_generate")
    validator = importlib.import_module("qem_bench.validation")
    original = generator.statevector_expectation
    calls = {"n": 0}

    def drifting(circuit, pauli):
        calls["n"] += 1
        return original(circuit, pauli) + 1e-6

    # The fault has to be live for generation, for validation, and for the
    # audit's own re-derivation. A routine that is wrong only while the data is
    # written is caught by the validator recomputing labels correctly; a routine
    # that is simply wrong agrees with itself everywhere, and that is the fault
    # a second implementation exists to find.
    root = tmp_path / "drifted"
    generator.statevector_expectation = drifting
    validator.statevector_expectation = drifting
    audit.statevector_expectation = drifting
    try:
        generate_split(_small_spec(), root, master_seed=101)
        assert calls["n"] > 0, "the fault has to reach the label calculation"
        report = audit_dataset(root)
    finally:
        generator.statevector_expectation = original
        validator.statevector_expectation = original
        audit.statevector_expectation = original

    assert report["passed"] is False
    kinds = {value["check"] for value in report["mismatches"]}
    assert kinds == {"independent_label"}
    # The shared-code re-derivation ran on every row and agreed with all of them.
    assert report["checks"]["exact_label"] == 24
    assert len(report["mismatches"]) == 24


def test_a_drifting_circuit_builder_is_caught_by_the_independent_protocol(
    artifact, monkeypatch,
):
    """A builder whose angle convention moved agrees with itself everywhere."""
    from qiskit import QuantumCircuit

    original = audit.build_circuit_from_canonical_descriptor

    def drifting(descriptor):
        source = original(descriptor)
        rebuilt = QuantumCircuit(source.num_qubits)
        for instruction in source.data:
            qubits = [source.find_bit(bit).index for bit in instruction.qubits]
            angle = float(instruction.operation.params[0]) * 1.000001
            getattr(rebuilt, instruction.operation.name)(angle, *qubits)
        return rebuilt

    monkeypatch.setattr(
        audit, "build_circuit_from_canonical_descriptor", drifting)
    report = audit_dataset(artifact)

    assert "independent_protocol" in {
        value["check"] for value in report["mismatches"]}


def test_the_independent_protocol_covers_the_sample(clean_report):
    checks = clean_report["checks"]

    assert checks["independent_protocol"] == 6
    assert checks["independent_label"] == 24
    assert CHECK_COVERAGE["independent_protocol"] == "independent"
    assert CHECK_COVERAGE["independent_label"] == "independent"


def test_the_independent_evolution_conserves_probability():
    """A normalization slip would move every label and no other check."""
    from tools.audit_protocol import (
        normalized_probability, protocol_gates, statevector)

    for family, parameters in (
        ("tfi", {"n_qubits": 4, "steps": 3, "j": 0.7, "h": 0.4, "dt": 0.2}),
        ("heisenberg",
         {"n_qubits": 4, "steps": 3, "jx": 0.6, "jy": 0.9, "jz": 0.3, "dt": 0.6}),
    ):
        gates = protocol_gates(family, parameters)
        state = statevector(gates, int(parameters["n_qubits"]))
        assert normalized_probability(state) == pytest.approx(1.0, abs=1e-12)


def test_the_frozen_index_rule_is_the_one_the_plan_review_declared():
    assert AUDIT_INDICES == {
        "train": (0, 79, 159, 160, 399, 639),
        "validation": (0, 159, 319),
        "test": (0, 79, 159),
    }


def test_a_measurement_replayed_from_another_seed_is_caught(artifact, monkeypatch):
    """The fault round 3 injected: valid hashes, valid structure, moved numbers."""
    original = audit.sample_counts

    def wrong_seed(circuit, **kwargs):
        return original(circuit, **dict(kwargs, sampler_seed=kwargs["sampler_seed"] + 1))

    monkeypatch.setattr(audit, "sample_counts", wrong_seed)
    report = audit_dataset(artifact)

    assert report["passed"] is False
    assert {value["check"] for value in report["mismatches"]} == {"histogram_replay"}
    # Every group is wrong, and the audit names each rather than stopping at one.
    assert len(report["mismatches"]) == 12


def test_a_single_moved_shot_is_caught(artifact, monkeypatch):
    original = audit.sample_counts

    def moved(circuit, **kwargs):
        counts, structure = original(circuit, **kwargs)
        counts = dict(counts)
        keys = sorted(counts)
        counts[keys[0]] -= 1
        counts[keys[-1]] = counts.get(keys[-1], 0) + 1
        return counts, structure

    monkeypatch.setattr(audit, "sample_counts", moved)
    report = audit_dataset(artifact)

    assert {value["check"] for value in report["mismatches"]} == {"histogram_replay"}


def test_a_compiled_depth_that_disagrees_is_caught(artifact, monkeypatch):
    """Two feature columns no shipped check verifies, so the audit does."""
    original = audit.sample_counts

    def deeper(circuit, **kwargs):
        counts, structure = original(circuit, **kwargs)
        return counts, dict(
            structure, transpiled_depth=int(structure["transpiled_depth"]) + 1)

    monkeypatch.setattr(audit, "sample_counts", deeper)
    report = audit_dataset(artifact)

    assert {value["check"] for value in report["mismatches"]} == {"structure"}
    assert all(value["detail"]["field"] == "transpiled_depth"
               for value in report["mismatches"])


def test_a_rebuilt_circuit_that_does_not_match_its_identity_is_caught(
    artifact, monkeypatch,
):
    """The identity names the executable circuit, so a changed one is a new one."""
    original = audit.canonical_physical_circuit_identity

    def shifted(item):
        item = dict(item)
        if "dt" in item:
            item["dt"] = float(item["dt"]) + 1e-6
        return original(item)

    monkeypatch.setattr(audit, "canonical_physical_circuit_identity", shifted)
    report = audit_dataset(artifact)

    # The exhaustive redraw reaches every circuit and the sampled pass reaches
    # six, so both identity checks fire and neither is silent.
    assert {value["check"] for value in report["mismatches"]} == {
        "circuit_identity", "redrawn_identity"}
    assert sum(value["check"] == "redrawn_identity"
               for value in report["mismatches"]) == 12


def test_an_observable_the_campaign_does_not_use_is_refused(artifact):
    with pytest.raises(ValueError, match="audits z_mid and zz_mid"):
        audit._support("zz_edge", 10)


def test_the_audit_reads_only_the_indices_it_declares(artifact):
    """Narrowing the rule narrows the sample, so the rule is what selects."""
    report = audit_dataset(artifact, indices={"train": (0,), "validation": (),
                                              "test": ()})

    assert report["circuits"] == 2
    assert report["audited_indices"] == {
        "train": [0], "validation": [], "test": []}
    assert report["passed"] is True


def test_a_tampered_counts_sidecar_never_reaches_the_audit(artifact, tmp_path):
    """Editing a stored histogram is caught before the audit does any work.

    Sidecars are content-addressed and their hashes sit in the manifest, so
    `validate_split_artifact` refuses the dataset outright. This is why the
    audit's own sidecar and label checks are labelled re-derivation rather than
    coverage: the fault they would find is one nothing can reach them with. The
    fault that does reach them is a correctly hashed histogram that the sampler
    would never have produced, which the previous test injects.
    """
    root = tmp_path / "tampered"
    root.mkdir()
    for source in artifact.rglob("*"):
        target = root / source.relative_to(artifact)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    items, _ = validate_split_artifact(root)
    row = next(item for item in items
               if item["split"] == "train" and item["instance"] == 0)
    path = root / str(row["counts_sidecar"])
    payload = json.loads(path.read_bytes())
    counts = dict(payload["counts"])
    keys = sorted(counts)
    counts[keys[0]] -= 1
    counts[keys[-1]] = counts.get(keys[-1], 0) + 1
    payload["counts"] = counts
    path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())

    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        audit_dataset(root)


@pytest.fixture(scope="module")
def roster(artifact, tmp_path_factory):
    """The shipped roster's own scoring of the same artifact, ZNE among it."""
    from qem_bench.runner.run import run, validate_run_artifact

    result = run(artifact, tmp_path_factory.mktemp("roster"))
    validate_run_artifact(result)
    return result


def test_the_folded_executions_replay_to_the_prediction_the_roster_recorded(
    artifact, roster,
):
    """Scale seeds re-derived from the declared rule, not read out of the code."""
    report = audit_dataset(artifact, roster=roster)

    assert report["passed"] is True
    assert report["zne_replay"]["checked"] is True
    assert report["zne_replay"]["scales"] == [3, 5]
    assert report["zne_replay"]["roster_artifact_id"] == roster["artifact_id"]
    # Two test circuits at index 0, two severities, two observables.
    assert report["checks"]["zne_prediction"] == 8
    assert report["zne_replay_executions"] == 8


def test_without_a_roster_no_zero_noise_result_is_claimed(clean_report):
    assert clean_report["zne_replay"]["checked"] is False
    assert clean_report["zne_replay"]["roster_artifact_id"] is None
    assert "zne_prediction" not in clean_report["checks"]


def test_a_zero_noise_prediction_that_moved_is_caught(artifact, roster):
    moved = copy.deepcopy(roster)
    predictions = list(moved["methods"]["zne"]["predictions"])
    predictions[0] = float(predictions[0]) + 1e-6
    moved["methods"]["zne"]["predictions"] = predictions

    report = audit_dataset(artifact, roster=moved)

    assert report["passed"] is False
    assert {value["check"] for value in report["mismatches"]} == {"zne_prediction"}
    assert len(report["mismatches"]) == 1


def test_a_roster_scored_from_another_dataset_is_refused(artifact, roster):
    foreign = copy.deepcopy(roster)
    foreign["dataset_hash"] = "someone-elses-dataset"

    with pytest.raises(ValueError, match="the roster scored dataset"):
        audit_dataset(artifact, roster=foreign)


def test_a_folded_execution_replayed_from_another_seed_is_caught(
    artifact, roster, monkeypatch,
):
    """The scale seeds are part of the protocol, so a wrong one has to show."""
    original = audit._scale_seeds

    def shifted(master_seed, group, scale_factor):
        sampler, transpile = original(master_seed, group, scale_factor)
        return sampler + 1, transpile

    monkeypatch.setattr(audit, "_scale_seeds", shifted)
    report = audit_dataset(artifact, roster=roster)

    assert {value["check"] for value in report["mismatches"]} == {"zne_prediction"}
    assert len(report["mismatches"]) == 8


def test_the_report_names_the_dataset_it_audited(clean_report, artifact):
    _, manifest = validate_split_artifact(artifact)

    assert clean_report["dataset_hash"] == manifest["dataset_hash"]
    assert clean_report["split_spec_hash"] == manifest["split_spec_hash"]
    assert clean_report["master_seed"] == manifest["master_seed"]
    assert copy.deepcopy(clean_report)["label_tolerance"] > 0
