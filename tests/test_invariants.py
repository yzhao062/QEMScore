"""Focused protocol-invariant tests: physics oracles, feature discipline,
grouped validation, dataset validation, and noise coverage."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Operator
from sklearn.model_selection import GroupKFold

from qem_bench.baselines.ridge import RidgeMitigator
from qem_bench.circuits.tfi import TFIParams, build_tfi_circuit
from qem_bench.datasets.generate import dataset_hash, generate
from qem_bench.datasets.schema import FAMILY_STRATA, FEATURE_SPEC_VERSION, FEATURES
from qem_bench.noise.models import BASIS_GATES, SEVERITY_GRID, build_noise_model
from qem_bench.observables import z_expectation_from_counts, z_support_label
from qem_bench.runner.run import run


def _rewrite_dataset(src, dst, mutate_row=None, mutate_manifest=None):
    """Copy a dataset, apply row/manifest mutations, and recompute the canonical
    hash so the deeper validation layers (not the hash check) are exercised."""
    shutil.copytree(src, dst)
    items = [
        json.loads(line) for line in (dst / "items.jsonl").read_text().splitlines() if line
    ]
    for item in items:
        item.setdefault("stratum", FAMILY_STRATA[item["family"]])
    if mutate_row is not None:
        mutate_row(items)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["dataset_hash"] = dataset_hash(items)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    lines = [
        json.dumps(it, sort_keys=True, separators=(",", ":"))
        for it in sorted(items, key=lambda it: it["item_id"])
    ]
    (dst / "items.jsonl").write_text("\n".join(lines) + "\n")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.fixture(scope="module")
def micro_dataset(tmp_path_factory):
    data = tmp_path_factory.mktemp("micro") / "data"
    manifest = generate("t0-micro", data)
    return data, manifest


def test_tfi_unitary_matches_first_order_product():
    """One Trotter step must equal exp(-i H_zz dt) then exp(-i H_x dt) for
    H = -J ZZ - h (X0 + X1) on two qubits."""
    j, h, dt = 0.7, 0.9, 0.2
    params = TFIParams(n_qubits=2, steps=1, j=j, h=h, dt=dt, circuit_seed=0, instance=0)
    u_circuit = Operator(build_tfi_circuit(params)).data

    # exp(-i * (-J ZZ) * dt): ZZ is diagonal with entries [1, -1, -1, 1].
    zz_diag = np.array([1.0, -1.0, -1.0, 1.0])
    u_zz = np.diag(np.exp(1j * j * dt * zz_diag))
    # exp(-i * (-h X) * dt) per qubit = cos(h dt) I + i sin(h dt) X.
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    u1 = np.cos(h * dt) * np.eye(2) + 1j * np.sin(h * dt) * x
    u_x = np.kron(u1, u1)
    expected = u_x @ u_zz  # circuit applies RZZ first, then RX

    assert np.allclose(u_circuit, expected, atol=1e-10)


def test_z_labels_and_count_parity():
    assert z_support_label(2, (0,)) == "IZ"
    assert z_support_label(2, (1,)) == "ZI"
    # Key '01' means q1=0, q0=1 (Qiskit bit order: rightmost character is qubit 0).
    counts = {"01": 100}
    est_q0, _ = z_expectation_from_counts(counts, (0,), 100)
    est_q1, _ = z_expectation_from_counts(counts, (1,), 100)
    est_zz, _ = z_expectation_from_counts(counts, (0, 1), 100)
    assert est_q0 == -1.0
    assert est_q1 == 1.0
    assert est_zz == -1.0


def test_forbidden_fields_stay_out_of_features():
    forbidden = {
        "severity",
        "stratum",
        "noise_family",
        "p1",
        "p2",
        "p_ro",
        "noisy_stderr",
        "ideal_expectation",
        "sampler_seed",
        "circuit_seed",
        "measurement_group",
        "label_method",
        "split",
    }
    assert forbidden.isdisjoint(set(FEATURES))


def test_feature_spec_v1_is_the_frozen_ordered_contract():
    assert FEATURE_SPEC_VERSION == "v1"
    assert FEATURES == [
        "noisy_expectation",
        "log2_shots",
        "n_qubits",
        "family_tfi",
        "family_qaoa",
        "family_heisenberg",
        "family_random_clifford",
        "family_near_clifford",
        "steps",
        "j",
        "h",
        "jx",
        "jy",
        "jz",
        "dt",
        "qaoa_p",
        "qaoa_edge_count",
        "qaoa_gamma_0",
        "qaoa_gamma_1",
        "qaoa_beta_0",
        "qaoa_beta_1",
        "graph_path",
        "graph_cycle",
        "graph_erdos_renyi",
        "graph_3_regular",
        "rc_depth",
        "nc_non_clifford_count",
        "two_qubit_gates",
        "transpiled_depth",
        "obs_locality",
    ]


def test_cv_folds_are_circuit_disjoint(micro_dataset):
    data, _ = micro_dataset
    items = [
        json.loads(line) for line in (data / "items.jsonl").read_text().splitlines() if line
    ]
    train = [it for it in items if it["split"] == "train"]
    model = RidgeMitigator().fit(train)
    assert isinstance(model._search.cv, GroupKFold)

    groups = np.array([f"{it['family']}:{it['instance']}" for it in train])
    x = np.zeros((len(train), 1))
    for tr_idx, va_idx in GroupKFold(n_splits=model.cv).split(x, groups=groups):
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))


def test_run_rejects_tampered_dataset(micro_dataset, tmp_path):
    data, _ = micro_dataset
    tampered = tmp_path / "tampered"
    shutil.copytree(data, tampered)
    lines = (tampered / "items.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["noisy_expectation"] = round(row["noisy_expectation"] + 0.1, 12)
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    (tampered / "items.jsonl").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        run(tampered, tmp_path / "out")


def test_noise_model_covers_transpiled_ops():
    """Every transpiled operation except rz and barrier must carry an error in the
    actual NoiseModel of every severity (mutation-sensitive: an empty or thinned
    noise model fails this test)."""
    params = TFIParams(n_qubits=3, steps=2, j=0.5, h=0.8, dt=0.2, circuit_seed=1, instance=0)
    circ = build_tfi_circuit(params)
    circ.measure_all()
    tcirc = transpile(circ, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=1)
    ops = set(tcirc.count_ops())
    noiseless_ok = {"rz", "barrier"}

    for severity in SEVERITY_GRID:
        model_dict = build_noise_model(severity).to_dict()
        covered = set()
        for err in model_dict.get("errors", []):
            covered.update(err.get("operations", []))
        assert ops - noiseless_ok <= covered, (
            f"severity {severity}: ops {sorted(ops - noiseless_ok - covered)} carry no error"
        )


def test_run_rejects_inconsistent_measurement_group(micro_dataset, tmp_path):
    """A group whose sibling rows disagree on an invariant field is impossible
    (one shared execution), and must be rejected even when the hash is current."""
    data, _ = micro_dataset

    def flip_seed(items):
        items[0]["sampler_seed"] = items[0]["sampler_seed"] + 1

    tampered = tmp_path / "bad-group"
    _rewrite_dataset(data, tampered, mutate_row=flip_seed)
    with pytest.raises(ValueError, match="measurement group .* disagrees"):
        run(tampered, tmp_path / "out1")


def test_run_rejects_false_manifest_counts(micro_dataset, tmp_path):
    data, _ = micro_dataset

    def inflate_counts(manifest):
        manifest["counts"]["items"] += 2

    tampered = tmp_path / "bad-counts"
    _rewrite_dataset(data, tampered, mutate_manifest=inflate_counts)
    with pytest.raises(ValueError, match="manifest counts"):
        run(tampered, tmp_path / "out2")


def test_run_rejects_false_label_evals(micro_dataset, tmp_path):
    data, _ = micro_dataset

    def inflate_labels(manifest):
        manifest["generation_ledger"]["label_evals_statevector"] += 1

    tampered = tmp_path / "bad-labels"
    _rewrite_dataset(data, tampered, mutate_manifest=inflate_labels)
    with pytest.raises(ValueError, match="label_evals_statevector"):
        run(tampered, tmp_path / "out3")


def test_v1_runner_rejects_v0_feature_manifest(micro_dataset, tmp_path):
    data, _ = micro_dataset

    def downgrade_feature_spec(manifest):
        manifest["feature_spec"] = {
            "version": "v0",
            "features": [
                "noisy_expectation",
                "log2_shots",
                "n_qubits",
                "steps",
                "j",
                "h",
                "dt",
                "two_qubit_gates",
                "transpiled_depth",
                "obs_locality",
            ],
        }

    old_manifest = tmp_path / "v0-manifest"
    _rewrite_dataset(data, old_manifest, mutate_manifest=downgrade_feature_spec)
    with pytest.raises(ValueError, match="feature_spec"):
        run(old_manifest, tmp_path / "out-v0")


def test_run_rejects_foreign_label_method(micro_dataset, tmp_path):
    """A row declaring an unsupported method is rejected."""
    data, _ = micro_dataset

    def flip_method(items):
        items[0]["label_method"] = "analytic-placeholder"

    tampered = tmp_path / "bad-method"
    _rewrite_dataset(data, tampered, mutate_row=flip_method)
    with pytest.raises(ValueError, match="label_method"):
        run(tampered, tmp_path / "out4")


def test_run_rejects_mixed_strata_with_headline_rule(micro_dataset, tmp_path):
    data, _ = micro_dataset
    changed_rows = 0

    def mix_strata(items):
        nonlocal changed_rows
        group = items[0]["measurement_group"]
        for item in items:
            if item["measurement_group"] == group:
                item["family"] = "random_clifford"
                item["stratum"] = "clifford_control"
                item["label_method"] = "stim"
                item["depth"] = 2
                changed_rows += 1

    def repair_label_ledger(manifest):
        manifest["generation_ledger"]["label_evals_statevector"] -= changed_rows
        manifest["generation_ledger"]["label_evals_stim"] += changed_rows

    tampered = tmp_path / "mixed-strata"
    _rewrite_dataset(
        data,
        tampered,
        mutate_row=mix_strata,
        mutate_manifest=repair_label_ledger,
    )
    with pytest.raises(
        ValueError,
        match="Clifford control stratum is excluded from the continuous-regression headline",
    ):
        run(tampered, tmp_path / "out5")


def test_cli_subprocess_determinism(tmp_path):
    for name in ("a", "b"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "qem_bench.cli",
                "generate",
                "--preset",
                "t0-micro",
                "--out",
                str(tmp_path / name),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr
    h_a = json.loads((tmp_path / "a" / "manifest.json").read_text())["dataset_hash"]
    h_b = json.loads((tmp_path / "b" / "manifest.json").read_text())["dataset_hash"]
    assert h_a == h_b
