"""Descriptor-information ladder, Part A: spin chains on the same circuits.

Frozen rule: docs/frozen-rules/2026-10-02-descriptor-information.md (Part A,
Arms, the R0 gate, and arm M on the near-Clifford datasets). Fitting, output
format, and convergence flags live in tools/descriptor_common.py and are
shared with Part B (tools/qaoa_intermediate.py).

Subcommands
-----------
gate     R0 reproduction gate. Through the feature-builder hook with the
         default builder (``build_features``, ``FEATURES``), refits F and C at
         the original learner seed (learner seed = dataset seed, as the
         campaign did) on the three regenerated datasets and compares the
         selected candidate's test predictions with the campaign archive. The
         frozen gate (criterion ``archive``) passes when the largest absolute
         difference is at most 1e-11. It also reports, outside the pass
         criterion, the difference from the pre-hook original-seed refits of
         Appendix M.4 (``--seedrep-fits``) and arm A at R0 against the archived
         affine feature-only control. ``--criterion m4-refit-identity`` or a
         ``--tolerance`` other than 1e-11 amends the frozen rule; the gate then
         runs only with ``--amendment TEXT`` whose text is already written into
         the rule file, and it records the amendment and the frozen criterion's
         own verdict in gate_r0.json. Choosing an amendment is the author's
         decision, never the tool's.
prepare  Verifies the datasets (the archive check of
         tools/learner_seed_replication.py), caches their rows, writes the
         N-rung coupling-noise table, builds the R4 encoder cache (one encoded
         vector per circuit), and writes the label-free R4 diagnostics. No fit.
run      Fits every planned rung, arm and learner seed. Refuses unless the gate
         file says the gate passed. ``--dry-run`` only lists the planned fits;
         ``--rungs`` and ``--arms`` filter them. A rerun resumes. When the plan
         holds a re-export rung (R0, NC-R0), it first checks that the cached
         row order equals the reference's, and afterwards runs check-reexport.
check-reexport
         No fit. Compares every re-exported fit with the fit it re-exports
         (R0 with Appendix M.4, NC-R0 with Appendix N) and writes
         reexport_check.json: a fit is ``exact`` when its selected predictions
         equal the reference's bit for bit (test and validation for M.4; test
         for Appendix N, which stored only test predictions), with the same
         learner and shuffle seeds and the same selected candidate.

Rungs (column layouts; every rung keeps the columns the rule keeps)
-------------------------------------------------------------------
R0       ``build_features`` itself (the schema vector, 30 columns). The rule
         reuses the Appendix M.4 fits, which stored only the selected
         candidate's predictions. The default plan therefore re-exports them:
         F, C, P at learner seeds 1 to 20 plus M.4's original-seed P (learner
         seed = dataset seed, archived shuffle seed 1234), refitted through the
         hook so that each fit also stores the random-forest and MLP
         predictions the learner-fixed secondary D needs. M.4's original-seed F
         and C are re-exported by the gate's own fits (gate/fits). check-reexport
         verifies each re-export against M.4 exactly, so R0 still counts as
         reusing M.4. No A at R0: the archived A and the gate's A check exist.
N1..N4   The schema vector with each coupling replaced by ``coupling + s*z``,
         s = 0.01, 0.03, 0.10, 0.30, no clipping. z is one standard-normal draw
         per circuit and coupling from
         ``default_rng(SeedSequence([20261002, dataset_seed, circuit_index]))``,
         circuit_index being the circuit's position in the sorted list of every
         circuit_id in the dataset (all roles, both families), drawn as
         ``standard_normal(n_couplings)`` in schema order (TFI: j, h;
         Heisenberg: jx, jy, jz). The same z serves every N rung and every
         split, because it is keyed to the circuit.
R3-TFI   The schema vector without the ``h`` column. Heisenberg rows carry 0 in
         that column, so they keep their R0 information.
R3-Heis  The schema vector without the ``jz`` column.
R4       The schema vector with steps, j, h, jx, jy, jz, dt replaced (in place)
         by the 165 released ML-QEM features of the item's transpiled circuit
         (tools/mlqem_binned_angle_encoder.py, scaled by 0.01 as released).
R5       Only the kept columns: noisy estimate, log2 shots, qubit count, the
         five family indicators, two-qubit gate count, compiled depth,
         observable locality.
NC-none  R5's columns on the near-Clifford datasets of Appendix N; arms M
         (= F there) and C.
NC-R0    Re-export of the Appendix N fits on the near-Clifford datasets: the
         schema vector (``build_features``, the descriptors Appendix N used),
         arms F, C, P at learner seeds 1 to 20 plus Appendix N's anchor seed
         (learner seed = dataset seed; P's shuffle seed = learner seed, as
         Appendix N did). check-reexport verifies each against Appendix N's
         stored test predictions exactly. No A (Appendix N stored A).

Choices where the rule is silent (closest to the existing tools):
- Columns that are identically zero in Part A (other families' descriptor
  columns of the schema) stay in every rung except R5, as they do in R0, so the
  N rungs differ from R0 only in the coupling values. A constant column is
  inert: scikit-learn's splitter does not count a constant feature toward
  ``max_features``, and the MLP sees a standardized zero with a zero gradient.
- Rung builders look descriptors up by ``item_id``, because the prediction
  rows the campaign path predicts on (``_prediction_items``) carry no
  circuit_id; the tables are built from the cached full rows.
- Arm A is the campaign's ``feat-only`` path (``_RidgeRunnerMethod``: ridge at
  each alpha of the frozen grid on standardized columns, source-validation
  one-standard-error selection) over the rung's columns without the noisy
  estimate, fitted once per dataset and rung.
- P uses shuffle seed = learner seed, as Appendix M.4 did.
- The Release section asks for per-candidate predictions of the existing
  Appendix M.4 and Appendix N fits; both stored only the selected candidate's.
  The re-export rungs refit every existing fit, the reproduction checks
  included (M.4's original-seed F, C, P; Appendix N's anchor seed), through
  the same hook, and check-reexport requires each to equal the stored fit bit
  for bit, so the re-export adds per-candidate predictions without replacing
  the analyzed fits. A re-export that differs is reported (exit code 4) and is
  not silently used in place of the stored fit.
- The s211 regenerated dataset differs from the archive in two validation
  labels by 1e-12, so verification uses ``--label-tolerance 1e-11`` exactly
  as the Appendix M.4 run did.

Usage
-----
  PYTHONPATH=. python tools/descriptor_ladder.py gate --datasets D101 D211 D307 \
      --archive ARCHIVE --out OUT [--seedrep-fits M4/fits] [--workers 6] \
      [--criterion archive|m4-refit-identity --tolerance X --amendment TEXT]
  PYTHONPATH=. python tools/descriptor_ladder.py prepare --datasets ... \
      --archive ARCHIVE --nc-datasets NC101 NC211 NC307 --out OUT [--workers 8]
  PYTHONPATH=. python tools/descriptor_ladder.py run --datasets ... \
      --archive ARCHIVE --nc-datasets ... --nc-results NCRES \
      --seedrep-fits M4/fits --out OUT --workers 8 [--dry-run] \
      [--rungs R0 N1 N2 ...] [--arms A F C P M] [--seeds 1-20]
  PYTHONPATH=. python tools/descriptor_ladder.py check-reexport --datasets ... \
      --nc-datasets ... --nc-results NCRES --seedrep-fits M4/fits --out OUT
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import functools
import gzip
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path[:1]:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools import descriptor_common as common  # noqa: E402
from tools.descriptor_common import (  # noqa: E402
    FEATURES,
    KEPT,
    LEARNER_SEEDS,
    RULE_SEED,
    STRENGTH_FEATURE_NAME,
    load_cache,
    severity_indicator,
    write_cache,
    write_json,
)
from qemscore.campaign.design import TRAINING_SHUFFLE_SEED  # noqa: E402
from qemscore.datasets.schema import build_features  # noqa: E402
from qemscore.runner.run import _prediction_items  # noqa: E402
from qemscore.validation import validate_split_artifact  # noqa: E402

PART = "A"
NOISE_LEVELS = {"N1": 0.01, "N2": 0.03, "N3": 0.10, "N4": 0.30}
COUPLINGS = {"tfi": ("j", "h"), "heisenberg": ("jx", "jy", "jz")}
ALL_COUPLINGS = ("j", "h", "jx", "jy", "jz")
SPIN_BLOCK = ("steps", "j", "h", "jx", "jy", "jz", "dt")
RUNGS = ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
NC_RUNG = "NC-none"
NC_REEXPORT_RUNG = "NC-R0"
NC_RUNGS = (NC_RUNG, NC_REEXPORT_RUNG)
# Rungs whose fits re-export existing fits (Appendix M.4 and Appendix N) with
# per-candidate predictions; they carry no arm A.
REEXPORT_RUNGS = ("R0", NC_REEXPORT_RUNG)
DEFAULT_RUNGS = (*RUNGS, *NC_RUNGS)
ARMS = ("A", "F", "C", "P", "M")
PRIMARY_KEY = re.compile(r"(shipped-s(\d+)-n(\d+))")
NC_KEY = re.compile(r"(nc-s(\d+)-n(\d+))")
ARCHIVE_METHODS = {"F": "liao", "C": "liao-feat-only"}
AFFINE_METHOD = "feat-only"
FAMILIES = ("tfi", "heisenberg")


# --------------------------------------------------------------------------
# Keys and datasets
# --------------------------------------------------------------------------


def _key(path: Path, pattern: re.Pattern) -> tuple[str, int]:
    match = pattern.search(path.name)
    if match is None:
        raise SystemExit(f"{path}: the directory name does not carry a setting key")
    return match.group(1), int(match.group(2))


def _cache_path(out: Path, key: str) -> Path:
    return out / "cache" / f"{key}.pkl"


def prepare_fresh_dataset(data_dir: Path, cache_dir: Path,
                          label_tolerance: float | None = None,
                          sweep: bool = False,
                          fresh_source: Path | None = None) -> dict:
    """Verify one fresh dataset with validate_split_artifact and structural assertions."""
    from qemscore.campaign.analysis import _assert_setting
    from qemscore.campaign.design import FAMILIES, setting_key
    from qemscore.validation import item_stream_hash, validate_split_artifact

    data_dir = Path(data_dir).resolve()
    roles_tuple = ("train", "validation", "test")

    if sweep:
        if fresh_source is None:
            raise SystemExit("--sweep with --fresh requires --fresh-source DIR")
        fresh_source = Path(fresh_source).resolve()

        rows = [
            json.loads(line)
            for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        actual_items_hash = item_stream_hash(rows)
        if manifest.get("items_hash") != actual_items_hash:
            raise ValueError(f"{data_dir}: items_hash mismatch in sweep dataset")
    else:
        rows, manifest = validate_split_artifact(data_dir)

    seed = int(manifest["master_seed"])
    regime = "shipped"

    key_match = PRIMARY_KEY.search(data_dir.name)
    if key_match is None:
        raise SystemExit(f"{data_dir}: directory name does not name a shipped-s<seed>-n<size> key")
    key = key_match.group(1)
    size = int(key_match.group(3))
    if int(key_match.group(2)) != seed:
        raise SystemExit(f"{data_dir}: directory names seed {key_match.group(2)}, manifest {seed}")

    roles = {role: [row for row in rows if row["split"] == role] for role in roles_tuple}

    expected = {
        role: count // len(FAMILIES)
        for role, count in manifest["split_spec"]["role_counts"].items()
    }
    structure = _assert_setting(regime, seed, size, rows, expected=expected)

    checks = {
        "setting": key,
        "data_dir": str(data_dir),
        "items_jsonl_sha256": common.sha256_file(data_dir / "items.jsonl"),
        "manifest_dataset_hash": str(manifest["dataset_hash"]),
        "validate_split_artifact": "passed" if not sweep else "sweep_passed",
        "structure_counts": structure["counts"],
        "n_items": {role: len(roles[role]) for role in roles_tuple},
        "exact_match": True,
        "fresh": True,
    }

    if sweep:
        checks["sweep"] = True
        # The 2,048-shot source tree holds regen-<key> for this dataset seed.
        src_dataset = fresh_source / f"regen-{key}"
        if not (src_dataset / "items.jsonl").exists():
            raise SystemExit(f"{key}: {src_dataset} is not a fresh 2,048-shot source dataset")
        src_rows = [
            json.loads(line)
            for line in (src_dataset / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        src_roles = {role: [row for row in src_rows if row["split"] == role] for role in roles_tuple}

        arch_test_cids = [it["circuit_id"] for it in src_roles["test"]]
        regen_test_cids = [it["circuit_id"] for it in roles["test"]]
        if arch_test_cids != regen_test_cids:
            raise SystemExit(f"{key}: sweep test circuit IDs differ from fresh source {src_dataset}")

        arch_val_cids = [it["circuit_id"] for it in src_roles["validation"]]
        regen_val_cids = [it["circuit_id"] for it in roles["validation"]]
        if arch_val_cids != regen_val_cids:
            raise SystemExit(f"{key}: sweep validation circuit IDs differ from fresh source {src_dataset}")

        tol = 1e-11 if label_tolerance is None else label_tolerance
        for role in ("test", "validation"):
            if len(roles[role]) != len(src_roles[role]):
                raise SystemExit(f"{key}: sweep {role} rows {len(roles[role])} differ from fresh source {len(src_roles[role])}")
            max_diff = max(abs(float(r["ideal_expectation"]) - float(s["ideal_expectation"]))
                           for r, s in zip(roles[role], src_roles[role]))
            checks[f"sweep_{role}_label_max_abs_diff"] = max_diff
            if max_diff > tol:
                raise SystemExit(f"{key}: {role} ideal expectation diff {max_diff} > {tol}")

    payload = {
        "key": key,
        "seed": seed,
        "train": roles["train"],
        "validation": roles["validation"],
        "test": roles["test"],
        "prediction_rows": {
            role: _prediction_items(roles[role]) for role in ("validation", "test")
        },
        "dataset_hash": str(manifest["dataset_hash"]),
    }
    cache_path = cache_dir / f"{key}.pkl"
    tmp = cache_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    order = {
        "test_item_ids": [str(row["item_id"]) for row in roles["test"]],
        "validation_item_ids": [str(row["item_id"]) for row in roles["validation"]],
    }
    (cache_dir / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
    return checks


def prepare_primary(data_dirs, archive: Path | None, out: Path,
                    label_tolerance: float | None,
                    sweep: bool = False,
                    fresh: bool = False,
                    fresh_source: Path | None = None) -> dict:
    """Verify each regenerated dataset and cache its rows."""
    from tools.learner_seed_replication import prepare_dataset

    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for data_dir in data_dirs:
        started = time.perf_counter()
        if fresh:
            check = prepare_fresh_dataset(Path(data_dir).resolve(), cache_dir,
                                          label_tolerance=label_tolerance,
                                          sweep=sweep,
                                          fresh_source=fresh_source)
        else:
            if archive is None:
                raise SystemExit("--archive is required when not in --fresh mode")
            check = prepare_dataset(Path(data_dir).resolve(), archive, cache_dir,
                                    label_tolerance=label_tolerance,
                                    sweep=sweep)
        check["load_and_verify_seconds"] = time.perf_counter() - started
        checks[check["setting"]] = check
        print(f"verified {check['setting']} in {check['load_and_verify_seconds']:.1f}s "
              f"(exact={check['exact_match']})", flush=True)
    return checks


def prepare_nc(data_dirs, out: Path, nc_results: Path | None) -> dict:
    """Validate each near-Clifford dataset and cache its rows."""
    checks = {}
    for data_dir in data_dirs:
        data_dir = Path(data_dir).resolve()
        key, seed = _key(data_dir, NC_KEY)
        rows, manifest = validate_split_artifact(data_dir)
        if int(manifest["master_seed"]) != seed:
            raise SystemExit(f"{data_dir}: master seed {manifest['master_seed']} != {seed}")
        check = {"setting": key, "data_dir": str(data_dir),
                 "dataset_hash": str(manifest["dataset_hash"]),
                 "items_jsonl_sha256": common.sha256_file(data_dir / "items.jsonl"),
                 "validate_split_artifact": "passed"}
        if nc_results is not None:
            summary_path = nc_results / f"s{seed}" / "summary.json"
            recorded = json.loads(summary_path.read_text(encoding="utf-8"))["dataset_hash"]
            check["appendix_n_dataset_hash"] = recorded
            check["matches_appendix_n"] = recorded == check["dataset_hash"]
            if not check["matches_appendix_n"]:
                raise SystemExit(f"{key}: dataset hash differs from Appendix N's record")
        roles = {role: [row for row in rows if row["split"] == role]
                 for role in ("train", "validation", "test")}
        check["n_items"] = {role: len(value) for role, value in roles.items()}
        payload = {
            "key": key, "seed": seed, **roles,
            "prediction_rows": {role: _prediction_items(roles[role])
                                for role in ("validation", "test")},
            "dataset_hash": check["dataset_hash"],
        }
        write_cache(_cache_path(out, key), payload)
        order = {f"{role}_item_ids": [str(row["item_id"]) for row in roles[role]]
                 for role in ("validation", "test")}
        (out / "cache" / f"{key}.order.json").write_text(json.dumps(order),
                                                        encoding="utf-8")
        checks[key] = check
        print(f"validated {key} ({check['n_items']})", flush=True)
    return checks


def _all_rows(data: dict) -> list[dict]:
    return [*data["train"], *data["validation"], *data["test"]]


# --------------------------------------------------------------------------
# Rung descriptor tables
# --------------------------------------------------------------------------


def coupling_noise_table(data: dict) -> dict:
    """z per circuit, from the rule's SeedSequence, keyed by circuit and by item."""
    rows = _all_rows(data)
    family_by_circuit: dict[str, str] = {}
    for row in rows:
        circuit = str(row["circuit_id"])
        if family_by_circuit.setdefault(circuit, str(row["family"])) != row["family"]:
            raise ValueError(f"{circuit} appears under two families")
    circuits = sorted(family_by_circuit)
    z_by_circuit = {}
    for index, circuit in enumerate(circuits):
        family = family_by_circuit[circuit]
        rng = np.random.default_rng(
            np.random.SeedSequence([RULE_SEED, int(data["seed"]), index]))
        z_by_circuit[circuit] = tuple(
            float(value) for value in rng.standard_normal(len(COUPLINGS[family])))
    return {
        "circuits": [{"circuit_index": index, "circuit_id": circuit,
                      "family": family_by_circuit[circuit],
                      "couplings": list(COUPLINGS[family_by_circuit[circuit]]),
                      "z": list(z_by_circuit[circuit])}
                     for index, circuit in enumerate(circuits)],
        "z_by_item": {str(row["item_id"]): z_by_circuit[str(row["circuit_id"])]
                      for row in rows},
    }


def _encoder_paths(out: Path, key: str) -> tuple[Path, Path]:
    return out / "encoder-cache" / f"{key}.npz", out / "encoder-cache" / f"{key}.json"


def _encode_rows(task: tuple[list[dict], str]) -> list[dict]:
    """Encode one chunk of circuits and check them against the recorded structure."""
    from tools import mlqem_binned_angle_encoder as encoder

    rows, data_dir = task
    results = []
    for row in rows:
        transpiled = encoder.rebuild_and_transpile_circuit(row, dataset_dir=data_dir)
        counts = transpiled.count_ops()
        gates = [float(counts.get(name, 0)) for name in encoder.GATES_SET]
        angles = [float(value) for value in
                  encoder.count_gates_by_rotation_angle(transpiled, encoder.BIN_SIZE)]
        vector = [value * 0.01 for value in gates + angles]
        results.append({
            "circuit_id": str(row["circuit_id"]),
            "vector": vector,
            "cx": int(counts.get("cx", 0)),
            "depth": int(transpiled.depth()),
            "structure_matches": (int(counts.get("cx", 0)) == int(row["two_qubit_gates"])
                                  and int(transpiled.depth())
                                  == int(row["transpiled_depth"])),
        })
    return results


def prepare_encoding(key: str, data: dict, data_dir: Path, out: Path,
                     workers: int) -> dict:
    """One released ML-QEM vector per circuit, cached to disk."""
    from tools import mlqem_binned_angle_encoder as encoder

    npz_path, json_path = _encoder_paths(out, key)
    if npz_path.exists() and json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    first: dict[str, dict] = {}
    for row in _all_rows(data):
        first.setdefault(str(row["circuit_id"]), row)
    circuits = sorted(first)
    fields = ("circuit_id", "circuit_sidecar", "family", "n_qubits", "circuit_seed",
              "two_qubit_gates", "transpiled_depth")
    slim = [{field: first[c][field] for field in fields} for c in circuits]
    chunks = [slim[i:i + 64] for i in range(0, len(slim), 64)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        encoded = [item for part in pool.map(_encode_rows,
                                             [(chunk, str(data_dir)) for chunk in chunks])
                   for item in part]
    by_circuit = {entry["circuit_id"]: entry for entry in encoded}
    matrix = np.asarray([by_circuit[c]["vector"] for c in circuits], dtype=float)
    mismatches = [c for c in circuits if not by_circuit[c]["structure_matches"]]
    # The verbatim released entry point must agree with the cached vector.
    spot = {}
    for family in FAMILIES:
        circuit = next(c for c in circuits if first[c]["family"] == family)
        released = encoder.encode_item_circuit(first[circuit], dataset_dir=data_dir)
        spot[family] = bool(np.array_equal(np.asarray(released, dtype=float),
                                           matrix[circuits.index(circuit)]))
    meta = {
        "key": key,
        "data_dir": str(data_dir),
        "n_circuits": len(circuits),
        "feature_names": list(encoder.FEATURE_NAMES),
        "scaling": 0.01,
        "upstream_commit": encoder.UPSTREAM_COMMIT,
        "upstream_mlp_sha256": encoder.UPSTREAM_MLP_SHA256,
        "lifted_file_sha256": encoder.LIFTED_FILE_SHA256,
        "structure_mismatches": mismatches,
        "released_entry_point_agrees": spot,
        "seconds": time.perf_counter() - started,
    }
    if mismatches or not all(spot.values()):
        raise SystemExit(f"{key}: encoder check failed: {json.dumps(meta)[:2000]}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(tmp, circuit_ids=np.asarray(circuits), vectors=matrix)
    tmp.replace(npz_path)
    write_json(json_path, meta)
    return meta


def _load_encoding(out: Path, key: str) -> tuple[dict[str, np.ndarray], list[str]]:
    npz_path, json_path = _encoder_paths(out, key)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    vectors = {str(c): arrays["vectors"][i] for i, c in enumerate(arrays["circuit_ids"])}
    return vectors, list(meta["feature_names"])


def r4_diagnostics(key: str, data: dict, out: Path) -> dict:
    """Label-free: distinct R4 vectors among training circuits, and kNN R^2."""
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor

    vectors, _ = _load_encoding(out, key)
    report = {}
    for family in FAMILIES:
        per_role = {}
        for role in ("train", "test"):
            seen: dict[str, dict] = {}
            for row in data[role]:
                if row["family"] == family:
                    seen.setdefault(str(row["circuit_id"]), row)
            per_role[role] = [seen[c] for c in sorted(seen)]
        x_train = np.asarray([vectors[str(r["circuit_id"])] for r in per_role["train"]])
        x_test = np.asarray([vectors[str(r["circuit_id"])] for r in per_role["test"]])
        entry = {
            "n_train_circuits": len(per_role["train"]),
            "n_distinct_r4_vectors_train": int(np.unique(x_train, axis=0).shape[0]),
            "n_test_circuits": len(per_role["test"]),
            "knn_r2_test": {},
            "note": ("kNN regression of each coupling on the R4 encoding block, fitted "
                     "on training circuits and scored on test circuits (Euclidean, "
                     "uniform weights); couplings are inputs, so no label is read"),
        }
        for k in (1, 5):
            entry["knn_r2_test"][f"k{k}"] = {}
            for coupling in COUPLINGS[family]:
                y_train = np.asarray([float(r[coupling]) for r in per_role["train"]])
                y_test = np.asarray([float(r[coupling]) for r in per_role["test"]])
                model = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train)
                entry["knn_r2_test"][f"k{k}"][coupling] = float(
                    r2_score(y_test, model.predict(x_test)))
        report[family] = entry
    return report


# --------------------------------------------------------------------------
# Rung builders
# --------------------------------------------------------------------------


class RungBuilder:
    """Feature builder for one Part A rung; R0 is ``build_features`` itself."""

    def __init__(self, rung: str, *, z_by_item: dict | None = None,
                 encoding_by_circuit: dict | None = None,
                 circuit_by_item: dict | None = None,
                 encoder_names: list[str] | None = None,
                 strength_indicator: bool = False) -> None:
        if rung not in RUNGS and rung != NC_RUNG:
            raise ValueError(f"unknown rung {rung!r}")
        self.rung = rung
        self.strength_indicator = bool(strength_indicator)
        self.name = f"descriptor-ladder:{rung}"
        self._z = z_by_item
        self._encoding = encoding_by_circuit
        self._circuit = circuit_by_item
        if rung in NOISE_LEVELS:
            if z_by_item is None:
                raise ValueError("an N rung needs the coupling-noise table")
            self.s = NOISE_LEVELS[rung]
            names = [f"{n}+{self.s:g}z" if n in ALL_COUPLINGS else n for n in FEATURES]
        elif rung == "R0":
            names = list(FEATURES)
        elif rung == "R3-TFI":
            names = [n for n in FEATURES if n != "h"]
        elif rung == "R3-Heis":
            names = [n for n in FEATURES if n != "jz"]
        elif rung == "R4":
            if encoding_by_circuit is None or circuit_by_item is None or not encoder_names:
                raise ValueError("R4 needs the encoder cache")
            names = []
            for n in FEATURES:
                if n == SPIN_BLOCK[0]:
                    names.extend(f"mlqem:{e}" for e in encoder_names)
                if n not in SPIN_BLOCK:
                    names.append(n)
        else:  # R5 and NC-none
            names = list(KEPT)
        if self.strength_indicator:
            names.append(STRENGTH_FEATURE_NAME)
        self.names = tuple(names)
        self._drop_index = {
            "R3-TFI": FEATURES.index("h"), "R3-Heis": FEATURES.index("jz")}.get(rung)

    def __call__(self, item: dict) -> list[float]:
        base = build_features(item)
        rung = self.rung
        if rung == "R0":
            res = base
        elif rung in NOISE_LEVELS:
            family = str(item["family"])
            values = list(base)
            if family in COUPLINGS:
                z = self._z[str(item["item_id"])]
                for coupling, draw in zip(COUPLINGS[family], z, strict=True):
                    index = FEATURES.index(coupling)
                    values[index] = values[index] + self.s * draw
            res = values
        elif self._drop_index is not None:
            res = base[:self._drop_index] + base[self._drop_index + 1:]
        elif rung == "R4":
            encoded = self._encoding[self._circuit[str(item["item_id"])]]
            values = dict(zip(FEATURES, base))
            row: list[float] = []
            for n in FEATURES:
                if n == SPIN_BLOCK[0]:
                    row.extend(float(v) for v in encoded)
                if n not in SPIN_BLOCK:
                    row.append(values[n])
            res = row
        else:
            values = dict(zip(FEATURES, base))
            res = [values[n] for n in KEPT]
        if self.strength_indicator:
            return [*res, severity_indicator(item)]
        return res


_BUILDERS: dict[tuple, tuple] = {}


def builder_for(out: Path, key: str, rung: str, data: dict, *,
                strength_indicator: bool = False):
    """The rung's (builder, names), built once per process."""
    cache_key = (str(out), key, rung, strength_indicator)
    if cache_key in _BUILDERS:
        return _BUILDERS[cache_key]
    if rung == "R0":
        if not strength_indicator:
            built = (build_features, tuple(FEATURES))
        else:
            builder = RungBuilder("R0", strength_indicator=True)
            built = (builder, builder.names)
    elif rung in NOISE_LEVELS:
        table = coupling_noise_table(data)
        builder = RungBuilder(rung, z_by_item=table["z_by_item"],
                              strength_indicator=strength_indicator)
        built = (builder, builder.names)
    elif rung == "R4":
        vectors, names = _load_encoding(out, key)
        circuit_by_item = {str(r["item_id"]): str(r["circuit_id"]) for r in _all_rows(data)}
        builder = RungBuilder(rung, encoding_by_circuit=vectors,
                              circuit_by_item=circuit_by_item, encoder_names=names,
                              strength_indicator=strength_indicator)
        built = (builder, builder.names)
    else:
        builder = RungBuilder(rung, strength_indicator=strength_indicator)
        built = (builder, builder.names)
    _BUILDERS[cache_key] = built
    return built


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


def run_job(job: dict) -> dict:
    """Worker entry point: one fit."""
    out = Path(job["out"])
    data = load_cache(job["cache"])
    strength = bool(job.get("strength_indicator", False))
    builder, names = builder_for(out, job["key"], job["rung_builder"], data,
                                 strength_indicator=strength)
    if job["kind"] == "affine":
        return common.run_affine_fit(job, data, builder, names)
    return common.run_liao_fit(job, data, builder, names)


def _job(out: Path, fit_dir: Path, key: str, rung: str, arm: str, *,
         learner_seed: int | None = None, shuffle_seed: int | None = None,
         purpose: str = "rung_fit", stem: str | None = None,
         strength_indicator: bool = False,
         strong_learners: bool = False,
         train_size: int | None = None,
         neural_es: bool = False) -> dict:
    fit_arm = {"M": "F"}.get(arm, arm)
    builder_rung = {NC_RUNG: "R5", NC_REEXPORT_RUNG: "R0"}.get(rung, rung)
    if stem is None:
        seed_part = "" if learner_seed is None else f"__k{learner_seed:02d}"
        stem = f"{key}__{rung}{seed_part}__{arm}"
    if fit_arm == "P" and shuffle_seed is None:
        shuffle_seed = learner_seed
    job_dict = {
        "part": PART, "kind": "affine" if arm == "A" else "liao",
        "key": key, "rung": rung, "rung_builder": builder_rung, "arm": arm,
        "fit_arm": fit_arm, "learner_seed": learner_seed,
        "shuffle_seed": shuffle_seed if fit_arm == "P" else None,
        "purpose": purpose, "stem": stem,
        "cache": str(_cache_path(out, key)), "out": str(out), "fit_dir": str(fit_dir),
    }
    if strength_indicator:
        job_dict["strength_indicator"] = True
    if strong_learners:
        job_dict["strong_learners"] = True
    if train_size is not None:
        job_dict["train_size"] = int(train_size)
    if neural_es:
        job_dict["neural_es"] = True
    return job_dict


def _dataset_seed(key: str, pattern: re.Pattern) -> int:
    return _key(Path(key), pattern)[1]


def build_jobs(out: Path, primary_keys: list[str], nc_keys: list[str],
               rungs: list[str], arms: list[str], seeds: list[int],
               strength_indicator: bool = False,
               strong_learners: bool = False,
               train_size: int | None = None,
               neural_es: bool = False,
               fresh: bool = False) -> list[dict]:
    """Every planned fit; the re-export rungs come first so a mismatch shows early."""
    fit_dir = out / "fits"
    jobs = []
    liao_arms = [arm for arm in ("F", "C", "P") if arm in arms]
    if not strength_indicator and train_size is None and not fresh:
        if "R0" in rungs:
            # Appendix M.4 re-export: seeds 1-20, then M.4's original-seed P
            # (archived shuffle seed). M.4's original-seed F and C are the gate fits.
            for k in seeds:
                for key in primary_keys:
                    for arm in liao_arms:
                        jobs.append(_job(out, fit_dir, key, "R0", arm, learner_seed=k,
                                         purpose="r0_reexport_appendix_m4",
                                         strong_learners=strong_learners,
                                         neural_es=neural_es))
            if "P" in arms:
                for key in primary_keys:
                    jobs.append(_job(
                        out, fit_dir, key, "R0", "P",
                        learner_seed=_dataset_seed(key, PRIMARY_KEY),
                        shuffle_seed=TRAINING_SHUFFLE_SEED,
                        purpose="r0_reexport_appendix_m4_original_seed_archived_shuffle",
                        stem=f"{key}__R0__orig__P",
                        strong_learners=strong_learners,
                        neural_es=neural_es))
        if NC_REEXPORT_RUNG in rungs:
            # Appendix N re-export: seeds 1-20 plus its anchor (learner seed = dataset seed).
            for key in nc_keys:
                anchor = _dataset_seed(key, NC_KEY)
                for k in [*seeds, *([anchor] if anchor not in seeds else [])]:
                    for arm in liao_arms:
                        jobs.append(_job(
                            out, fit_dir, key, NC_REEXPORT_RUNG, arm, learner_seed=k,
                            purpose=("nc_reexport_appendix_n_anchor" if k == anchor
                                     and anchor not in seeds else "nc_reexport_appendix_n"),
                            strong_learners=strong_learners,
                            neural_es=neural_es))
        spin_rungs = [r for r in rungs if r in RUNGS and r not in REEXPORT_RUNGS]
        if "A" in arms:
            for rung in spin_rungs:
                for key in primary_keys:
                    jobs.append(_job(out, fit_dir, key, rung, "A",
                                     strong_learners=strong_learners,
                                     neural_es=neural_es))
        for k in seeds:
            for rung in spin_rungs:
                for key in primary_keys:
                    for arm in liao_arms:
                        jobs.append(_job(out, fit_dir, key, rung, arm, learner_seed=k,
                                         strong_learners=strong_learners,
                                         neural_es=neural_es))
            if NC_RUNG in rungs:
                for key in nc_keys:
                    for arm in ("M", "C"):
                        if arm in arms:
                            jobs.append(_job(out, fit_dir, key, NC_RUNG, arm,
                                             learner_seed=k,
                                             purpose="near_clifford_no_descriptors",
                                             strong_learners=strong_learners,
                                             neural_es=neural_es))
    else:
        part_a_rungs = [r for r in rungs if r in RUNGS]
        if "A" in arms:
            for rung in part_a_rungs:
                for key in primary_keys:
                    jobs.append(_job(out, fit_dir, key, rung, "A",
                                     strength_indicator=strength_indicator,
                                     strong_learners=strong_learners,
                                     train_size=train_size,
                                     neural_es=neural_es))
        for k in seeds:
            for rung in part_a_rungs:
                for key in primary_keys:
                    for arm in liao_arms:
                        jobs.append(_job(out, fit_dir, key, rung, arm,
                                         learner_seed=k,
                                         strength_indicator=strength_indicator,
                                         strong_learners=strong_learners,
                                         train_size=train_size,
                                         neural_es=neural_es))
        if NC_REEXPORT_RUNG in rungs:
            if "A" in arms:
                for key in nc_keys:
                    jobs.append(_job(out, fit_dir, key, NC_REEXPORT_RUNG, "A",
                                     strength_indicator=strength_indicator,
                                     strong_learners=strong_learners,
                                     train_size=train_size,
                                     neural_es=neural_es))
            for k in seeds:
                for key in nc_keys:
                    for arm in liao_arms:
                        jobs.append(_job(out, fit_dir, key, NC_REEXPORT_RUNG, arm,
                                         learner_seed=k,
                                         strength_indicator=strength_indicator,
                                         strong_learners=strong_learners,
                                         train_size=train_size,
                                         neural_es=neural_es))
        if NC_RUNG in rungs:
            for k in seeds:
                for key in nc_keys:
                    for arm in ("M", "C"):
                        if arm in arms:
                            jobs.append(_job(out, fit_dir, key, NC_RUNG, arm,
                                             learner_seed=k,
                                             purpose="near_clifford_no_descriptors",
                                             strength_indicator=strength_indicator,
                                             strong_learners=strong_learners,
                                             train_size=train_size,
                                             neural_es=neural_es))
    return jobs


# --------------------------------------------------------------------------
# Re-export check (no fit)
# --------------------------------------------------------------------------


M4_GROUP = "R0_vs_appendix_M4"
NC_GROUP = "NC-R0_vs_appendix_N"


@functools.lru_cache(maxsize=8)
def _appendix_n(nc_results: str, seed: int) -> tuple[dict, dict]:
    base = Path(nc_results) / f"s{seed}"
    with gzip.open(base / "test_predictions.json.gz", "rt", encoding="utf-8") as handle:
        predictions = json.load(handle)
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    return predictions, summary


def _appendix_n_selected(summary: dict) -> dict[tuple[int, str], str]:
    entries = [*summary.get("per_seed_results", [])]
    if summary.get("anchor_fit"):
        entries.append(summary["anchor_fit"])
    return {(int(entry["seed"]), arm): entry["selected_candidates"][arm]
            for entry in entries for arm in ("F", "C", "P")}


def _order(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reexport_preconditions(out: Path, primary_keys: list[str], nc_keys: list[str],
                           seedrep_fits: Path | None, nc_results: Path | None) -> dict:
    """Row order of the cached rows against each reference; checked before fitting."""
    report: dict = {M4_GROUP: {}, NC_GROUP: {}}
    if seedrep_fits is not None:
        reference_cache = Path(seedrep_fits).parent / "cache"
        for key in primary_keys:
            report[M4_GROUP][key] = (
                _order(out / "cache" / f"{key}.order.json")
                == _order(reference_cache / f"{key}.order.json"))
    if nc_results is not None:
        for key in nc_keys:
            predictions, _ = _appendix_n(str(nc_results), _dataset_seed(key, NC_KEY))
            report[NC_GROUP][key] = (
                _order(out / "cache" / f"{key}.order.json")["test_item_ids"]
                == predictions["test_item_ids"])
    report["all_match"] = all(value for group in (M4_GROUP, NC_GROUP)
                              for value in report[group].values())
    return report


def _compare(new: np.ndarray, reference: np.ndarray) -> dict:
    new = np.asarray(new, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if new.shape != reference.shape:
        return {"exact": False, "max_abs_diff": None,
                "shape": [list(new.shape), list(reference.shape)]}
    return {"exact": bool(np.array_equal(new, reference)),
            "max_abs_diff": float(np.max(np.abs(new - reference))) if new.size else 0.0}


def _reexport_entry(group: str, stem: str, fit_dir: Path, reference: str,
                    reference_arrays: dict | None, reference_meta: dict) -> dict:
    entry = {"group": group, "stem": stem, "fit_dir": str(fit_dir),
             "reference": reference}
    meta_path = fit_dir / f"{stem}.json"
    if reference_arrays is None:
        entry["status"] = "no_reference"
        return entry
    if not meta_path.exists():
        entry["status"] = "missing"
        return entry
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("error"):
        entry.update(status="error", error=meta["error"])
        return entry
    arrays = np.load(fit_dir / f"{stem}.npz")
    entry["roles"] = {role: _compare(arrays[role], reference_arrays[role])
                      for role in ("test", "validation") if role in reference_arrays}
    entry["learner_seed"] = meta["learner_seed"]
    entry["shuffle_seed"] = meta.get("shuffle_seed")
    entry["seeds_match"] = bool(
        meta["learner_seed"] == reference_meta.get("learner_seed")
        and meta.get("shuffle_seed") == reference_meta.get("shuffle_seed"))
    entry["selected_model"] = meta["selected_model"]
    entry["reference_selected_model"] = reference_meta.get("selected_model")
    entry["selected_model_matches"] = (
        None if entry["reference_selected_model"] is None
        else meta["selected_model"] == entry["reference_selected_model"])
    entry["has_candidate_predictions"] = all(
        f"{role}__{name}" in arrays.files
        for role in ("validation", "test") for name in common.CANDIDATES)
    if "flagged_non_converged" in reference_meta:
        entry["flag_matches_reference"] = (
            meta["flagged_non_converged"] == reference_meta["flagged_non_converged"])
    exact = (all(role["exact"] for role in entry["roles"].values())
             and entry["seeds_match"] and entry["selected_model_matches"] is not False
             and entry["has_candidate_predictions"])
    entry["status"] = "exact" if exact else "differs"
    return entry


def check_reexports(out: Path, primary_keys: list[str], nc_keys: list[str],
                    seeds: list[int], seedrep_fits: Path | None,
                    nc_results: Path | None) -> dict:
    """Compare each re-exported fit with the fit it re-exports; no fit."""
    entries: list[dict] = []
    if seedrep_fits is not None:
        seedrep_fits = Path(seedrep_fits)
        for key in primary_keys:
            specs = [(f"{key}__R0__k{k:02d}__{arm}", out / "fits", f"{key}__k{k:02d}__{arm}")
                     for k in seeds for arm in ("F", "C", "P")]
            specs += [(f"{key}__R0__orig__F", out / "gate" / "fits", f"{key}__orig__F"),
                      (f"{key}__R0__orig__C", out / "gate" / "fits", f"{key}__orig__C"),
                      (f"{key}__R0__orig__P", out / "fits", f"{key}__orig__P")]
            for stem, fit_dir, reference in specs:
                ref_json = seedrep_fits / f"{reference}.json"
                if ref_json.exists():
                    ref_meta = json.loads(ref_json.read_text(encoding="utf-8"))
                    ref_arrays = dict(np.load(seedrep_fits / f"{reference}.npz"))
                else:
                    ref_meta, ref_arrays = {}, None
                entries.append(_reexport_entry(M4_GROUP, stem, fit_dir, reference,
                                               ref_arrays, ref_meta))
    if nc_results is not None:
        for key in nc_keys:
            anchor = _dataset_seed(key, NC_KEY)
            predictions, summary = _appendix_n(str(nc_results), anchor)
            selected = _appendix_n_selected(summary)
            order = _order(out / "cache" / f"{key}.order.json")["test_item_ids"]
            for k in [*seeds, *([anchor] if anchor not in seeds else [])]:
                for arm in ("F", "C", "P"):
                    stem = f"{key}__{NC_REEXPORT_RUNG}__k{k:02d}__{arm}"
                    by_item = predictions["seeds"].get(str(k), {}).get(arm)
                    ref_arrays = (None if by_item is None else
                                  {"test": np.asarray([by_item[i] for i in order],
                                                      dtype=float)})
                    ref_meta = {"learner_seed": k,
                                "shuffle_seed": k if arm == "P" else None,
                                "selected_model": selected.get((k, arm))}
                    entries.append(_reexport_entry(
                        NC_GROUP, stem, out / "fits",
                        f"appendix-n s{anchor} seed {k} arm {arm}", ref_arrays, ref_meta))
    groups = {}
    for group in (M4_GROUP, NC_GROUP):
        members = [entry for entry in entries if entry["group"] == group]
        counts = Counter(entry["status"] for entry in members)
        diffs = [role["max_abs_diff"] for entry in members
                 for role in entry.get("roles", {}).values()
                 if role.get("max_abs_diff") is not None]
        groups[group] = {
            "expected": len(members), "counts": dict(counts),
            # None when the group was not checked at all.
            "complete_and_exact": (None if not members
                                   else counts["exact"] == len(members)),
            "max_abs_diff_present": max(diffs) if diffs else None,
        }
    return {
        "schema": "descriptor-information-reexport-check-v1",
        "frozen_rule": common.RULE_FILE,
        "seeds": list(seeds),
        "criterion": ("a re-export is exact when its selected predictions equal the "
                      "reference's bit for bit (test and validation for Appendix M.4; "
                      "test for Appendix N, which stored only test predictions), its "
                      "learner and shuffle seeds and selected candidate match, and it "
                      "stores both candidates' predictions"),
        "row_order": reexport_preconditions(out, primary_keys, nc_keys, seedrep_fits,
                                            nc_results),
        "groups": groups,
        "r0_reuses_appendix_m4": groups[M4_GROUP]["complete_and_exact"],
        "nc_reexport_matches_appendix_n": groups[NC_GROUP]["complete_and_exact"],
        "any_differs": any(entry["status"] in ("differs", "error") for entry in entries),
        "entries": entries,
    }


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def _primary_keys(args) -> list[tuple[str, int]]:
    return [_key(Path(d), PRIMARY_KEY) for d in args.datasets]


def _nc_keys(args) -> list[tuple[str, int]]:
    return [_key(Path(d), NC_KEY) for d in (args.nc_datasets or [])]


def cmd_gate(args) -> int:
    # Before any work: the frozen gate, or an amendment already written into the rule.
    amendment = common.check_gate_amendment(
        criterion=args.criterion, tolerance=args.tolerance, amendment=args.amendment,
        seedrep_fits=args.seedrep_fits)
    out = args.out.resolve()
    gate_dir = out / "gate"
    fit_dir = gate_dir / "fits"
    fit_dir.mkdir(parents=True, exist_ok=True)
    checks = prepare_primary(args.datasets, args.archive, out, args.label_tolerance)
    keys = [(key, int(check["setting"].split("-s")[1].split("-")[0]))
            for key, check in checks.items()]
    jobs = []
    for key, seed in keys:
        for arm in ("F", "C"):
            jobs.append(_job(out, fit_dir, key, "R0", arm, learner_seed=seed,
                             purpose="r0_gate_original_seed",
                             stem=f"{key}__R0__orig__{arm}"))
        jobs.append(_job(out, fit_dir, key, "R0", "A", purpose="r0_gate_affine_check",
                         stem=f"{key}__R0__A"))
    run_info = common.drive(jobs, run_job, gate_dir, args.workers)

    report = {"schema": "descriptor-information-r0-gate-v2",
              "frozen_rule": common.RULE_FILE,
              "rule_file_sha256": common.sha256_file(common.rule_path()),
              "criterion_id": args.criterion,
              "criterion": common.GATE_CRITERIA[args.criterion],
              "tolerance": args.tolerance,
              "frozen_criterion": {"criterion_id": common.FROZEN_GATE_CRITERION,
                                   "criterion": common.GATE_CRITERIA[
                                       common.FROZEN_GATE_CRITERION],
                                   "tolerance": common.GATE_TOLERANCE},
              **amendment,
              "datasets": checks, "rows": {},
              "run": run_info, "blas_fpe_probe": common.blas_fpe_probe()}
    gate_max = 0.0
    affine_max = 0.0
    seedrep_max = 0.0 if args.seedrep_fits else None
    for key, seed in keys:
        record = json.loads((args.archive / "records" / f"{key}.json").read_text(
            encoding="utf-8"))
        data = load_cache(_cache_path(out, key))
        test_ids = [str(r["item_id"]) for r in data["test"]]
        val_ids = [str(r["item_id"]) for r in data["validation"]]
        row = {}
        for arm, method in ARCHIVE_METHODS.items():
            stem = f"{key}__R0__orig__{arm}"
            meta = json.loads((fit_dir / f"{stem}.json").read_text(encoding="utf-8"))
            arrays = np.load(fit_dir / f"{stem}.npz")
            archived_test = np.asarray([record["test_predictions"][method][i]
                                        for i in test_ids])
            archived_val = np.asarray([record["validation_predictions"][method][i]
                                       for i in val_ids])
            entry = {
                "method": method, "learner_seed": meta["learner_seed"],
                "feature_builder": meta["feature_builder"],
                "n_features": meta["n_features"],
                "selected_model": meta["selected_model"],
                "archived_selected_model":
                    record["selected_configs"][method]["selected_model"],
                "max_abs_diff_test": float(np.max(np.abs(arrays["test"] - archived_test))),
                "max_abs_diff_validation": float(
                    np.max(np.abs(arrays["validation"] - archived_val))),
                "flagged_non_converged": meta["flagged_non_converged"],
                "error": meta["error"],
            }
            gate_max = max(gate_max, entry["max_abs_diff_test"])
            if args.seedrep_fits:
                previous = np.load(Path(args.seedrep_fits) / f"{key}__orig__{arm}.npz")
                entry["max_abs_diff_test_vs_pre_hook_refit"] = float(
                    np.max(np.abs(arrays["test"] - previous["test"])))
                entry["max_abs_diff_validation_vs_pre_hook_refit"] = float(
                    np.max(np.abs(arrays["validation"] - previous["validation"])))
                seedrep_max = max(seedrep_max,
                                  entry["max_abs_diff_test_vs_pre_hook_refit"],
                                  entry["max_abs_diff_validation_vs_pre_hook_refit"])
            row[arm] = entry
        meta = json.loads((fit_dir / f"{key}__R0__A.json").read_text(encoding="utf-8"))
        arrays = np.load(fit_dir / f"{key}__R0__A.npz")
        archived = np.asarray([record["test_predictions"][AFFINE_METHOD][i]
                               for i in test_ids])
        archived_val = np.asarray([record["validation_predictions"][AFFINE_METHOD][i]
                                   for i in val_ids])
        row["A"] = {
            "method": AFFINE_METHOD,
            "max_abs_diff_test": float(np.max(np.abs(arrays["test"] - archived))),
            "max_abs_diff_validation": float(
                np.max(np.abs(arrays["validation"] - archived_val))),
            "selected_alpha": meta["selection"]["best_alpha"],
            "archived_selected_alpha": record["selected_configs"][AFFINE_METHOD].get(
                "best_alpha"),
            "error": meta["error"],
        }
        affine_max = max(affine_max, row["A"]["max_abs_diff_test"])
        report["rows"][key] = row
    report["max_abs_diff_test"] = gate_max
    report.update(common.gate_decision(
        criterion=args.criterion, tolerance=args.tolerance, archive_max=gate_max,
        refit_identity_max=seedrep_max, crashes=run_info["crashes"]))
    report["supplementary"] = {
        "affine_A_at_R0_max_abs_diff_test": affine_max,
        "hook_vs_pre_hook_refit_max_abs_diff": seedrep_max,
        "note": ("outside the pass criterion: arm A at R0 against the archived "
                 "feat-only control, and the hooked refits against the pre-hook "
                 "original-seed refits of Appendix M.4 on this machine"),
    }
    write_json(gate_dir / "gate_r0.json", report)
    print(json.dumps({"passed": report["passed"], "criterion_id": args.criterion,
                      "frozen_criterion_passed": report["frozen_criterion_passed"],
                      "amendment": report["amendment"], "max_abs_diff_test": gate_max,
                      "tolerance": args.tolerance, "supplementary":
                      report["supplementary"]}, indent=1))
    return 0 if report["passed"] else 2


def _write_dataset_index(out: Path, checks: dict, nc_checks: dict, fresh: bool = False) -> None:
    index = {key: {"kind": "primary", "data_dir": check["data_dir"],
                   "dataset_hash": check["manifest_dataset_hash"],
                   "cache": str(_cache_path(out, key))}
             for key, check in checks.items()}
    if fresh:
        for val in index.values():
            val["fresh"] = True
        index["_fresh"] = True
    index.update({key: {"kind": "near_clifford", "data_dir": check["data_dir"],
                        "dataset_hash": check["dataset_hash"],
                        "cache": str(_cache_path(out, key))}
                  for key, check in nc_checks.items()})
    write_json(out / "datasets.json", index)


def cmd_prepare(args) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sweep = bool(getattr(args, "sweep", False))
    fresh = bool(getattr(args, "fresh", False))
    fresh_source = getattr(args, "fresh_source", None)
    if not fresh and args.archive is None:
        raise SystemExit("--archive is required when --fresh is not specified")
    checks = prepare_primary(args.datasets, args.archive, out, args.label_tolerance,
                             sweep=sweep, fresh=fresh, fresh_source=fresh_source)
    nc_checks = prepare_nc(args.nc_datasets or [], out, args.nc_results)
    write_json(out / "dataset_checks.json", {"primary": checks, "near_clifford": nc_checks})
    _write_dataset_index(out, checks, nc_checks, fresh=fresh)
    diagnostics = {"schema": "descriptor-information-partA-diagnostics-v1",
                   "frozen_rule": common.RULE_FILE, "r4": {}, "encoder": {},
                   "coupling_noise": {}}
    for key, check in checks.items():
        data = load_cache(_cache_path(out, key))
        table = coupling_noise_table(data)
        write_json(out / "descriptors" / f"{key}__coupling_noise.json",
                   {"rule": "z ~ default_rng(SeedSequence([20261002, dataset_seed, "
                            "circuit_index])).standard_normal(n_couplings); "
                            "circuit_index = position in sorted circuit_id over all "
                            "roles and families",
                    "dataset_seed": int(data["seed"]), "circuits": table["circuits"]})
        z = np.asarray([v for c in table["circuits"] for v in c["z"]])
        diagnostics["coupling_noise"][key] = {
            "n_circuits": len(table["circuits"]), "z_mean": float(z.mean()),
            "z_sd": float(z.std(ddof=1))}
        if sweep:
            # The sweep rungs (R0, N1, N2, R5) never read the R4 encoding.
            continue
        meta = prepare_encoding(key, data, Path(check["data_dir"]), out, args.workers)
        diagnostics["encoder"][key] = {k: meta[k] for k in (
            "n_circuits", "upstream_commit", "upstream_mlp_sha256",
            "released_entry_point_agrees", "structure_mismatches", "seconds")}
        diagnostics["r4"][key] = r4_diagnostics(key, data, out)
        print(f"{key}: R4 {json.dumps(diagnostics['r4'][key])[:400]}", flush=True)
    write_json(out / "diagnostics" / "partA_label_free.json", diagnostics)
    print(f"wrote {out / 'diagnostics' / 'partA_label_free.json'}")
    return 0


def _selected(args) -> tuple[list[str], list[str], list[int]]:
    sweep = bool(getattr(args, "sweep", False))
    if sweep:
        default_rungs = ["R0", "N1", "N2", "R5"]
        default_arms = ["A", "C", "F", "P"]
    else:
        default_rungs = DEFAULT_RUNGS
        default_arms = ARMS
    rungs = list(args.rungs) if args.rungs else list(default_rungs)
    unknown = sorted(set(rungs) - set(RUNGS) - set(NC_RUNGS))
    if unknown:
        raise SystemExit(f"unknown rungs {unknown}; choose from {[*RUNGS, *NC_RUNGS]}")
    arms = list(args.arms) if args.arms else list(default_arms)
    if set(arms) - set(ARMS):
        raise SystemExit(f"unknown arms; choose from {ARMS}")
    return rungs, arms, common._parse_seeds(args.seeds)


def _write_reexport_check(args, out: Path, primary: list[str], nc: list[str],
                          seeds: list[int]) -> dict:
    report = check_reexports(out, primary, nc, seeds, args.seedrep_fits,
                             args.nc_results if nc else None)
    write_json(out / "reexport_check.json", report)
    print(json.dumps({"row_order_all_match": report["row_order"]["all_match"],
                      "groups": report["groups"],
                      "r0_reuses_appendix_m4": report["r0_reuses_appendix_m4"],
                      "nc_reexport_matches_appendix_n":
                          report["nc_reexport_matches_appendix_n"],
                      "any_differs": report["any_differs"]}, indent=1))
    print(f"wrote {out / 'reexport_check.json'}")
    return report


def cmd_check_reexport(args) -> int:
    out = args.out.resolve()
    primary = [key for key, _ in _primary_keys(args)]
    nc = [key for key, _ in _nc_keys(args)]
    if args.seedrep_fits is None and not (nc and args.nc_results):
        raise SystemExit("check-reexport needs --seedrep-fits and/or "
                         "--nc-datasets with --nc-results")
    report = _write_reexport_check(args, out, primary, nc,
                                   common._parse_seeds(args.seeds))
    if report["any_differs"] or not report["row_order"]["all_match"]:
        return 4
    return 0


def _check_train_size_bounds(train_size: int, primary_keys: list[str], nc_keys: list[str],
                             nc_rungs: list[str], out: Path,
                             datasets: list[Path], nc_datasets: list[Path] | None) -> None:
    dataset_paths = {k: Path(d) for k, d in zip(primary_keys, datasets)}
    for key in primary_keys:
        cache_p = _cache_path(out, key)
        if cache_p.exists():
            data = load_cache(cache_p)
            families = {r.get("family") for r in data["train"]}
            for fam in families:
                n_circuits = len({r["circuit_id"] for r in data["train"] if r.get("family") == fam})
                if train_size > n_circuits:
                    raise SystemExit(
                        f"Refusal: --train-size {train_size} exceeds dataset {key} family {fam} "
                        f"training circuit count ({n_circuits})"
                    )
        elif key in dataset_paths and (dataset_paths[key] / "manifest.json").exists():
            manifest = json.loads((dataset_paths[key] / "manifest.json").read_text(encoding="utf-8"))
            rc = manifest.get("split_spec", {}).get("role_counts", {})
            fams = manifest.get("split_spec", {}).get("fixed_axes", {}).get("circuit_family", [])
            total_train = rc.get("train", 0)
            n_fams = len(fams) if fams else 1
            per_fam = total_train // n_fams if total_train else 0
            if per_fam and train_size > per_fam:
                raise SystemExit(
                    f"Refusal: --train-size {train_size} exceeds dataset {key} "
                    f"training circuit count ({per_fam})"
                )

    if nc_rungs and nc_datasets:
        nc_dataset_paths = {k: Path(d) for k, d in zip(nc_keys, nc_datasets)}
        for key in nc_keys:
            cache_p = _cache_path(out, key)
            if cache_p.exists():
                data = load_cache(cache_p)
                n_circuits = len({r["circuit_id"] for r in data["train"]})
                if train_size > n_circuits:
                    raise SystemExit(
                        f"Refusal: --train-size {train_size} exceeds dataset {key} "
                        f"training circuit count ({n_circuits})"
                    )
            elif key in nc_dataset_paths and (nc_dataset_paths[key] / "manifest.json").exists():
                manifest = json.loads((nc_dataset_paths[key] / "manifest.json").read_text(encoding="utf-8"))
                rc = manifest.get("split_spec", {}).get("role_counts", {})
                train_c = rc.get("train", 0)
                if train_c and train_size > train_c:
                    raise SystemExit(
                        f"Refusal: --train-size {train_size} exceeds dataset {key} "
                        f"training circuit count ({train_c})"
                    )


def cmd_run(args) -> int:
    out = args.out.resolve()
    fresh = bool(getattr(args, "fresh", False))
    index_path = out / "datasets.json"
    if not fresh and index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            if index_data.get("_fresh") or any(
                isinstance(v, dict) and v.get("fresh") for v in index_data.values()
            ):
                fresh = True
                args.fresh = True
        except Exception:
            pass
    rungs, arms, seeds = _selected(args)
    primary = [key for key, _ in _primary_keys(args)]
    nc = [key for key, _ in _nc_keys(args)]
    nc_rungs = [rung for rung in rungs if rung in NC_RUNGS]
    if nc_rungs and not nc:
        raise SystemExit(f"{nc_rungs} need --nc-datasets")
    sweep = bool(getattr(args, "sweep", False))
    strength_indicator = bool(getattr(args, "strength_indicator", False) or sweep)
    strong_learners = bool(getattr(args, "strong_learners", False))
    train_size = getattr(args, "train_size", None)
    if train_size is not None:
        if train_size <= 0:
            raise SystemExit(f"Refusal: --train-size must be a positive integer, got {train_size}")
    neural_es = bool(getattr(args, "neural_es", False))
    if neural_es and strong_learners:
        raise SystemExit(
            "Refusal: --neural-es cannot be used with --strong-learners; "
            "--neural-es is incompatible with --strong-learners."
        )
    fit_dir = out / "fits"
    if fit_dir.exists():
        for json_path in fit_dir.glob("*.json"):
            try:
                fit_meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if strong_learners and not fit_meta.get("strong_learners"):
                raise SystemExit(
                    f"Refusal: {json_path.name} does not have strong_learners: true; "
                    "run --strong-learners refuses to start in a fits directory holding non-strong fits."
                )
            if neural_es and not fit_meta.get("neural_es"):
                raise SystemExit(
                    f"Refusal: {json_path.name} does not have neural_es: true; "
                    "run --neural-es refuses to start in a fits directory holding non-neural-es fits."
                )
            if not neural_es and fit_meta.get("neural_es"):
                raise SystemExit(
                    f"Refusal: {json_path.name} has neural_es: true; "
                    "run without --neural-es refuses to start in a fits directory holding neural-es fits."
                )
            existing_size = fit_meta.get("train_size")
            if existing_size != train_size:
                raise SystemExit(
                    f"Refusal: {json_path.name} has train_size {existing_size!r} != {train_size!r}; "
                    "refuse to mix training sizes in one fits directory."
                )
    if train_size is not None:
        _check_train_size_bounds(train_size, primary, nc, nc_rungs, out, args.datasets, args.nc_datasets)
    jobs = build_jobs(out, primary, nc, rungs, arms, seeds,
                      strength_indicator=strength_indicator,
                      strong_learners=strong_learners,
                      train_size=train_size,
                      neural_es=neural_es,
                      fresh=fresh)
    aliases = []
    if "R5" in rungs:
        aliases.append("M (Part A) = F at R5, the same fits")
    if "R0" in rungs and not strength_indicator and train_size is None and not fresh:
        aliases.append("R0 original-seed F and C = the gate fits in gate/fits "
                       "(re-exports of Appendix M.4's __orig__ F and C)")
    if args.dry_run:
        common.print_plan(jobs, out / "fits", aliases)
        return 0
    common.require_gate(args.gate_file or out / "gate" / "gate_r0.json")
    if not strength_indicator and train_size is None and not fresh:
        # A re-export has to be checkable against what it re-exports.
        if "R0" in rungs and args.seedrep_fits is None:
            raise SystemExit("R0 re-exports Appendix M.4: pass --seedrep-fits so each fit "
                             "is checked against M.4")
        if NC_REEXPORT_RUNG in rungs and args.nc_results is None:
            raise SystemExit("NC-R0 re-exports Appendix N: pass --nc-results so each fit "
                             "is checked against Appendix N")
    index_path = out / "datasets.json"
    needed = set(primary) | (set(nc) if nc_rungs else set())
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    encoder_ready = all(_encoder_paths(out, key)[0].exists() for key in primary)
    if not needed <= set(index) or ("R4" in rungs and not encoder_ready):
        if not fresh and args.archive is None:
            raise SystemExit("--archive is required when --fresh is not specified")
        cmd_prepare(args)
    reexporting = [rung for rung in rungs if rung in REEXPORT_RUNGS] if (not strength_indicator and train_size is None and not fresh) else []
    if reexporting:
        order = reexport_preconditions(
            out, primary if "R0" in rungs else [],
            nc if NC_REEXPORT_RUNG in rungs else [],
            args.seedrep_fits if "R0" in rungs else None,
            args.nc_results if NC_REEXPORT_RUNG in rungs else None)
        if not order["all_match"]:
            raise SystemExit(f"cached row order differs from the re-export reference: "
                             f"{json.dumps(order)}")
    # A strong run checks its caches against the strength-indicator record by
    # default; any run that names --cache-record checks against that record.
    cache_record = getattr(args, "cache_record", None)
    cache_digests = (common.check_caches_recorded(out / "cache", primary,
                                                  record_path=cache_record)
                     if strong_learners or cache_record is not None else None)
    run_cfg = {
        "rungs": rungs, "arms": arms, "seeds": seeds, "workers": args.workers,
        "n_jobs": len(jobs), "started_unix": time.time(),
    }
    if strength_indicator:
        run_cfg["strength_indicator"] = True
    if strong_learners:
        run_cfg["strong_learners"] = True
    if cache_digests is not None:
        run_cfg["cache_sha256"] = cache_digests
    if cache_record is not None:
        run_cfg["cache_record"] = {"path": str(cache_record),
                                   "sha256": common.sha256_file(Path(cache_record))}
    if train_size is not None:
        run_cfg["train_size"] = train_size
    if neural_es:
        run_cfg["neural_es"] = True
    common.write_json(out / "run_config.json", run_cfg)
    info = common.drive(jobs, run_job, out, args.workers, limit=args.limit_jobs)
    write_json(out / f"run_{int(time.time())}.json", info)
    if info["crashes"]:
        return 3
    if reexporting:
        report = _write_reexport_check(
            args, out, primary if "R0" in rungs else [],
            nc if NC_REEXPORT_RUNG in rungs else [], seeds)
        if report["any_differs"]:
            return 4
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common_arguments(p, *, nc: bool):
        p.add_argument("--datasets", nargs="+", required=True, type=Path,
                       help="the regenerated primary datasets")
        p.add_argument("--archive", required=False, default=None, type=Path)
        p.add_argument("--fresh", action="store_true",
                       help="accept fresh datasets without requiring campaign archive records")
        p.add_argument("--fresh-source", type=Path, default=None,
                       help="2048-shot source dataset directory for fresh sweep preparation")
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--label-tolerance", type=float, default=1e-11,
                       help="as in the Appendix M.4 run (s211 differs by 1e-12 in "
                            "two validation labels)")
        p.add_argument("--sweep", action="store_true",
                       help="accept sweep datasets without requiring 2048-shot archive hashes")
        if nc:
            p.add_argument("--nc-datasets", nargs="*", type=Path, default=[])
            p.add_argument("--nc-results", type=Path, default=None,
                           help="Appendix N results dir; checks the dataset hashes")

    seedrep_help = ("Appendix M.4 fits dir (seedrep/fits); its sibling cache/ holds the "
                    "row order the re-export check compares")
    gate = sub.add_parser("gate", help="R0 reproduction gate (F, C at the original seed)")
    common_arguments(gate, nc=False)
    gate.add_argument("--criterion", choices=sorted(common.GATE_CRITERIA),
                      default=common.FROZEN_GATE_CRITERION,
                      help="the frozen gate is 'archive' at 1e-11; anything else is an "
                           "amendment and needs --amendment")
    gate.add_argument("--tolerance", type=float, default=common.GATE_TOLERANCE)
    gate.add_argument("--amendment", default=None,
                      help="the amendment text, verbatim as written into the rule file; "
                           "required for any criterion or tolerance other than the "
                           "frozen one")
    gate.add_argument("--seedrep-fits", type=Path, default=None, help=seedrep_help)
    prepare = sub.add_parser("prepare", help="verify, cache, encode; no fit")
    common_arguments(prepare, nc=True)
    run = sub.add_parser("run", help="fit the ladder (needs a passed gate)")
    common_arguments(run, nc=True)
    run.add_argument("--rungs", nargs="+", default=None)
    run.add_argument("--arms", nargs="+", default=None)
    run.add_argument("--seeds", default="1-20")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--strength-indicator", action="store_true",
                     help="append noise_strength_L3 feature to every arm and rung")
    run.add_argument("--strong-learners", action="store_true",
                     help="enable strong learner candidates (hgbr, poly5_ridge)")
    run.add_argument("--train-size", type=int, default=None,
                     help="use only the first N training circuits per family")
    run.add_argument("--neural-es", action="store_true",
                     help="enable well-trained MLP with early stopping and lr plateau schedule")
    run.add_argument("--gate-file", type=Path, default=None)
    run.add_argument("--cache-record", type=Path, default=None,
                     help="a JSON whose inputs.caches_used gives each primary cache's "
                          "SHA-256; the run exits before any fit on a difference")
    run.add_argument("--limit-jobs", type=int, default=None)
    run.add_argument("--seedrep-fits", type=Path, default=None, help=seedrep_help)
    check = sub.add_parser("check-reexport",
                           help="compare R0 and NC-R0 re-exports with M.4 and Appendix N")
    check.add_argument("--datasets", nargs="+", required=True, type=Path)
    check.add_argument("--nc-datasets", nargs="*", type=Path, default=[])
    check.add_argument("--nc-results", type=Path, default=None)
    check.add_argument("--seedrep-fits", type=Path, default=None, help=seedrep_help)
    check.add_argument("--out", required=True, type=Path)
    check.add_argument("--seeds", default="1-20")
    args = parser.parse_args(argv)
    if args.command == "gate":
        return cmd_gate(args)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "check-reexport":
        return cmd_check_reexport(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
