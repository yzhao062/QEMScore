"""Descriptor-information experiment, Part B: QAOA-MaxCut on random graphs.

Frozen rule: docs/frozen-rules/2026-10-02-descriptor-information.md (Part B,
Arms, Reference, Pre-Fit Diagnostics, Deviations). Fitting, output format and
convergence flags are shared with Part A through tools/descriptor_common.py.

Subcommands
-----------
generate     Builds and validates the three QAOA datasets with the released
             generator (``generate_split``) and validator
             (``validate_split_artifact``), the way
             tools/near_clifford_positive_control.py builds its control: setting
             S0, 10 qubits, p in {1, 2} (the family-native depth axis, so p is
             balanced), graph classes Erdos-Renyi (edge probability 0.3) and
             3-regular, depolarizing-plus-readout noise at L1 and L3, observable
             class zz_mid only, 2,048 shots, 640/320/160 circuits, dataset
             seeds 101, 211, 307. If the grammar refuses zz_mid as the only
             observable class, it says so and stops. If validation fails at 10
             qubits, it tries the next smaller even count (8, 6, 4) for all
             three seeds and records the choice, which the rule requires to be
             made before any fit and disclosed.
diagnostics  Label-free pre-fit diagnostics (no test row, no fit): training-label
             SD per cell and per graph class, and the attenuation slope of the
             noisy estimate per cell; applies the rule's drop of any cell whose
             training-label SD is below 0.01 and writes diagnostics.json.
run          Fits rungs B-partial, B-complete, B-none with arms A (once per
             dataset and rung), C, F, P at learner seeds 1 to 20, and the
             gradient-boosted-tree reference on B-complete. M is F at B-none (the
             same fits). Refuses unless the R0 gate passed. ``--dry-run`` lists
             the planned fits; ``--rungs`` and ``--arms`` filter them.

Rungs
-----
B-partial   The schema vector (``build_features``): its QAOA columns are p, edge
            count, two gammas, two betas and the graph-class indicators.
B-complete  B-partial plus one indicator per qubit pair (45 at 10 qubits) from
            the stored edge set, appended in ``itertools.combinations`` order.
B-none      Only the kept columns (the R5 analogue).

Choices where the rule is silent:
- The training-label SD is the sample SD (ddof = 1) over the cell's training
  items; ddof = 0 is recorded beside it. The attenuation slope is the
  least-squares slope of the noisy estimate on the ideal label over the cell's
  training items.
- A dropped cell's rows leave training, validation and test alike.
- The reference reads B-complete's columns without the noisy estimate, which is
  C's input at B-complete. It is tuned over a fixed 8-point grid (learning rate
  0.05, 0.1; max leaf nodes 7, 31; iterations 100, 300; no early stopping;
  random_state 0) by source-validation macro MAE, ties to the earlier grid
  point. It is a reference, not an arm.

Usage
-----
  PYTHONPATH=. python tools/qaoa_intermediate.py generate --data-root RUNS/qaoa-data
  PYTHONPATH=. python tools/qaoa_intermediate.py diagnostics --data-root RUNS/qaoa-data
  PYTHONPATH=. python tools/qaoa_intermediate.py run --data-root RUNS/qaoa-data \
      --out RUNS/partB --gate-file RUNS/partA/gate/gate_r0.json --workers 8 \
      [--dry-run] [--rungs B-partial ...] [--arms A F C P GBT] [--seeds 1-20]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path[:1]:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools import descriptor_common as common  # noqa: E402
from tools.descriptor_common import (  # noqa: E402
    FEATURES,
    KEPT,
    NOISY,
    load_cache,
    write_cache,
    write_json,
)
from qemscore.datasets.schema import build_features  # noqa: E402
from qemscore.datasets.split_generate import generate_split  # noqa: E402
from qemscore.datasets.splits import SplitSpec, resolve_split_spec  # noqa: E402
from qemscore.runner.run import _prediction_items  # noqa: E402
from qemscore.validation import split_spec_hash, validate_split_artifact  # noqa: E402

PART = "B"
DATASET_SEEDS = (101, 211, 307)
N_QUBITS = 10
FALLBACK_QUBITS = (10, 8, 6, 4)
P_VALUES = [1, 2]
GRAPH_CLASSES = ["erdos_renyi", "3_regular"]
ER_EDGE_PROBABILITY = 0.3
ROLE_COUNTS = {"train": 640, "validation": 320, "test": 160}
NOISE_FAMILY = "depolarizing_readout"
SEVERITIES = ["L1", "L3"]
OBSERVABLES = ["zz_mid"]
SHOTS = 2048
SD_DROP_THRESHOLD = 0.01
CELL_FIELDS = ("noise_family", "severity", "observable")
RUNGS = ("B-partial", "B-complete", "B-none")
ARMS = ("A", "F", "C", "P", "GBT")
GBT_GRID = tuple(
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": iterations}
    for iterations in (100, 300)
    for leaves in (7, 31)
    for lr in (0.05, 0.1)
)


def qaoa_spec(n_qubits: int = N_QUBITS, observables=OBSERVABLES) -> SplitSpec:
    """The frozen Part B split specification (setting S0)."""
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": [NOISE_FAMILY],
            "noise_strength": list(SEVERITIES),
            "circuit_family": ["qaoa"],
            "family_native_depth": list(P_VALUES),
            "observable_class": list(observables),
            "shots": [SHOTS],
        },
        n_qubits=[n_qubits],
        role_counts=dict(ROLE_COUNTS),
        family_parameters={"qaoa": {"graph_classes": list(GRAPH_CLASSES),
                                    "er_edge_probability": ER_EDGE_PROBABILITY}},
        budget_tier="H",
    )


def dataset_name(seed: int, n_qubits: int) -> str:
    prefix = "qaoa" if n_qubits == N_QUBITS else f"qaoa-q{n_qubits}"
    return f"{prefix}-s{seed}-n640"


def _cell(row) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in CELL_FIELDS)


# --------------------------------------------------------------------------
# Generation and validation
# --------------------------------------------------------------------------


def _generate_one(task: tuple[int, int, str]) -> dict:
    seed, n_qubits, root = task
    data_dir = Path(root) / dataset_name(seed, n_qubits)
    entry = {"seed": seed, "n_qubits": n_qubits, "data_dir": str(data_dir)}
    started = time.perf_counter()
    try:
        if (data_dir / "items.jsonl").exists() and (data_dir / "manifest.json").exists():
            entry["generated_now"] = False
        else:
            generate_split(qaoa_spec(n_qubits), data_dir, master_seed=seed)
            entry["generated_now"] = True
        entry["generation_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        rows, manifest = validate_split_artifact(data_dir)
        entry["validation_seconds"] = time.perf_counter() - started
        entry["validated"] = True
        entry.update(_describe(rows, manifest, data_dir))
    except Exception as exc:  # recorded and decided on by the caller
        entry["validated"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-3000:]
    return entry


def _describe(rows: list[dict], manifest: dict, data_dir: Path) -> dict:
    roles = ("train", "validation", "test")
    circuits = {}
    for row in rows:
        circuits.setdefault(str(row["circuit_id"]), row)
    by_role = {role: [r for r in circuits.values() if r["split"] == role] for role in roles}
    edges = np.asarray([len(r["edges"]) for r in circuits.values()])
    return {
        "dataset_hash": str(manifest["dataset_hash"]),
        "split_spec_hash": str(manifest["split_spec_hash"]),
        "master_seed": int(manifest["master_seed"]),
        "items_jsonl_sha256": common.sha256_file(data_dir / "items.jsonl"),
        "manifest_sha256": common.sha256_file(data_dir / "manifest.json"),
        "n_items": {role: sum(r["split"] == role for r in rows) for role in roles},
        "n_circuits": {role: len(by_role[role]) for role in roles},
        "graph_class_circuits": {
            role: {g: sum(r["graph_class"] == g for r in by_role[role])
                   for g in GRAPH_CLASSES} for role in roles},
        "p_circuits": {role: {str(p): sum(r["p"] == p for r in by_role[role])
                              for p in P_VALUES} for role in roles},
        "cells": sorted({"/".join(_cell(r)) for r in rows}),
        "edge_count": {"min": int(edges.min()), "max": int(edges.max()),
                       "mean": float(edges.mean())},
        "distinct_edge_sets": len({json.dumps(sorted(map(sorted, r["edges"])))
                                   for r in circuits.values()}),
        "versions": manifest.get("versions"),
    }


def cmd_generate(args) -> int:
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"schema": "qaoa-intermediate-generation-v1",
              "frozen_rule": common.RULE_FILE, "attempts": [], "chosen_n_qubits": None}
    # Does the grammar accept zz_mid as the only observable class?
    try:
        resolved = resolve_split_spec(qaoa_spec(N_QUBITS))
        report["zz_mid_only_accepted"] = True
        report["resolved_cells"] = len(resolved.cells)
        report["resolved_pools"] = len(resolved.circuit_pools)
    except Exception as exc:
        report["zz_mid_only_accepted"] = False
        report["spec_error"] = f"{type(exc).__name__}: {exc}"
        try:
            resolve_split_spec(qaoa_spec(N_QUBITS, ["z_mid", "zz_mid"]))
            report["spec_with_z_mid_resolves"] = True
        except Exception as other:
            report["spec_with_z_mid_resolves"] = f"{type(other).__name__}: {other}"
        write_json(root / "generation_report.json", report)
        print("STOP: the split grammar does not accept zz_mid as the only observable "
              f"class: {report['spec_error']}", flush=True)
        return 4
    for n_qubits in FALLBACK_QUBITS:
        spec = qaoa_spec(n_qubits)
        attempt = {"n_qubits": n_qubits, "split_spec": spec.to_dict(),
                   "split_spec_hash": split_spec_hash(spec), "datasets": []}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(DATASET_SEEDS))) as pool:
            attempt["datasets"] = list(pool.map(
                _generate_one, [(seed, n_qubits, str(root)) for seed in DATASET_SEEDS]))
        attempt["all_validated"] = all(d["validated"] for d in attempt["datasets"])
        report["attempts"].append(attempt)
        write_json(root / "generation_report.json", report)
        if attempt["all_validated"]:
            report["chosen_n_qubits"] = n_qubits
            report["fallback_used"] = n_qubits != N_QUBITS
            break
        print(f"validation failed at {n_qubits} qubits: "
              f"{[d.get('error') for d in attempt['datasets']]}", flush=True)
    write_json(root / "generation_report.json", report)
    if report["chosen_n_qubits"] is None:
        print("STOP: no even qubit count up to 10 validates", flush=True)
        return 5
    for d in report["attempts"][-1]["datasets"]:
        print(f"{Path(d['data_dir']).name}: hash={d['dataset_hash'][:16]} "
              f"items={d['n_items']} graphs={d['graph_class_circuits']['train']} "
              f"gen={d['generation_seconds']:.0f}s val={d['validation_seconds']:.0f}s",
              flush=True)
    return 0


def _chosen(root: Path) -> tuple[int, list[dict]]:
    report = json.loads((root / "generation_report.json").read_text(encoding="utf-8"))
    if report.get("chosen_n_qubits") is None:
        raise SystemExit(f"{root}: no validated QAOA datasets; run generate first")
    attempt = next(a for a in report["attempts"]
                   if a["n_qubits"] == report["chosen_n_qubits"])
    return int(report["chosen_n_qubits"]), attempt["datasets"]


# --------------------------------------------------------------------------
# Label-free diagnostics and the cell drop
# --------------------------------------------------------------------------


def _label_stats(rows: list[dict]) -> dict:
    ideal = np.asarray([float(r["ideal_expectation"]) for r in rows])
    noisy = np.asarray([float(r["noisy_expectation"]) for r in rows])
    entry = {
        "n_items": int(ideal.size),
        "n_circuits": len({str(r["circuit_id"]) for r in rows}),
        "label_mean": float(ideal.mean()),
        "label_sd": float(ideal.std(ddof=1)) if ideal.size > 1 else None,
        "label_sd_ddof0": float(ideal.std(ddof=0)),
        "label_min": float(ideal.min()),
        "label_max": float(ideal.max()),
        "noisy_sd": float(noisy.std(ddof=1)) if noisy.size > 1 else None,
    }
    variance = float(np.var(ideal))
    if variance > 0.0:
        slope = float(np.mean((ideal - ideal.mean()) * (noisy - noisy.mean())) / variance)
        entry["attenuation_slope"] = slope
        entry["attenuation_intercept"] = float(noisy.mean() - slope * ideal.mean())
        entry["noisy_ideal_correlation"] = float(np.corrcoef(ideal, noisy)[0, 1])
    else:
        entry["attenuation_slope"] = None
    entry["raw_mae"] = float(np.mean(np.abs(noisy - ideal)))
    return entry


def diagnostics_for(rows: list[dict]) -> dict:
    train = [r for r in rows if r["split"] == "train"]
    cells = sorted({_cell(r) for r in train})
    report = {"cells": {}, "graph_class_pooled": {}, "dropped_cells": []}
    for cell in cells:
        members = [r for r in train if _cell(r) == cell]
        name = "/".join(cell)
        entry = _label_stats(members)
        entry["by_graph_class"] = {
            g: _label_stats([r for r in members if r["graph_class"] == g])
            for g in GRAPH_CLASSES if any(r["graph_class"] == g for r in members)}
        entry["by_p"] = {
            str(p): _label_stats([r for r in members if r["p"] == p])
            for p in P_VALUES if any(r["p"] == p for r in members)}
        entry["dropped"] = bool(entry["label_sd"] is None
                                or entry["label_sd"] < SD_DROP_THRESHOLD)
        if entry["dropped"]:
            report["dropped_cells"].append(name)
        report["cells"][name] = entry
    # Graph class pooled over circuits (labels do not depend on the noise cell).
    circuits = {}
    for r in train:
        circuits.setdefault((str(r["circuit_id"]), str(r["observable"])), r)
    for g in GRAPH_CLASSES:
        labels = np.asarray([float(r["ideal_expectation"]) for r in circuits.values()
                             if r["graph_class"] == g])
        report["graph_class_pooled"][g] = {
            "n_circuits": int(labels.size),
            "label_sd": float(labels.std(ddof=1)) if labels.size > 1 else None,
            "label_mean": float(labels.mean()) if labels.size else None}
    return report


def cmd_diagnostics(args) -> int:
    root = args.data_root.resolve()
    n_qubits, datasets = _chosen(root)
    out = {"schema": "qaoa-intermediate-diagnostics-v1", "frozen_rule": common.RULE_FILE,
           "label_free": ("training rows only; no validation or test label, no fit"),
           "sd_drop_threshold": SD_DROP_THRESHOLD,
           "sd_definition": "sample SD (ddof=1) of ideal_expectation over the cell's "
                            "training items",
           "attenuation_slope_definition": "least-squares slope of noisy_expectation "
                                           "on ideal_expectation over the cell's "
                                           "training items",
           "n_qubits": n_qubits, "datasets": {}, "dropped_cells": {}}
    for entry in datasets:
        data_dir = Path(entry["data_dir"])
        rows, manifest = validate_split_artifact(data_dir)
        if str(manifest["dataset_hash"]) != entry["dataset_hash"]:
            raise SystemExit(f"{data_dir}: dataset hash changed since generation")
        report = diagnostics_for(rows)
        report["dataset_hash"] = entry["dataset_hash"]
        out["datasets"][data_dir.name] = report
        out["dropped_cells"][data_dir.name] = report["dropped_cells"]
        summary = {cell: (round(v["label_sd"], 4), round(v["attenuation_slope"], 4))
                   for cell, v in report["cells"].items()}
        print(f"{data_dir.name}: (label SD, attenuation slope) by cell {summary}; "
              f"dropped {report['dropped_cells']}", flush=True)
    out["all_cells_dropped_somewhere"] = any(
        len(v) == len(out["datasets"][k]["cells"]) for k, v in out["dropped_cells"].items())
    write_json(root / "diagnostics.json", out)
    print(f"wrote {root / 'diagnostics.json'}")
    return 0


# --------------------------------------------------------------------------
# Rung builders
# --------------------------------------------------------------------------


class QAOABuilder:
    """Feature builder for one Part B rung."""

    def __init__(self, rung: str, n_qubits: int) -> None:
        if rung not in RUNGS:
            raise ValueError(f"unknown rung {rung!r}")
        self.rung = rung
        self.name = f"qaoa-intermediate:{rung}"
        self.pairs = tuple(combinations(range(n_qubits), 2))
        self._pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        if rung == "B-partial":
            names = list(FEATURES)
        elif rung == "B-complete":
            names = [*FEATURES, *(f"edge_{i}_{j}" for i, j in self.pairs)]
        else:
            names = list(KEPT)
        self.names = tuple(names)

    def edge_indicators(self, item: dict) -> list[float]:
        indicators = [0.0] * len(self.pairs)
        for q0, q1 in item["edges"]:
            pair = (min(int(q0), int(q1)), max(int(q0), int(q1)))
            indicators[self._pair_index[pair]] = 1.0
        return indicators

    def __call__(self, item: dict) -> list[float]:
        base = build_features(item)
        if self.rung == "B-partial":
            return base
        if self.rung == "B-complete":
            return [*base, *self.edge_indicators(item)]
        values = dict(zip(FEATURES, base))
        return [values[n] for n in KEPT]


_BUILDERS: dict[tuple[str, int], tuple] = {}


def builder_for(rung: str, n_qubits: int):
    key = (rung, n_qubits)
    if key not in _BUILDERS:
        if rung == "B-partial":
            _BUILDERS[key] = (build_features, tuple(FEATURES))
        else:
            builder = QAOABuilder(rung, n_qubits)
            _BUILDERS[key] = (builder, builder.names)
    return _BUILDERS[key]


# --------------------------------------------------------------------------
# The gradient-boosted reference
# --------------------------------------------------------------------------


def run_gbt_fit(job: dict, data: dict) -> dict:
    """Reference: HistGradientBoostingRegressor on B-complete without the noisy estimate."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    started = time.perf_counter()
    builder = QAOABuilder("B-complete", int(data["n_qubits"]))
    columns = [i for i, name in enumerate(builder.names) if name != NOISY]

    def matrix(rows):
        return np.asarray([builder(dict(r)) for r in rows], dtype=float)[:, columns]

    x_train = matrix(data["train"])
    y_train = np.asarray([float(r["ideal_expectation"]) for r in data["train"]])
    x_val = matrix(data["prediction_rows"]["validation"])
    x_test = matrix(data["prediction_rows"]["test"])
    scores = []
    best = None
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for index, params in enumerate(GBT_GRID):
                model = HistGradientBoostingRegressor(
                    loss="squared_error", early_stopping=False, random_state=0, **params)
                model.fit(x_train, y_train)
                val_pred = model.predict(x_val)
                mae = common._family_mae(data["validation"], val_pred,
                                         artifact_id=str(data["dataset_hash"]))["qaoa"]
                scores.append({"grid_index": index, **params, "validation_mae": mae})
                if best is None or mae < best[0]:
                    best = (mae, index, model, val_pred)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    predictions = {}
    if best is not None and error is None:
        predictions = {"validation": best[3], "test": best[2].predict(x_test)}
    dataset_hash = str(data["dataset_hash"])
    result = {
        "schema": "descriptor-information-fit-v1", "frozen_rule": common.RULE_FILE,
        "part": PART, "key": job["key"], "dataset_seed": int(data["seed"]),
        "dataset_hash": dataset_hash, "rung": job["rung"], "arm": "GBT",
        "method": "HistGradientBoostingRegressor reference (not an arm)",
        "feature_names": [builder.names[i] for i in columns],
        "n_features": len(columns), "grid": list(GBT_GRID),
        "fixed": {"loss": "squared_error", "early_stopping": False, "random_state": 0},
        "selection": "lowest source-validation macro MAE; ties to the earlier grid point",
        "validation_scores": scores,
        "selected": None if best is None else {"grid_index": best[1],
                                               **GBT_GRID[best[1]]},
        "error": error,
        "warnings_aggregated": [{"category": w.category.__name__,
                                 "message": str(w.message)[:300]} for w in caught][:50],
        "test_family_mae": (common._family_mae(data["test"], predictions["test"],
                                               artifact_id=dataset_hash)
                            if predictions else None),
        "seconds": time.perf_counter() - started,
    }
    common._write_fit(job, result, predictions)
    return {"stem": job["stem"], "seconds": result["seconds"], "flagged": None,
            "flagged_secondary": None, "error": error, "selected": "gbt"}


# --------------------------------------------------------------------------
# Jobs and the run
# --------------------------------------------------------------------------


def run_job(job: dict) -> dict:
    data = load_cache(job["cache"])
    if job["kind"] == "gbt":
        return run_gbt_fit(job, data)
    builder, names = builder_for(job["rung"], int(data["n_qubits"]))
    if job["kind"] == "affine":
        return common.run_affine_fit(job, data, builder, names)
    return common.run_liao_fit(job, data, builder, names)


def _job(out: Path, key: str, rung: str, arm: str, *, learner_seed=None) -> dict:
    kind = {"A": "affine", "GBT": "gbt"}.get(arm, "liao")
    seed_part = "" if learner_seed is None else f"__k{learner_seed:02d}"
    return {
        "part": PART, "kind": kind, "key": key, "rung": rung, "arm": arm,
        "fit_arm": arm, "learner_seed": learner_seed,
        "shuffle_seed": learner_seed if arm == "P" else None,
        "purpose": "reference" if arm == "GBT" else "rung_fit",
        "stem": f"{key}__{rung}{seed_part}__{arm}",
        "cache": str(out / "cache" / f"{key}.pkl"), "fit_dir": str(out / "fits"),
    }


def build_jobs(out: Path, keys: list[str], rungs: list[str], arms: list[str],
               seeds: list[int]) -> list[dict]:
    jobs = []
    for rung in rungs:
        for key in keys:
            if "A" in arms:
                jobs.append(_job(out, key, rung, "A"))
            if "GBT" in arms and rung == "B-complete":
                jobs.append(_job(out, key, rung, "GBT"))
    for k in seeds:
        for rung in rungs:
            for key in keys:
                for arm in ("F", "C", "P"):
                    if arm in arms:
                        jobs.append(_job(out, key, rung, arm, learner_seed=k))
    return jobs


def prepare_caches(root: Path, out: Path) -> dict:
    """Validate each dataset, check its hash, drop cells, and cache the rows."""
    n_qubits, datasets = _chosen(root)
    diagnostics_path = root / "diagnostics.json"
    if not diagnostics_path.exists():
        raise SystemExit(f"{diagnostics_path} is missing: run diagnostics before any fit")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    checks = {}
    for entry in datasets:
        data_dir = Path(entry["data_dir"])
        key = data_dir.name
        rows, manifest = validate_split_artifact(data_dir)
        if str(manifest["dataset_hash"]) != entry["dataset_hash"]:
            raise SystemExit(f"{key}: dataset hash changed since generation")
        dropped = set(diagnostics["dropped_cells"][key])
        kept = [r for r in rows if "/".join(_cell(r)) not in dropped]
        if not kept:
            raise SystemExit(f"{key}: every cell is dropped; Part B cannot run here")
        roles = {role: [r for r in kept if r["split"] == role]
                 for role in ("train", "validation", "test")}
        payload = {"key": key, "seed": int(manifest["master_seed"]), **roles,
                   "prediction_rows": {role: _prediction_items(roles[role])
                                       for role in ("validation", "test")},
                   "dataset_hash": entry["dataset_hash"], "n_qubits": n_qubits,
                   "dropped_cells": sorted(dropped)}
        write_cache(out / "cache" / f"{key}.pkl", payload)
        order = {f"{role}_item_ids": [str(r["item_id"]) for r in roles[role]]
                 for role in ("validation", "test")}
        (out / "cache" / f"{key}.order.json").write_text(json.dumps(order),
                                                        encoding="utf-8")
        checks[key] = {"data_dir": str(data_dir), "dataset_hash": entry["dataset_hash"],
                       "dropped_cells": sorted(dropped),
                       "n_items": {role: len(v) for role, v in roles.items()}}
        print(f"cached {key}: {checks[key]['n_items']} dropped={sorted(dropped)}",
              flush=True)
    write_json(out / "datasets.json", checks)
    return checks


def cmd_run(args) -> int:
    root = args.data_root.resolve()
    out = args.out.resolve()
    rungs = list(args.rungs) if args.rungs else list(RUNGS)
    arms = list(args.arms) if args.arms else list(ARMS)
    if set(rungs) - set(RUNGS) or set(arms) - set(ARMS):
        raise SystemExit(f"rungs from {RUNGS}, arms from {ARMS}")
    seeds = common._parse_seeds(args.seeds)
    report_path = root / "generation_report.json"
    if report_path.exists():
        _, datasets = _chosen(root)
        keys = [Path(d["data_dir"]).name for d in datasets]
    else:
        keys = [dataset_name(seed, N_QUBITS) for seed in DATASET_SEEDS]
    jobs = build_jobs(out, keys, rungs, arms, seeds)
    aliases = ["M (Part B) = F at B-none, the same fits"] if "B-none" in rungs else []
    if args.dry_run:
        common.print_plan(jobs, out / "fits", aliases)
        return 0
    common.require_gate(args.gate_file)
    out.mkdir(parents=True, exist_ok=True)
    prepare_caches(root, out)
    write_json(out / "run_config.json", {"rungs": rungs, "arms": arms, "seeds": seeds,
                                         "workers": args.workers, "n_jobs": len(jobs)})
    info = common.drive(jobs, run_job, out, args.workers, limit=args.limit_jobs)
    write_json(out / f"run_{int(time.time())}.json", info)
    return 0 if not info["crashes"] else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--data-root", required=True, type=Path)
    gen.add_argument("--workers", type=int, default=3)
    diag = sub.add_parser("diagnostics")
    diag.add_argument("--data-root", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--gate-file", required=True, type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--rungs", nargs="+", default=None)
    run.add_argument("--arms", nargs="+", default=None)
    run.add_argument("--seeds", default="1-20")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--limit-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "diagnostics":
        return cmd_diagnostics(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
