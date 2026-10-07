"""Zero-shot transfer of the descriptor ladder under a depth shift and a noise shift.

Frozen rule: docs/frozen-rules/2026-10-06-shift-transfer.md. Fitting, feature
builders, and analysis are those of the descriptor-information ladder
(tools/descriptor_ladder.py, tools/descriptor_common.py,
tools/descriptor_ladder_analysis.py, rule 2026-10-02); this tool only supplies
the shifted test rows and checks that the fitted models are the S0 ladder's.

Shifts
------
S4  family_native_depth: source [3] (the S0 value), target [6].
S2  noise_strength: source [L1, L3] (the S0 values), target [L4].
Every other axis is the campaign's shipped regime at training size 640
(``campaign_split_spec("shipped", 640)``). Dataset seeds 101, 211, 307.

Subcommands
-----------
generate  Generates one split-v2 artifact per shift and seed with
          ``generate_split`` (master seed = dataset seed) into
          ``<root>/<shift>/shipped-s<seed>-n640`` and checks it with
          ``validate_split_artifact``.
prepare   For each shift and seed: checks the artifact's specification and
          target structure; checks that its train and validation rows equal the
          S0 dataset's rows field by field (source identity); then writes a
          ladder cache whose train and validation rows (and validation
          prediction rows) are the S0 dataset's and whose test rows are the
          artifact's target rows. The S0 dataset is the 2,048-shot level of
          release descriptor-information-v1 (asset shot-sweep-v1), byte-equal to
          the regenerated campaign dataset; its rows are cached by the ladder's
          own ``prepare_fresh_dataset``, and its dataset hash must equal the one
          analysis A recorded for the S0 ladder. It writes the N3 coupling-noise table (S0
          draws for S0 circuits, new draws for target circuits) and the R4
          encoder cache. No fit.
run       Fits F and C at R0, N3, R4, R5 and learner seeds 1 to 20 through
          ``descriptor_common.run_liao_fit``, after the archived R0 gate. Then
          compares each fit's validation predictions with the archived S0 ladder
          fit of the same key, rung, arm, and learner seed (pairing check).
analyze   Analysis A of rule 2026-10-02 on each shift's fits, plus the shift
          report beside the archived S0 values of the same rows and rungs.

Usage
-----
  PYTHONPATH=. python tools/shift_transfer.py generate --root DATA [--shifts S4 S2]
      [--seeds 101 211 307]
  MLQEM_PATH=<ml-qem at b1eccf8>/docs/tutorials/mlp.py PYTHONPATH=. \
      python tools/shift_transfer.py prepare --root DATA --s0-data LEVEL2048 \
      --out OUT [--workers 8]
  PYTHONPATH=. python tools/shift_transfer.py run --out OUT --s0-fits FITS
      [--workers 8] [--dry-run]
  PYTHONPATH=. python tools/shift_transfer.py analyze --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools import descriptor_common as common  # noqa: E402
from tools import descriptor_ladder as ladder  # noqa: E402

RULE_FILE = "docs/frozen-rules/2026-10-06-shift-transfer.md"
SHIFTS = {
    "S4": ("family_native_depth", [6]),
    "S2": ("noise_strength", ["L4"]),
}
# Seed stream code of each shift's target-circuit coupling noise (N3).
SHIFT_CODE = {"S2": 2, "S4": 4}
SEEDS = (101, 211, 307)
SIZE = 640
RUNGS = ("R0", "N3", "R4", "R5")
ARMS = ("F", "C")
LEARNER_SEEDS = tuple(range(1, 21))
TARGET_CIRCUITS_PER_FAMILY = 160
GATE_FILE = "artifacts/descriptor-information/gate/gate_r0.json"
# Dataset hashes of the prototype artifacts (rule, Known When This Rule Was
# Written); the governed regeneration must reproduce them.
PROTOTYPE_HASHES = {
    ("S4", 101): "1c0c027c753c36e04a10b9f3924fa24ddece42618f9cbe220c8fbc5a05442fdc",
    ("S2", 101): "1e605e7461c1f5f70d06cdf8e6d0ecd0fea75208965ec4b767d0294899874520",
}
# Dataset hashes of the S0 ladder's test caches (analysis-a.json, caches_used).
S0_DATASET_HASHES = {
    "shipped-s101-n640": "668df524395f3fe0ebeb49591916528f21bd29065c71e2fc7b5b785b652ece16",
    "shipped-s211-n640": "bd38104ecb8370b5b86524993793a36ecb45a68adf3dd0a333c5a5ec4ce40c88",
    "shipped-s307-n640": "05babeea98e0a8e748074325fb9d84c990162376348621406cca174998bf054f",
}
S0_ANALYSIS = "artifacts/descriptor-information/analysis-a.json"
# Fields that identify a source row physically; source identity requires each
# to be equal in the artifact and the S0 cache.
IDENTITY_FIELDS = (
    "family", "split", "domain", "instance", "severity", "observable", "shots",
    "n_qubits", "circuit_id", "circuit_hash", "circuit_sidecar", "steps", "dt",
    "j", "h", "jx", "jy", "jz", "two_qubit_gates", "transpiled_depth",
    "obs_locality", "ideal_expectation", "noisy_expectation", "noisy_stderr",
    "counts_hash",
)


def key_for(seed: int) -> str:
    return f"shipped-s{seed}-n{SIZE}"


def shift_split_spec(shift: str):
    """The campaign's S0 specification with one axis moved from fixed to source/target."""
    from qemscore.campaign.design import BASELINE_REGIME, campaign_split_spec
    from qemscore.datasets.splits import SplitSpec

    axis, target = SHIFTS[shift]
    base = campaign_split_spec(BASELINE_REGIME, SIZE).to_dict()
    fixed = dict(base["fixed_axes"])
    source = fixed.pop(axis)
    spec = dict(base)
    spec.pop("split_axis", None)
    spec.update(split_id=shift, source_domain={axis: list(source)},
                target_domain={axis: list(target)}, fixed_axes=fixed)
    return SplitSpec.from_dict(spec)


def artifact_dir(root: Path, shift: str, seed: int) -> Path:
    return Path(root) / shift / key_for(seed)


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def cmd_generate(args) -> int:
    from qemscore.datasets.split_generate import generate_split
    from qemscore.validation import validate_split_artifact

    for shift in args.shifts:
        spec = shift_split_spec(shift)
        for seed in args.seeds:
            target = artifact_dir(args.root, shift, seed)
            if target.exists():
                print(f"{shift} {seed}: already generated, skipping", flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            generate_split(spec, target, master_seed=int(seed))
            rows, manifest = validate_split_artifact(target)
            print(f"{shift} {seed}: {manifest['dataset_hash']}, {len(rows)} rows, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def _row_key(row: dict) -> tuple:
    return (str(row["family"]), str(row["split"]), int(row["instance"]),
            str(row["severity"]), str(row["observable"]))


def source_identity(artifact_rows: list[dict], s0: dict) -> dict:
    """Field-by-field comparison of the artifact's train and validation rows with S0's."""
    report = {}
    for role in ("train", "validation"):
        mine = {_row_key(r): r for r in artifact_rows if r["split"] == role}
        theirs = {_row_key(r): r for r in s0[role]}
        mismatched = {}
        for key in set(mine) & set(theirs):
            for field in IDENTITY_FIELDS:
                if mine[key].get(field) != theirs[key].get(field):
                    mismatched[field] = mismatched.get(field, 0) + 1
        report[role] = {
            "n_artifact": len(mine), "n_s0": len(theirs),
            "same_keys": set(mine) == set(theirs),
            "mismatches_by_field": mismatched,
        }
    report["identical"] = all(
        entry["same_keys"] and not entry["mismatches_by_field"]
        and entry["n_artifact"] == entry["n_s0"]
        for entry in (report["train"], report["validation"]))
    return report


def target_structure(shift: str, test_rows: list[dict], s0: dict | None = None) -> dict:
    """The target rows hold the target value of the moved axis and S0's other values.

    With ``s0``, it also checks the circuits: S2 keeps exactly S0's test
    circuits (only their noise changes); S4's deeper circuits are all new.
    """
    axis, target = SHIFTS[shift]
    families = sorted({str(r["family"]) for r in test_rows})
    circuits = {f: len({r["circuit_id"] for r in test_rows if r["family"] == f})
                for f in families}
    steps = sorted({int(r["steps"]) for r in test_rows})
    severities = sorted({str(r["severity"]) for r in test_rows})
    expected_steps = [6] if shift == "S4" else [3]
    expected_sev = ["L1", "L3"] if shift == "S4" else ["L4"]
    ok = (families == ["heisenberg", "tfi"]
          and all(n == TARGET_CIRCUITS_PER_FAMILY for n in circuits.values())
          and steps == expected_steps and severities == expected_sev
          and all(r["domain"] == "target" for r in test_rows))
    relation = None
    if s0 is not None:
        target_ids = {str(r["circuit_id"]) for r in test_rows}
        s0_test = {str(r["circuit_id"]) for r in s0["test"]}
        s0_all = s0_test | {str(r["circuit_id"]) for role in ("train", "validation")
                            for r in s0[role]}
        relation = {"in_s0_test": len(target_ids & s0_test),
                    "in_s0_any": len(target_ids & s0_all), "n_target": len(target_ids)}
        ok = ok and (target_ids == s0_test if shift == "S2" else not target_ids & s0_all)
    return {"axis": axis, "target": target, "families": families,
            "circuits_per_family": circuits, "steps": steps,
            "severities": severities, "n_rows": len(test_rows),
            "circuits_vs_s0": relation, "passed": bool(ok)}


def target_noise_table(s0: dict, test_rows: list[dict], seed: int, shift: str) -> dict:
    """N-rung z: S0's draws for every S0 circuit; new draws for new target circuits.

    A target circuit that is an S0 circuit (S2 keeps S0's test circuits) keeps
    its S0 draw. A new target circuit's z is ``default_rng(SeedSequence(
    [20261002, seed, index], spawn_key=(1000 + shift_code,)))
    .standard_normal(n_couplings)``, index being its position in the sorted
    target circuit_ids. The spawn key keeps these streams apart from S0's
    ``SeedSequence([20261002, seed, index])``. A target circuit that is an S0
    training or validation circuit is refused.
    """
    s0_table = ladder.coupling_noise_table(s0)
    family_by_circuit = {}
    for row in test_rows:
        family_by_circuit.setdefault(str(row["circuit_id"]), str(row["family"]))
    source_circuits = {str(row["circuit_id"])
                       for role in ("train", "validation") for row in s0[role]}
    overlap = set(family_by_circuit) & source_circuits
    if overlap:
        raise SystemExit(f"{len(overlap)} target circuits occur in S0 source rows")
    s0_z = {str(c["circuit_id"]): tuple(c["z"]) for c in s0_table["circuits"]}
    circuits = sorted(family_by_circuit)
    z_by_circuit = {}
    for index, circuit in enumerate(circuits):
        if circuit in s0_z:
            z_by_circuit[circuit] = s0_z[circuit]
            continue
        family = family_by_circuit[circuit]
        rng = np.random.default_rng(np.random.SeedSequence(
            [common.RULE_SEED, int(seed), index], spawn_key=(1000 + SHIFT_CODE[shift],)))
        z_by_circuit[circuit] = tuple(
            float(v) for v in rng.standard_normal(len(ladder.COUPLINGS[family])))
    z_by_item = dict(s0_table["z_by_item"])
    for row in test_rows:
        z_by_item[str(row["item_id"])] = z_by_circuit[str(row["circuit_id"])]
    return {
        "s0_circuits": s0_table["circuits"],
        "target_circuits": [
            {"circuit_index": i, "circuit_id": c, "family": family_by_circuit[c],
             "couplings": list(ladder.COUPLINGS[family_by_circuit[c]]),
             "z": list(z_by_circuit[c]), "s0_circuit": c in s0_z}
            for i, c in enumerate(circuits)],
        "z_by_item": z_by_item,
    }


def s0_cache(s0_dir: Path, cache_dir: Path) -> Path:
    """Cache one S0 dataset with the ladder's own fresh-dataset preparation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = ladder.PRIMARY_KEY.search(s0_dir.name).group(1)
    path = cache_dir / f"{key}.pkl"
    if not path.exists():
        ladder.prepare_fresh_dataset(s0_dir, cache_dir)
    return path


def _z_path(out: Path, key: str) -> Path:
    return out / "descriptors" / f"{key}__coupling_noise.json"


def cmd_prepare(args) -> int:
    from qemscore.runner.run import _prediction_items
    from qemscore.validation import validate_split_artifact

    for shift in args.shifts:
        record = {"schema": "shift-transfer-prepare-v1", "frozen_rule": RULE_FILE,
                  "shift": shift, "datasets": {}}
        out = Path(args.out).resolve() / shift
        (out / "cache").mkdir(parents=True, exist_ok=True)
        expected_spec = shift_split_spec(shift).to_dict()
        for seed in args.seeds:
            key = key_for(seed)
            data_dir = artifact_dir(args.root, shift, seed).resolve()
            rows, manifest = validate_split_artifact(data_dir)
            if manifest["split_spec"] != expected_spec or int(manifest["master_seed"]) != seed:
                raise SystemExit(f"{shift} {seed}: the artifact does not match the frozen specification")
            expected_hash = PROTOTYPE_HASHES.get((shift, seed))
            if expected_hash is not None and manifest["dataset_hash"] != expected_hash:
                raise SystemExit(f"{shift} {seed}: the artifact differs from the prototype's")
            s0_dir = (Path(args.s0_data) / f"regen-{key}").resolve()
            s0_path = s0_cache(s0_dir, Path(args.out).resolve() / "s0-cache")
            s0 = common.load_cache(s0_path)
            if str(s0["dataset_hash"]) != S0_DATASET_HASHES[key]:
                raise SystemExit(f"{key}: the S0 dataset hash differs from the S0 ladder's")
            identity = source_identity(rows, s0)
            test_rows = [r for r in rows if r["split"] == "test"]
            structure = target_structure(shift, test_rows, s0)
            entry = {"data_dir": str(data_dir),
                     "items_jsonl_sha256": common.sha256_file(data_dir / "items.jsonl"),
                     "dataset_hash": str(manifest["dataset_hash"]),
                     "s0_data_dir": str(s0_dir),
                     "s0_items_jsonl_sha256": common.sha256_file(s0_dir / "items.jsonl"),
                     "s0_cache_sha256": common.sha256_file(s0_path),
                     "s0_dataset_hash": str(s0["dataset_hash"]),
                     "source_identity": identity, "target_structure": structure}
            record["datasets"][f"{shift}/{key}"] = entry
            if not identity["identical"] or not structure["passed"]:
                common.write_json(out / "prepare.json", record)
                raise SystemExit(f"{shift} {seed}: source identity or target structure failed; "
                                 "see prepare.json (the rule stops here)")
            payload = {
                "key": key, "seed": int(seed), "shift": shift,
                "train": s0["train"], "validation": s0["validation"], "test": test_rows,
                "prediction_rows": {"validation": s0["prediction_rows"]["validation"],
                                    "test": _prediction_items(test_rows)},
                "dataset_hash": str(manifest["dataset_hash"]),
                "s0_dataset_hash": str(s0["dataset_hash"]),
            }
            cache_path = ladder._cache_path(out, key)
            tmp = cache_path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            (out / "cache" / f"{key}.order.json").write_text(json.dumps({
                "test_item_ids": [str(r["item_id"]) for r in test_rows],
                "validation_item_ids": [str(r["item_id"]) for r in s0["validation"]],
            }), encoding="utf-8")
            table = target_noise_table(s0, test_rows, seed, shift)
            common.write_json(_z_path(out, key), {
                "rule": RULE_FILE, "dataset_seed": int(seed), "shift": shift,
                "s0_circuits": table["s0_circuits"],
                "target_circuits": table["target_circuits"]})
            meta = ladder.prepare_encoding(key, payload, data_dir, out, args.workers)
            entry["encoder"] = {k: meta[k] for k in (
                "n_circuits", "upstream_commit", "released_entry_point_agrees",
                "structure_mismatches")}
            entry["cache_sha256"] = common.sha256_file(cache_path)
            print(f"{shift} {key}: source identical, {structure['n_rows']} target rows, "
                  f"{meta['n_circuits']} encoded circuits", flush=True)
        common.write_json(out / "prepare.json", record)
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


_BUILDERS: dict = {}


def _builder(out: Path, key: str, rung: str, data: dict):
    cache_key = (str(out), key, rung)
    if cache_key in _BUILDERS:
        return _BUILDERS[cache_key]
    if rung == "R0":
        built = (ladder.build_features, tuple(ladder.FEATURES))
    elif rung == "N3":
        stored = json.loads(_z_path(out, key).read_text(encoding="utf-8"))
        z_circuit = {c["circuit_id"]: tuple(c["z"])
                     for c in [*stored["s0_circuits"], *stored["target_circuits"]]}
        z_by_item = {str(r["item_id"]): z_circuit[str(r["circuit_id"])]
                     for r in ladder._all_rows(data)}
        builder = ladder.RungBuilder("N3", z_by_item=z_by_item)
        built = (builder, builder.names)
    elif rung == "R4":
        vectors, names = ladder._load_encoding(out, key)
        circuit_by_item = {str(r["item_id"]): str(r["circuit_id"])
                           for r in ladder._all_rows(data)}
        builder = ladder.RungBuilder("R4", encoding_by_circuit=vectors,
                                     circuit_by_item=circuit_by_item, encoder_names=names)
        built = (builder, builder.names)
    elif rung == "R5":
        builder = ladder.RungBuilder("R5")
        built = (builder, builder.names)
    else:
        raise ValueError(f"rung {rung!r} is outside the rule")
    _BUILDERS[cache_key] = built
    return built


def run_job(job: dict) -> dict:
    out = Path(job["out"])
    data = common.load_cache(job["cache"])
    builder, names = _builder(out, job["key"], job["rung"], data)
    return common.run_liao_fit(job, data, builder, names)


def build_jobs(out: Path) -> list[dict]:
    jobs = []
    for seed in SEEDS:
        for rung in RUNGS:
            for arm in ARMS:
                for k in LEARNER_SEEDS:
                    jobs.append(ladder._job(out, out / "fits", key_for(seed), rung, arm,
                                            learner_seed=k))
    return jobs


def pairing_check(out: Path, s0_fits: Path) -> dict:
    """Each fit's validation predictions against the archived S0 ladder fit, bit for bit."""
    exact, differs, missing = 0, [], []
    for job in build_jobs(out):
        stem = job["stem"]
        mine = out / "fits" / f"{stem}.npz"
        theirs = Path(s0_fits) / f"{stem}.npz"
        if not mine.exists() or not theirs.exists():
            missing.append(stem)
            continue
        a, b = np.load(mine), np.load(theirs)
        same = all(name in a.files and name in b.files and np.array_equal(a[name], b[name])
                   for name in ("validation", "validation__random_forest", "validation__mlp"))
        if same:
            exact += 1
        else:
            differs.append(stem)
    return {"n_exact": exact, "differs": differs, "missing": missing,
            "all_exact": not differs and not missing}


def cmd_run(args) -> int:
    for shift in args.shifts:
        out = Path(args.out).resolve() / shift
        jobs = build_jobs(out)
        if args.dry_run:
            common.print_plan(jobs, out / "fits", [])
            continue
        common.require_gate(_REPO / GATE_FILE)
        common.write_json(out / "run_config.json", {
            "frozen_rule": RULE_FILE, "rule_sha256": common.sha256_file(_REPO / RULE_FILE),
            "shift": shift, "rungs": list(RUNGS), "arms": list(ARMS),
            "learner_seeds": list(LEARNER_SEEDS), "workers": args.workers,
            "n_jobs": len(jobs), "started_unix": time.time()})
        info = common.drive(jobs, run_job, out, args.workers)
        common.write_json(out / f"run_{int(time.time())}.json", info)
        if info["crashes"]:
            return 3
        check = pairing_check(out, Path(args.s0_fits))
        common.write_json(out / "pairing_check.json", check)
        print(f"{shift}: pairing check {check['n_exact']} exact, "
              f"{len(check['differs'])} differ, {len(check['missing'])} missing", flush=True)
        if not check["all_exact"]:
            return 4
    return 0


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def _cell(entry: dict) -> dict:
    means = entry["means"]
    point = lambda name: (means.get(name) or {}).get("point")  # noqa: E731
    c, f, r = point("C"), point("F"), point("R")
    return {"C": c, "F": f, "R": r, "D": entry["D"]["point"],
            "D_interval": entry["D"].get("interval"),
            "D_over_C": entry["D_over_C"]["point"],
            "D_over_C_interval": entry["D_over_C"].get("interval"),
            "F_minus_R": None if f is None or r is None else f - r,
            "C_minus_R": None if c is None or r is None else c - r,
            "label": entry["classification"]["label"]}


def shift_report(analyses: dict, s0: dict) -> dict:
    rows = {}
    s0_rows = s0["parts"]["A"]["rows"]
    for shift, analysis in analyses.items():
        for label, row in analysis["parts"]["A"]["rows"].items():
            for rung in RUNGS:
                if rung not in row["rungs"]:
                    continue
                shifted = _cell(row["rungs"][rung])
                reference = _cell(s0_rows[label]["rungs"][rung])
                rows[f"{shift}/{label}/{rung}"] = {
                    "shift": shift, "row": label, "rung": rung,
                    "shifted": shifted, "s0": reference,
                    "change_C": shifted["C"] - reference["C"],
                    "change_F": shifted["F"] - reference["F"],
                    "change_D_over_C": shifted["D_over_C"] - reference["D_over_C"]}
    statements = {shift: analysis["parts"]["A"]["rung_statements"]
                  for shift, analysis in analyses.items()}
    return {"schema": "shift-transfer-report-v1", "frozen_rule": RULE_FILE,
            "rule_sha256": common.sha256_file(_REPO / RULE_FILE),
            "s0_reference": S0_ANALYSIS,
            "s0_reference_sha256": common.sha256_file(_REPO / S0_ANALYSIS),
            "rung_statements": statements,
            "s0_rung_statements": s0["parts"]["A"]["rung_statements"],
            "cells": rows}


def cmd_analyze(args) -> int:
    from tools import descriptor_ladder_analysis as analysis_a

    analyses = {}
    for shift in args.shifts:
        out = Path(args.out).resolve() / shift
        check = json.loads((out / "pairing_check.json").read_text(encoding="utf-8"))
        if not check["all_exact"]:
            raise SystemExit(f"{shift}: the pairing check did not pass; the rule stops here")
        result = analysis_a.analyze([out / "fits"], [out / "cache"],
                                    draws=10_000, seed=common.RULE_SEED, verbose=False)
        result["shift_transfer"] = {"shift": shift, "frozen_rule": RULE_FILE,
                                    "pairing_check": check}
        common.write_json(out / "analysis-a.json", result)
        analyses[shift] = result
    s0 = json.loads((_REPO / S0_ANALYSIS).read_text(encoding="utf-8"))
    report = shift_report(analyses, s0)
    common.write_json(Path(args.out).resolve() / "shift-report.json", report)
    for name, cell in report["cells"].items():
        sh, ref = cell["shifted"], cell["s0"]
        print(f"{name:42s} C {sh['C']:.4f} ({ref['C']:.4f}) F {sh['F']:.4f} ({ref['F']:.4f}) "
              f"R {sh['R']:.4f} D/C {sh['D_over_C']:+.3f} ({ref['D_over_C']:+.3f}) "
              f"{sh['label']} ({ref['label']})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def shifts_and_seeds(p):
        p.add_argument("--shifts", nargs="+", default=list(SHIFTS), choices=list(SHIFTS))
        p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))

    gen = sub.add_parser("generate")
    gen.add_argument("--root", required=True, type=Path)
    shifts_and_seeds(gen)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", required=True, type=Path)
    prep.add_argument("--s0-data", required=True, type=Path)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--workers", type=int, default=8)
    shifts_and_seeds(prep)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--s0-fits", required=True, type=Path)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--shifts", nargs="+", default=list(SHIFTS), choices=list(SHIFTS))
    ana = sub.add_parser("analyze")
    ana.add_argument("--out", required=True, type=Path)
    ana.add_argument("--shifts", nargs="+", default=list(SHIFTS), choices=list(SHIFTS))
    args = parser.parse_args(argv)
    if args.command in ("prepare",) and sorted(args.seeds) != sorted(SEEDS):
        raise SystemExit("prepare covers the rule's three dataset seeds")
    return {"generate": cmd_generate, "prepare": cmd_prepare, "run": cmd_run,
            "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
