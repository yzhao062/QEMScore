"""Generate, fit, and analyze the surrogate campaign.

Three subcommands, in the order they run:

    generate   build the frozen split-v2 artifact for each requested setting
    fit        assert the realized structure, then write one record per setting
    roster     score each setting with the shipped roster, ZNE among it
    analyze    read only those records and emit the report and its tables

Splitting fit from analyze is the point rather than an ergonomic choice. Every
statistic the manuscript prints comes out of `analyze`, which never fits and
never touches a dataset, so a printed number traces to a file that names the
dataset hash and code revision behind it. Artifacts go under --root, which
should be a short absolute path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.campaign.analysis import (
    assert_campaign_structure,
    build_campaign_tables,
    build_setting_record,
    evaluate_campaign,
)
from qem_bench.campaign.design import (
    BOOTSTRAP_RESAMPLES,
    REGIMES,
    SEEDS,
    SIZES,
    campaign_setting_keys,
    campaign_split_spec,
    setting_key,
)
from qem_bench.datasets.split_generate import generate_split
from qem_bench.runner.run import run, validate_run_artifact
from qem_bench.validation import validate_split_artifact


def _parse_setting(key: str) -> tuple[str, int, int]:
    """Turn ``shipped-s101-n160`` back into its three fields."""
    for regime in REGIMES:
        for seed in SEEDS:
            for size in SIZES:
                if setting_key(regime, seed, size) == key:
                    return regime, seed, size
    raise argparse.ArgumentTypeError(f"unknown setting {key!r}")


def _parse_rehearsal(value: str) -> dict[str, int]:
    """Read ``SEED:TRAIN:VALIDATION:TEST`` per-family counts for a rehearsal.

    The plan requires a rehearsal on a separate seed, limited to throughput and
    end-to-end checks. Its scores do not enter the paper, and nothing here has to
    remember that: a seed outside the frozen three marks every record it produces
    as a rehearsal, and `evaluate_campaign` refuses to give one a publication
    path.
    """
    try:
        seed, train, validation, test = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--rehearsal takes SEED:TRAIN:VALIDATION:TEST, all integers"
        ) from error
    if seed in SEEDS:
        raise argparse.ArgumentTypeError(
            f"a rehearsal runs on a separate seed; {seed} is a campaign seed")
    if min(train, validation, test) < 2:
        raise argparse.ArgumentTypeError(
            "each role needs at least two circuits per family to resample")
    return {"seed": seed, "train": train, "validation": validation, "test": test}


def _counts(args) -> dict[str, int] | None:
    if args.rehearsal is None:
        return None
    return {role: args.rehearsal[role] for role in ("train", "validation", "test")}


def _requested(args) -> list[tuple[str, int, int]]:
    if args.rehearsal is not None:
        if args.setting:
            raise SystemExit("--setting names frozen settings; --rehearsal replaces it")
        seed, size = args.rehearsal["seed"], args.rehearsal["train"]
        return [(regime, seed, size) for regime in REGIMES]
    keys = args.setting or list(campaign_setting_keys())
    return [_parse_setting(key) for key in keys]


def _code_revision(args) -> str:
    if args.code_revision:
        return args.code_revision
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        raise SystemExit(
            "could not resolve the code revision; pass --code-revision explicitly"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return f"{revision}-dirty" if dirty else revision


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _data_dir(root: Path, key: str) -> Path:
    return root / "data" / key


def _record_path(root: Path, key: str) -> Path:
    return root / "records" / f"{key}.json"


def generate(args) -> None:
    start = time.perf_counter()
    for regime, seed, size in _requested(args):
        key = setting_key(regime, seed, size)
        target = _data_dir(args.root, key)
        if target.exists():
            print(f"{key}: already generated, skipping", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        generate_split(
            campaign_split_spec(regime, size, counts=_counts(args)),
            target, master_seed=seed)
        rows, manifest = validate_split_artifact(target)
        print(
            f"{key}: {manifest['dataset_hash']}, {len(rows)} rows, "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )


def fit(args) -> None:
    start = time.perf_counter()
    requested = _requested(args)
    revision = _code_revision(args)
    loaded = []
    for regime, seed, size in requested:
        key = setting_key(regime, seed, size)
        target = _data_dir(args.root, key)
        if not target.exists():
            raise SystemExit(f"{key}: {target} does not exist; run generate first")
        rows, manifest = validate_split_artifact(target)
        expected = campaign_split_spec(
            regime, size, counts=_counts(args)).to_dict()
        if manifest["split_spec"] != expected or manifest["master_seed"] != seed:
            raise SystemExit(
                f"{key}: the artifact does not match the frozen specification"
            )
        loaded.append({
            "regime": regime, "seed": seed, "size": size,
            "items": rows, "manifest": manifest, "path": target,
        })
    print(f"loaded {len(loaded)} settings in {time.perf_counter() - start:.1f}s",
          flush=True)

    # Every structural check runs before the first fit, so a mis-generated
    # dataset is caught as a structure rather than inferred from a score.
    counts = _counts(args)
    structure = assert_campaign_structure(
        loaded,
        expected_by_size=None if counts is None else {
            setting["size"]: counts for setting in loaded},
    )
    _write_json(args.root / "structure.json", structure)
    print("structure: "
          f"{len(structure['cross_size_checks'])} cross-size, "
          f"{len(structure['cross_regime_checks'])} cross-regime checks passed",
          flush=True)

    for setting in loaded:
        key = setting_key(setting["regime"], setting["seed"], setting["size"])
        path = _record_path(args.root, key)
        if path.exists() and not args.overwrite:
            print(f"{key}: record exists, skipping", flush=True)
            continue
        record = build_setting_record(
            setting["items"], setting["manifest"],
            regime=setting["regime"], seed=setting["seed"], size=setting["size"],
            data_path=str(setting["path"].resolve()), code_revision=revision,
            expected=counts, n_resamples=args.resamples,
        )
        _write_json(path, record)
        gate = {name: value["status"] for name, value in record["gates"].items()}
        print(f"{key}: {json.dumps(gate)}, {time.perf_counter() - start:.1f}s",
              flush=True)


def roster(args) -> None:
    """Score each setting with the shipped eight-method roster.

    The endpoint arms are fitted in `fit`; this produces the descriptive results
    the paper also reports, zero-noise extrapolation among them, and writes the
    binding that ties each run artifact to the dataset it scored.
    """
    start = time.perf_counter()
    for regime, seed, size in _requested(args):
        key = setting_key(regime, seed, size)
        target = _data_dir(args.root, key)
        if not target.exists():
            raise SystemExit(f"{key}: {target} does not exist; run generate first")
        out = args.root / "rosters" / key
        binding_path = out / "binding.json"
        if binding_path.exists() and not args.overwrite:
            print(f"{key}: roster exists, skipping", flush=True)
            continue
        result = run(target, out)
        validate_run_artifact(result)
        _write_json(binding_path, {
            "setting": key,
            "artifact_id": result["artifact_id"],
            "dataset_hash": result["dataset_hash"],
            "dataset_manifest_sha256": result["dataset_manifest_sha256"],
            "methods": sorted(result["methods"]),
            "ledger_totals": {name: spec["ledger"]["total"]
                              for name, spec in result["methods"].items()},
        })
        print(f"{key}: roster {result['artifact_id']}, "
              f"{len(result['methods'])} methods, "
              f"{time.perf_counter() - start:.1f}s", flush=True)


def analyze(args) -> None:
    directory = args.root / "records"
    if not directory.exists():
        raise SystemExit(f"{directory} does not exist; run fit first")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if not records:
        raise SystemExit(f"no records under {directory}")
    rosters = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.root / "rosters").glob("*/binding.json"))
    }
    report = evaluate_campaign(records, rosters=rosters, n_resamples=args.resamples)
    _write_json(args.root / "report.json", report)
    _write_json(args.root / "tables.json", build_campaign_tables(report))
    primary = report["primary"]
    print(json.dumps({
        "records": len(records),
        "failed_settings": report["failed_settings"],
        "settings_without_a_roster": report["settings_without_a_roster"],
        "primary_successes": primary["successes"],
        "successful_seeds": primary["successful_seeds"],
        "publication_path": primary["publication_path"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--setting", action="append",
        help="repeatable; defaults to every setting in the frozen design")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--rehearsal", type=_parse_rehearsal,
        help="SEED:TRAIN:VALIDATION:TEST per-family counts for a throughput "
             "rehearsal on a separate seed; its records can never be published")
    parser.add_argument(
        "--code-revision",
        help="override the revision recorded in each record; resolved from git "
             "otherwise")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild records that already exist")
    parser.add_argument(
        "command", choices=("generate", "fit", "roster", "analyze"))
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    args.root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "command": args.command,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }), flush=True)
    {"generate": generate, "fit": fit, "roster": roster,
     "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
