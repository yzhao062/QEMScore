"""Generate, fit, and analyze the surrogate campaign.

Three subcommands, in the order they run:

    generate   build the frozen split-v2 artifact for each requested setting
    fit        assert the realized structure, then write one record per setting
    roster     score each setting with the shipped roster, ZNE among it
    analyze    read only those records and emit the report and its tables

Splitting fit from analyze is the point rather than an ergonomic choice. Every
statistic the manuscript prints comes out of `analyze`, which never fits, so a
printed number traces to a file that names the dataset hash and code revision
behind it. `analyze` reads each dataset once and for one purpose: a record
names a dataset hash, and the rows it retained have to be that dataset's
complete test projection rather than a subset of it. Artifacts go under --root,
which should be a short absolute path.

The rows are half of a record. The other half is what the fit produced: every
prediction map, every selected configuration, every diagnostic input, none of
which exists anywhere else because `analyze` never refits. So a campaign `fit`
verifies the pre-fit freeze before fitting anything and writes a receipt of the
complete record beside it, and `analyze` verifies both before it reads a number.
A rehearsal has no freeze and takes neither path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.campaign.analysis import (
    TEST_ROW_FIELDS,
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
    is_frozen_setting,
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
    regimes = [args.regime] if args.regime else list(REGIMES)
    if args.rehearsal is not None:
        if args.setting:
            raise SystemExit("--setting names frozen settings; --rehearsal replaces it")
        seed, size = args.rehearsal["seed"], args.rehearsal["train"]
        return [(regime, seed, size) for regime in regimes]
    keys = args.setting or list(campaign_setting_keys())
    resolved = [_parse_setting(key) for key in keys]
    return [value for value in resolved if value[0] in regimes]


def _code_revision(args) -> str:
    """Resolve the revision from Git, and let ``--code-revision`` only agree.

    The flag used to return before Git ran, which made it an override: passing
    the current HEAD erased the `-dirty` suffix the tree had earned, and every
    downstream restriction that reads the suffix went with it. Checking the flag
    against the resolved value keeps it useful as an assertion the caller makes
    about the tree and removes the one thing it could do that Git could not.
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        raise SystemExit(
            "could not resolve the code revision from git; a record cannot name "
            "a revision nothing here has read"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    resolved = f"{revision}-dirty" if dirty else revision
    if args.code_revision and args.code_revision != resolved:
        raise SystemExit("--code-revision disagrees with the actual working tree")
    return resolved


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


# --------------------------------------------------------------------------
# The pre-fit freeze, and the receipt each fit leaves behind
# --------------------------------------------------------------------------


def _record_digest(record: dict) -> str:
    """Hash the whole record, in an encoding that survives being written out.

    Sorted keys and tight separators make the digest independent of how
    `_write_json` happens to lay the file out, so the value computed from the
    record the fit built equals the value computed from the record `analyze`
    reads back.
    """
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fit_binding_path(root: Path, key: str) -> Path:
    return root / "fit-bindings" / f"{key}.json"


def _fit_binding(record: dict, manifest: dict) -> dict:
    """The receipt one frozen fit writes: what it produced, under what freeze."""
    return {
        "schema_version": "qem-bench-fit-binding-v1",
        "setting": record["setting"],
        "dataset_hash": record["dataset_hash"],
        "code_revision": record["code_revision"],
        "freeze_manifest_sha256": manifest["manifest_sha256"],
        "record_sha256": _record_digest(record),
    }


def _assert_fit_bindings(root: Path, records: list[dict], manifest: dict) -> None:
    """Check each record against the receipt its own fit left.

    `_assert_records_retain_whole_datasets` binds the retained rows to the
    artifact they claim, and the fitted outputs are the rest of the record:
    every prediction map, every selected configuration, every diagnostic input.
    None of them exists anywhere else, because `analyze` never refits, so
    copying one saved method's prediction map into another's slot moved the
    share across its whole range while the rows, the dataset hash, the code
    revision and the named configurations all stayed as they were. Comparing the
    record against a digest written when it was produced is what makes the
    fitted half as checkable as the retained half.

    The receipt is the fit's, not the analysis's: manufacturing one here from
    whatever record is present would authenticate the substitution instead of
    catching it, so a missing receipt is a refusal rather than a repair.
    """
    for record in records:
        path = _fit_binding_path(root, record["setting"])
        if not path.is_file():
            raise SystemExit(f"{path}: missing receipt from the frozen fit")
        binding = json.loads(path.read_text(encoding="utf-8"))
        if binding != _fit_binding(record, manifest):
            raise SystemExit(f"{path}: record differs from its frozen-fit receipt")


def _verified_campaign_manifest(root: Path) -> dict:
    """Re-check the pre-fit freeze, and return it for the receipts to name.

    The manifest was written and never read again: analysis ignored an absent
    one and a malformed one alike. Verifying it here is what turns the design,
    the audit rules, the environment and the code revision into claims that
    failed if they moved, rather than a file beside the results.
    """
    from tools.freeze_campaign import verify

    path = root / "campaign-manifest.json"
    if not path.is_file():
        raise SystemExit("campaign execution requires its pre-fit freeze")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest["code"]["clean"] or not manifest["code"]["revision"]:
        raise SystemExit("campaign execution requires a clean frozen revision")
    if verify(argparse.Namespace(
        root=root,
        repository=Path(__file__).resolve().parents[1],
        out=path,
    )) != 0:
        raise SystemExit("campaign differs from its recorded freeze")
    return manifest


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
    # A campaign fit runs against a verified freeze or it does not run. A
    # rehearsal has no freeze to check, which is why the check keys on the
    # rehearsal counts rather than on a flag a caller could pass.
    campaign_manifest = (
        _verified_campaign_manifest(args.root) if counts is None else None
    )
    if (campaign_manifest is not None
            and revision != campaign_manifest["code"]["revision"]):
        raise SystemExit("fit revision differs from the frozen revision")
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
        # `--overwrite` is a rehearsal convenience. Letting it replace a sealed
        # campaign fit would let a second fit inherit the first one's receipt,
        # which is the one thing the receipt exists to make impossible.
        if campaign_manifest is not None and _fit_binding_path(args.root, key).exists():
            raise SystemExit(f"{key}: sealed campaign fit cannot be overwritten")
        record = build_setting_record(
            setting["items"], setting["manifest"],
            regime=setting["regime"], seed=setting["seed"], size=setting["size"],
            data_path=str(setting["path"].resolve()), code_revision=revision,
            expected=counts, n_resamples=args.resamples,
        )
        _write_json(path, record)
        if campaign_manifest is not None:
            _write_json(
                _fit_binding_path(args.root, key),
                _fit_binding(record, campaign_manifest),
            )
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


def _assert_records_retain_whole_datasets(root: Path, records: list[dict]) -> None:
    """Check each record's test rows against the artifact whose hash it asserts.

    A record names its dataset by hash, and that name was the only thing tying
    the rows it carries to the artifact it claims. Dropping one observable from
    every circuit leaves the cross-size circuit comparison satisfied, changes no
    prediction value, and moves the share; the freeze records neither the
    retained rows nor their count, so nothing else notices. Comparing the
    retained projection against the validated artifact is what turns the
    record's asserted identity into one.

    This binds the rows and nothing else. A prediction substituted for a
    retained row is not visible here, because the record is the only place the
    fitted outputs exist and `analyze` does not refit.
    """
    for record in records:
        key = setting_key(
            str(record["regime"]), int(record["seed"]), int(record["size"])
        )
        if key != record["setting"]:
            raise ValueError("record setting disagrees with its fields")
        source_rows, source_manifest = validate_split_artifact(
            _data_dir(root, key)
        )
        if str(source_manifest["dataset_hash"]) != str(record["dataset_hash"]):
            raise ValueError(f"{key}: retained record names a different dataset")
        expected_rows = {
            str(row["item_id"]): {field: row[field] for field in TEST_ROW_FIELDS}
            for row in source_rows if row["split"] == "test"
        }
        retained_rows = {
            str(row["item_id"]): {field: row[field] for field in TEST_ROW_FIELDS}
            for row in record["test_items"]
        }
        if (len(retained_rows) != len(record["test_items"])
                or retained_rows != expected_rows):
            raise ValueError(
                f"{key}: retained test rows differ from the complete validated "
                "artifact; rebuild the record without filtering or relabeling rows"
            )


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
    # A missing audit is read rather than refused. The rehearsal runs analyze
    # before any audit exists, and the library already withholds the publication
    # path from an unaudited setting, so refusing here would only make the
    # rehearsal impossible without making the campaign safer.
    audit_path = args.root / "audit.json"
    audits = None
    if audit_path.is_file():
        from tools.audit_campaign import SCHEMA_VERSION as AUDIT_SCHEMA_VERSION

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise SystemExit(
                f"{audit_path}: expected schema_version {AUDIT_SCHEMA_VERSION!r}")
        audits = audit.get("settings")
    _assert_records_retain_whole_datasets(args.root, records)
    # A frozen record claims a campaign fit, so its freeze and its receipts have
    # to be there. A rehearsal claims neither and is analyzed as before. The
    # seed and size are read as well as the claim: a campaign record whose
    # `design_frozen` was cleared by hand still occupies a campaign key in the
    # reported grid, so clearing the flag must not be a way out of the check.
    campaign_manifest = None
    if any(record.get("design_frozen") is True
           or is_frozen_setting(int(record["seed"]), int(record["size"]))
           for record in records):
        campaign_manifest = _verified_campaign_manifest(args.root)
        if any(record["code_revision"] != campaign_manifest["code"]["revision"]
               for record in records):
            raise SystemExit("fitted records differ from the frozen revision")
        _assert_fit_bindings(args.root, records, campaign_manifest)
    report = evaluate_campaign(
        records, rosters=rosters, audits=audits, n_resamples=args.resamples)
    # The library reads records and opens no manifest, so it can only say that a
    # share is unverified. These three fields are what this layer verified, and
    # they ride on every share rather than on the report alone, because a
    # consumer reading one share would otherwise inherit none of it.
    report["freeze_verified"] = campaign_manifest is not None
    report["freeze_manifest_sha256"] = (
        None if campaign_manifest is None else campaign_manifest["manifest_sha256"]
    )
    report["fit_bindings_verified"] = campaign_manifest is not None
    for share in report["shares"].values():
        for field in ("freeze_verified", "freeze_manifest_sha256",
                      "fit_bindings_verified"):
            share[field] = report[field]
    _write_json(args.root / "report.json", report)
    _write_json(args.root / "tables.json", build_campaign_tables(report))
    primary = report["primary"]
    print(json.dumps({
        "records": len(records),
        "failed_settings": report["failed_settings"],
        "settings_without_a_roster": report["settings_without_a_roster"],
        "settings_without_an_audit": report["settings_without_an_audit"],
        "settings_failing_audit": report["settings_failing_audit"],
        "settings_without_a_zero_noise_replay":
            report["settings_without_a_zero_noise_replay"],
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
        help="assert the revision the working tree is on; it is compared against "
             "git rather than substituted for it, so a dirty tree stays dirty")
    parser.add_argument(
        "--regime", choices=sorted(REGIMES),
        help="restrict to one evolution-step regime; a maximum-size rehearsal "
             "calibrates throughput on one rather than paying for both")
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
