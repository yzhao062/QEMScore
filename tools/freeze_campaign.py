"""Write the campaign manifest, and check later that nothing in it moved.

The plan's discipline is that the design, the promotion rule, the audit rules and
the code revision are fixed before any final score exists. A sentence saying so
is not checkable. This writes them into one dated file and gives that file a
content hash, so `verify` can state afterwards whether the run that produced the
numbers is the run the freeze describes.

Two subcommands:

    freeze   capture the resolved design, the audit rules, the environment and
             the code revision, plus every generated dataset's identity
    verify   re-read a manifest and report every field that no longer matches

`verify` reports rather than raises, because the useful output after a campaign
is the complete list of what drifted. It exits non-zero when anything did.

`freeze` refuses a dirty or unresolvable working tree. Everything else here
compares a recorded field against a recomputed one, and an uncommitted edit is
the one difference that comparison cannot see, so the manifest is worth checking
only if the revision it names is one that can be checked out again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.campaign.design import (
    campaign_setting_keys,
    declared_design,
    expected_circuits,
    role_counts,
)
from qem_bench.reproducibility import environment_contract
from qem_bench.validation import validate_split_artifact

SCHEMA_VERSION = "qem-bench-campaign-manifest-v1"
# Every package whose version can change a stored histogram or a fitted model.
_PINNED = ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "stim")


def _audit_rules() -> dict:
    """The audit's frozen constants, read from the module that enforces them."""
    from tools import audit_campaign

    return {
        "indices": {role: list(values)
                    for role, values in audit_campaign.AUDIT_INDICES.items()},
        "check_coverage": dict(audit_campaign.CHECK_COVERAGE),
        "tolerances": {
            "label": audit_campaign.LABEL_TOLERANCE,
            "estimate": audit_campaign.ESTIMATE_TOLERANCE,
            "prediction": audit_campaign.PREDICTION_TOLERANCE,
            "parameter": audit_campaign.PARAMETER_TOLERANCE,
            "independent_label": audit_campaign.INDEPENDENT_LABEL_TOLERANCE,
        },
        "exact_comparisons": ["histogram_replay", "structure",
                             "exhaustive_structure", "circuit_identity",
                             "redrawn_identity", "pool_descriptor"],
    }


def _versions() -> dict:
    import importlib.metadata as metadata

    resolved = {}
    for name in _PINNED:
        try:
            resolved[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            resolved[name] = None
    return resolved


def _code_revision(root: Path) -> dict:
    def run(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True, text=True, check=False)
        # Only the trailing newline is stripped. `git status --porcelain` puts
        # the status in the first two columns, so stripping whitespace would
        # eat the leading space of the first entry and `line[3:]` would then
        # drop a character from that path.
        return result.stdout.strip("\n") if result.returncode == 0 else None

    revision = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "revision": revision,
        "clean": dirty == "",
        "uncommitted_paths": sorted(
            line[3:] for line in (dirty or "").splitlines() if line[3:]),
    }


def _datasets(root: Path) -> dict:
    """Every generated artifact's identity, so the freeze names its inputs."""
    directory = root / "data"
    if not directory.exists():
        return {}
    resolved = {}
    for path in sorted(value for value in directory.iterdir() if value.is_dir()):
        _, manifest = validate_split_artifact(path)
        resolved[path.name] = {
            "dataset_hash": str(manifest["dataset_hash"]),
            "split_spec_hash": str(manifest["split_spec_hash"]),
            "master_seed": int(manifest["master_seed"]),
        }
    return resolved


def _payload(root: Path, repository: Path) -> dict:
    design = declared_design()
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign_root": str(root.resolve()),
        "expected_settings": list(campaign_setting_keys()),
        "design": design,
        "role_counts": {str(size): role_counts(size)
                        for size in design["sizes"]},
        "circuits_per_family": {str(size): expected_circuits(size)
                                for size in design["sizes"]},
        "audit_rules": _audit_rules(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "contract": environment_contract(),
        },
        "code": _code_revision(repository),
        "datasets": _datasets(root),
    }


def _digest(payload: dict) -> str:
    """Hash everything except the timestamp and the digest itself."""
    body = {key: value for key, value in payload.items()
            if key not in ("frozen_utc", "manifest_sha256")}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _differences(recorded, current, path=""):
    """Every leaf that changed, named by its path through the manifest."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        found = []
        for key in sorted(set(recorded) | set(current)):
            found.extend(_differences(
                recorded.get(key), current.get(key),
                f"{path}.{key}" if path else str(key)))
        return found
    if recorded != current:
        return [{"field": path, "frozen": recorded, "current": current}]
    return []


def freeze(args) -> int:
    payload = _payload(args.root, args.repository)
    # A dirty tree names a revision nobody can check out again, so the manifest
    # would freeze a revision string rather than the code that ran. Recording the
    # uncommitted bytes instead would be a second provenance format to maintain;
    # refusing the run is the cheaper half of that choice, and it is what lets
    # `run_campaign_analysis` treat a verified manifest as a checkable claim.
    if not payload["code"]["revision"] or not payload["code"]["clean"]:
        raise SystemExit("campaign freeze requires a clean committed revision")
    payload["manifest_sha256"] = _digest(payload)
    out = args.out or args.root / "campaign-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out)
    print(json.dumps({
        "written": str(out),
        "manifest_sha256": payload["manifest_sha256"],
        "code_revision": payload["code"]["revision"],
        "code_clean": payload["code"]["clean"],
        "datasets": len(payload["datasets"]),
    }, indent=2), flush=True)
    return 0


def verify(args) -> int:
    path = args.out or args.root / "campaign-manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run freeze first")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"expected schema_version {SCHEMA_VERSION!r}")

    stored = recorded.get("manifest_sha256")
    if stored != _digest(recorded):
        print(json.dumps({
            "manifest_intact": False,
            "detail": "the manifest's own content hash does not match its body",
        }, indent=2), flush=True)
        return 1

    current = _payload(args.root, args.repository)
    # The timestamp and the digest are properties of the freeze, not of what it
    # froze, so they are excluded from the comparison rather than reported.
    drift = _differences(
        {key: value for key, value in recorded.items()
         if key not in ("frozen_utc", "manifest_sha256")},
        {key: value for key, value in current.items() if key != "frozen_utc"},
    )
    print(json.dumps({
        "manifest": str(path),
        "manifest_intact": True,
        "frozen_utc": recorded["frozen_utc"],
        "matches": not drift,
        "drift": drift,
    }, indent=2), flush=True)
    return 0 if not drift else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="campaign root holding data/ and records/")
    parser.add_argument("--out", type=Path,
                        help="manifest path (default <root>/campaign-manifest.json)")
    parser.add_argument(
        "--repository", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository whose revision the manifest records")
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    raise SystemExit({"freeze": freeze, "verify": verify}[args.command](args))


if __name__ == "__main__":
    main()
