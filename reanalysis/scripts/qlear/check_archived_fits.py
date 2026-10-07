"""Check archived Q-LEAR per-fit predictions against the published field report.

This script verifies SHA-256 checksums of the 41 archived prediction files.
It rescores the ladder under the frozen Hellinger metric.
It enforces the HEAD schema requirement that roster_status is complete.
It verifies all numerical scores match field-report.json to tolerance 1e-12.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(fits_dir: str, checksum_file: str) -> bool:
    """Verify that all files declared in checksum_file match on disk."""
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}", file=sys.stderr)
        return False
    with open(checksum_file, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    for line in lines:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0], parts[1].strip()
        full_path = os.path.join(fits_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.isfile(full_path):
            print(f"Missing file in archive: {full_path}", file=sys.stderr)
            return False
        actual_hash = compute_sha256(full_path)
        if actual_hash != expected_hash:
            print(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}", file=sys.stderr)
            return False
    return True


def compare_structures(a, b, tol=1e-12, path=""):
    """Compare two JSON-like nested objects with float tolerance."""
    if type(a) != type(b):
        return False, f"Type mismatch at {path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False, f"Key mismatch at {path}: {set(a.keys()) ^ set(b.keys())}"
        for k in a:
            ok, msg = compare_structures(a[k], b[k], tol=tol, path=f"{path}.{k}")
            if not ok:
                return False, msg
        return True, ""
    elif isinstance(a, list):
        if len(a) != len(b):
            return False, f"Length mismatch at {path}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, msg = compare_structures(x, y, tol=tol, path=f"{path}[{i}]")
            if not ok:
                return False, msg
        return True, ""
    elif isinstance(a, float):
        diff = abs(a - b)
        if diff > tol:
            return False, f"Float diff at {path}: {a} vs {b}, diff={diff:.2e} > {tol:.2e}"
        return True, ""
    else:
        if a != b:
            return False, f"Value mismatch at {path}: {a} vs {b}"
        return True, ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--fits",
        default=os.path.join(REPO_ROOT, "reanalysis", "outputs", "qlear", "fits"),
        help="Path to archived fits directory",
    )
    parser.add_argument(
        "--report",
        default=os.path.join(REPO_ROOT, "reanalysis", "outputs", "qlear", "field-report.json"),
        help="Path to packaged field-report.json",
    )
    parser.add_argument(
        "--checksums",
        default=os.path.join(REPO_ROOT, "reanalysis", "outputs", "qlear", "fits", "SHA256SUMS"),
        help="Path to SHA256SUMS file",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Tolerance for floating point comparisons (default: 1e-12)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Step 1: Verifying SHA-256 checksums of archived fits...")
    if not verify_checksums(args.fits, args.checksums):
        sys.exit(1)
    print("All checksums verified successfully.")

    print("\nStep 2: Rescoring ladder from archived fits...")
    from reanalysis.scripts.qlear import score_field_half

    with tempfile.TemporaryDirectory() as tmp_dir:
        rescore_out = os.path.join(tmp_dir, "field-report.json")
        saved_argv = sys.argv
        sys.argv = ["score_field_half.py", args.fits, rescore_out]
        try:
            score_field_half.main()
        finally:
            sys.argv = saved_argv

        print("\nStep 3: Comparing rescored report against packaged report...")
        with open(args.report, "r", encoding="utf-8") as fh:
            expected_report = json.load(fh)
        with open(rescore_out, "r", encoding="utf-8") as fh:
            actual_report = json.load(fh)

        # Enforce HEAD schema requirement: roster_status must be present and 'complete'
        for p in ("hardware", "simulator"):
            p_block = actual_report.get("panels", {}).get(p, {})
            if "roster_status" not in p_block:
                print(f"FAIL: HEAD schema violation: roster_status missing in panel {p}", file=sys.stderr)
                sys.exit(1)
            if p_block["roster_status"] != "complete":
                print(f"FAIL: Expected roster_status 'complete', got {p_block['roster_status']!r} in panel {p}", file=sys.stderr)
                sys.exit(1)

        # The packaged reference report field-report.json was serialized without
        # roster_status, whereas HEAD schema includes roster_status: 'complete'.
        # We verify that roster_status is produced under HEAD schema above,
        # then strip it for the value-by-value numerical comparison with the packaged report.
        actual_for_comp = json.loads(json.dumps(actual_report))
        for p in ("hardware", "simulator"):
            if "roster_status" not in expected_report.get("panels", {}).get(p, {}):
                actual_for_comp["panels"][p].pop("roster_status", None)

        ok, msg = compare_structures(expected_report, actual_for_comp, tol=args.tol)
        if not ok:
            print(f"FAIL: Reproduction check failed: {msg}", file=sys.stderr)
            sys.exit(1)

    print(f"PASS: Rescoring from archived fits reproduces field-report.json within tolerance {args.tol:.1e}.")


if __name__ == "__main__":
    main()
