"""Compare base and candidate dependency-sensitive snapshots."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def _changed_item_fields(baseline: dict, candidate: dict, field: str) -> int:
    changed = 0
    presets = set(baseline["datasets"]) | set(candidate["datasets"])
    for preset in presets:
        old_items = baseline["datasets"].get(preset, {}).get("items", {})
        new_items = candidate["datasets"].get(preset, {}).get("items", {})
        for item_id in set(old_items) | set(new_items):
            if old_items.get(item_id, {}).get(field) != new_items.get(item_id, {}).get(field):
                changed += 1
    return changed


def _write_report(path: Path, report: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")
    visible = (
        report
        if len(report) <= 420
        else report[:417] + ["... diff truncated; see artifact", "```", ""]
    )
    print("\n".join(visible))


def _record_initial_snapshot(candidate: dict, state: dict, output: Path) -> int:
    if state.get("baseline_contract") != "absent":
        raise ValueError("missing baseline snapshot requires an absent-contract state marker")
    missing_paths = state.get("missing_paths")
    if not isinstance(missing_paths, list) or not missing_paths:
        raise ValueError("absent-contract state marker must list missing_paths")
    dataset_hashes = {
        preset: payload["dataset_hash"]
        for preset, payload in sorted(candidate["datasets"].items())
    }
    report = [
        "# Dependency Drift Initial Snapshot",
        "",
        "The pull-request base predates the dependency-drift contract. The candidate",
        "snapshot is the initial recorded contract, so no base comparison is possible.",
        "",
        f"- Base commit: `{state.get('base_sha', '<unknown>')}`",
        f"- Missing base paths: {', '.join(f'`{path}`' for path in missing_paths)}",
        f"- Initial lock SHA-256: `{candidate['lock_sha256']}`",
        f"- Initial frozen dataset hashes: `{json.dumps(dataset_hashes, sort_keys=True)}`",
        f"- Initial folded circuits: {len(candidate['folded_circuits'])}",
        "",
    ]
    _write_report(output, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-state", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not args.baseline.is_file():
        if args.baseline_state is None or not args.baseline_state.is_file():
            raise FileNotFoundError(
                "baseline snapshot is missing without a first-introduction state marker"
            )
        state = json.loads(args.baseline_state.read_text(encoding="utf-8"))
        return _record_initial_snapshot(candidate, state, args.output)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    old_target = {
        "datasets": baseline["datasets"],
        "folded_circuits": baseline["folded_circuits"],
    }
    new_target = {
        "datasets": candidate["datasets"],
        "folded_circuits": candidate["folded_circuits"],
    }
    old_lines = json.dumps(old_target, indent=2, sort_keys=True).splitlines()
    new_lines = json.dumps(new_target, indent=2, sort_keys=True).splitlines()
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="committed-base",
            tofile="dependency-candidate",
            lineterm="",
        )
    )

    depth_changes = _changed_item_fields(baseline, candidate, "transpiled_depth")
    gate_changes = _changed_item_fields(baseline, candidate, "two_qubit_gates")
    folded_keys = set(baseline["folded_circuits"]) | set(candidate["folded_circuits"])
    folded_changes = sum(
        baseline["folded_circuits"].get(key) != candidate["folded_circuits"].get(key)
        for key in folded_keys
    )
    hash_changes = sum(
        baseline["datasets"].get(preset, {}).get("dataset_hash")
        != candidate["datasets"].get(preset, {}).get("dataset_hash")
        for preset in set(baseline["datasets"]) | set(candidate["datasets"])
    )

    report = [
        "# Dependency Drift Report",
        "",
        f"- Baseline lock SHA-256: `{baseline['lock_sha256']}`",
        f"- Candidate lock SHA-256: `{candidate['lock_sha256']}`",
        f"- Frozen dataset hashes changed: {hash_changes}",
        f"- `transpiled_depth` rows changed: {depth_changes}",
        f"- `two_qubit_gates` rows changed: {gate_changes}",
        f"- Folded circuits changed: {folded_changes}",
        "",
        "## Unified Diff",
        "",
        "```diff",
        *(diff or ["No dependency-sensitive structural drift."]),
        "```",
        "",
    ]
    _write_report(args.output, report)
    if diff:
        raise SystemExit("dependency-sensitive outputs changed; inspect the report artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
