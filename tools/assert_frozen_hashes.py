"""Fail with explicit expected and regenerated frozen dataset hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_HASHES = {
    "t0-micro": "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7",
    "t0-smoke": "edf5837e0da10f1dc93151d5b29d1855f66ad76341ff6c28019096308f8fcdf3",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    mismatches: list[str] = []
    for preset, expected in EXPECTED_HASHES.items():
        manifest_path = args.root / preset / "manifest.json"
        if not manifest_path.is_file():
            mismatches.append(f"{preset}: manifest is missing at {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = str(manifest.get("dataset_hash", "<missing>"))
        print(f"{preset}: expected {expected}, regenerated {actual}")
        if actual != expected:
            mismatches.append(
                f"{preset} frozen hash drift: expected {expected}, regenerated {actual}"
            )

    if mismatches:
        raise SystemExit("\n".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
