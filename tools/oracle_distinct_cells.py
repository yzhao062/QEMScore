"""Distinct-cell counts of the Bayes oracle's confusion tables (post hoc).

Reads ``artifacts/descriptor-information/round9/bayes-oracle.json`` and counts
each pipeline label against the oracle's reading under two stated identity
rules, writing ``artifacts/descriptor-information/round9/oracle-distinct-cells.json``.

- ``fit_identity``: every fit set except the derived ones. A derived set
  (name ending in ``_derived``) reuses the fits of another set, so its cells
  repeat cells counted elsewhere.
- ``design``: as ``fit_identity``, and the 2,048-shot level of the shot sweep
  (``sweep_2048``) is also dropped. It refits the strength-indicator design at
  2,048 shots on another platform, so its cells repeat that design's cells.

Counting follows the paper's confusion table: "information" cells read "F beats
C" (detection) or "not distinguished" (miss); "no information" cells (R0) read
"F beats C" (false positive) or anything else (other); "undetermined" cells
are counted whatever their label.

Usage: PYTHONPATH=. python tools/oracle_distinct_cells.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "artifacts/descriptor-information/round9/bayes-oracle.json"
OUTPUT = REPO / "artifacts/descriptor-information/round9/oracle-distinct-cells.json"

RULES = {
    "all": lambda name: True,
    "fit_identity": lambda name: not name.endswith("_derived"),
    "design": lambda name: not name.endswith("_derived") and name != "sweep_2048",
}


def tally(tables: dict, keep) -> dict:
    counts = {"fit_sets": 0, "cells": 0, "detections": 0, "misses": 0,
              "wrong_signs": 0, "other_no_information": 0, "false_positives": 0,
              "undetermined": 0}
    for name, table in tables.items():
        if not keep(name):
            continue
        matrix = table["matrix"]
        counts["fit_sets"] += 1
        counts["cells"] += int(table["total_cells"])
        counts["detections"] += matrix["information"]["F beats C"]
        counts["misses"] += matrix["information"]["not distinguished"]
        counts["wrong_signs"] += matrix["information"]["C beats F"]
        counts["false_positives"] += matrix["no information"]["F beats C"]
        counts["other_no_information"] += (matrix["no information"]["not distinguished"]
                                           + matrix["no information"]["C beats F"])
        counts["undetermined"] += sum(matrix["undetermined"].values())
    counts["informative"] = counts["detections"] + counts["misses"] + counts["wrong_signs"]
    return counts


def main() -> int:
    raw = SOURCE.read_bytes()
    tables = json.loads(raw)["confusion_tables"]
    record = {
        "schema": "oracle-distinct-cells-v1",
        "post_hoc": True,
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "rules": {
            "all": "every fit set",
            "fit_identity": "drop derived fit sets (they reuse another set's fits)",
            "design": "also drop sweep_2048 (a refit of the strength-indicator design)",
        },
        "dropped": {rule: sorted(n for n in tables if not keep(n)) for rule, keep in RULES.items()},
        "counts": {rule: tally(tables, keep) for rule, keep in RULES.items()},
    }
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for rule, counts in record["counts"].items():
        print(rule, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
