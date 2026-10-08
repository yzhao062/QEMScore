"""Numerical sensitivity of the Bayes oracle's readings (round-10 rule, Part B).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part B.

Reads the governed oracle (``artifacts/descriptor-information/round9/
bayes-oracle.json``) and the sensitivity runs of ``tools/bayes_oracle.py
--sensitivity`` (degree-7 surrogates, three further Monte Carlo streams, five
times the draws, the binomial likelihood, and a default-option rerun). For
every oracle cell outside R0 it reads G* and its class in each run:
"information" when the G* interval lies above zero, "undetermined"
otherwise. A cell is resolved when every run gives the governed class and
unresolved otherwise. The default-option rerun must reproduce the governed
risks exactly.

It then recounts the confusion table of the round-9 rule with unresolved cells
apart: for every fit-set row of the governed Equation 2 table, the matching
oracle cell (dataset seed, family, rung, shot level, strength variant) is
resolved or not, and the pipeline label is tallied against the resolved class.

Usage:
  PYTHONPATH=. python tools/oracle_sensitivity.py \\
      --run default=PATH --run degree7=PATH --run stream1=PATH ... \\
      --frozen-rule docs/frozen-rules/2026-10-07-round10-checks.md
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
RULE_FILE = "docs/frozen-rules/2026-10-07-round10-checks.md"
GOVERNED = _REPO / "artifacts/descriptor-information/round9/bayes-oracle.json"
DEFAULT_OUT = _REPO / "artifacts/descriptor-information/round10/oracle-sensitivity.json"
DEFAULT_MD = _REPO / "artifacts/descriptor-information/round10/oracle-sensitivity.md"
REQUIRED_RUNS = {
    "default": {"surrogate_degree": 5, "stream": None, "draw_multiplier": 1,
                "likelihood": "gaussian"},
    "degree7": {"surrogate_degree": 7, "stream": None, "draw_multiplier": 1,
                "likelihood": "gaussian"},
    "stream1": {"surrogate_degree": 5, "stream": 1, "draw_multiplier": 1,
                "likelihood": "gaussian"},
    "stream2": {"surrogate_degree": 5, "stream": 2, "draw_multiplier": 1,
                "likelihood": "gaussian"},
    "stream3": {"surrogate_degree": 5, "stream": 3, "draw_multiplier": 1,
                "likelihood": "gaussian"},
    "draws5x": {"surrogate_degree": 5, "stream": None, "draw_multiplier": 5,
                "likelihood": "gaussian"},
    "binomial": {"surrogate_degree": 5, "stream": None, "draw_multiplier": 1,
                 "likelihood": "binomial"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def klass(entry: dict) -> str:
    return "information" if float(entry["G_star"]["interval"]["lower"]) > 0.0 else "undetermined"


def oracle_key(row: dict) -> str:
    return (f"s{row['dataset_seed']}__{row['family']}__{row['rung']}__shots_"
            f"{row['shot_level']}__{row['variant']}")


def surrogate_max_error(doc: dict) -> dict:
    out = {}
    for fam_key, fam in doc["surrogates"].items():
        errors = [e["test_mae"] for kind in ("y", "e") for e in fam[kind].values()
                  if e.get("test_mae") is not None]
        out[fam_key] = max(errors) if errors else None
    return out


def run(args: argparse.Namespace) -> dict:
    rule_path = _REPO / RULE_FILE
    if not args.frozen_rule.is_file() or args.frozen_rule.resolve() != rule_path.resolve():
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")
    governed = json.loads(GOVERNED.read_text(encoding="utf-8"))
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("schema") != "bayes_oracle_sensitivity_v1":
            raise SystemExit(f"{path}: not a sensitivity run")
        expected = REQUIRED_RUNS.get(name)
        if expected is None:
            raise SystemExit(f"unknown run name {name}")
        got = {k: doc["options"][k] for k in expected}
        if got != expected:
            raise SystemExit(f"{name}: options {got} differ from the rule's {expected}")
        if doc["options"].get("rungs") is not None:
            raise SystemExit(f"{name}: a sensitivity run must cover the full schedule")
        runs[name] = {"doc": doc, "path": path, "sha256": sha256_file(Path(path))}
    missing = set(REQUIRED_RUNS) - set(runs)
    if missing:
        raise SystemExit(f"missing runs: {sorted(missing)}")

    g_risks = governed["oracle_risks"]
    keys = sorted(k for k in g_risks if "__R0__" not in k)
    for name, r in runs.items():
        absent = [k for k in keys if k not in r["doc"]["oracle_risks"]]
        if absent:
            raise SystemExit(f"{name}: {len(absent)} governed cells absent, e.g. {absent[:3]}")

    # The default-option rerun reproduces the governed risks exactly.
    worst = 0.0
    for k in keys:
        a, b = g_risks[k], runs["default"]["doc"]["oracle_risks"][k]
        for q in ("C_star", "F_star", "G_star"):
            worst = max(worst, abs(a[q]["point"] - b[q]["point"]),
                        abs(a[q]["interval"]["lower"] - b[q]["interval"]["lower"]),
                        abs(a[q]["interval"]["upper"] - b[q]["interval"]["upper"]))
    if worst != 0.0:
        raise SystemExit(f"the default-option rerun differs from the governed oracle by {worst}")

    cells = {}
    for k in keys:
        g = g_risks[k]
        per_run = {"governed": {"G_star": g["G_star"]["point"],
                                "lower": g["G_star"]["interval"]["lower"],
                                "upper": g["G_star"]["interval"]["upper"],
                                "class": klass(g)}}
        for name, r in runs.items():
            if name == "default":
                continue
            e = r["doc"]["oracle_risks"][k]
            per_run[name] = {"G_star": e["G_star"]["point"],
                             "lower": e["G_star"]["interval"]["lower"],
                             "upper": e["G_star"]["interval"]["upper"],
                             "class": klass(e)}
        classes = {v["class"] for v in per_run.values()}
        seed, fam, rung, shots, variant = k.split("__")
        cells[k] = {
            "dataset_seed": int(seed[1:]), "family": fam, "rung": rung,
            "shot_level": shots[len("shots_"):], "variant": variant,
            "governed_class": per_run["governed"]["class"],
            "resolved": len(classes) == 1,
            "max_abs_G_star_change": max(abs(v["G_star"] - per_run["governed"]["G_star"])
                                         for v in per_run.values()),
            "runs": per_run,
        }

    # Counts by rung and shot level.
    by_stratum = defaultdict(Counter)
    for c in cells.values():
        stratum = f"{c['rung']} @ {c['shot_level']}"
        by_stratum[stratum]["cells"] += 1
        by_stratum[stratum][c["governed_class"]] += 1
        if not c["resolved"]:
            by_stratum[stratum]["unresolved"] += 1
            by_stratum[stratum]["unresolved_" + c["governed_class"]] += 1

    # Confusion recount with unresolved cells apart.
    def recount(keep) -> dict:
        t = Counter()
        for row in governed["equation_2"]["rows"]:
            if not keep(row["fit_set"]):
                continue
            label = row["pipeline_label"]
            if row["rung"] == "R0":
                t["no_information"] += 1
                t["false_positives"] += int(label == "F beats C")
                continue
            cell = cells[oracle_key(row)]
            state = "resolved" if cell["resolved"] else "unresolved"
            t[f"{state}_{cell['governed_class']}"] += 1
            if cell["governed_class"] == "information":
                t[f"{state}_information_detected"] += int(label == "F beats C")
                t[f"{state}_information_missed"] += int(label == "not distinguished")
                t[f"{state}_information_wrong_sign"] += int(label == "C beats F")
        return dict(sorted(t.items()))

    surrogates = {name: surrogate_max_error(r["doc"]) for name, r in runs.items()}
    surrogates["governed"] = surrogate_max_error(governed)
    return {
        "schema": "oracle-sensitivity-v1",
        "frozen_rule": RULE_FILE,
        "rule_file_sha256": sha256_file(rule_path),
        "script_sha256": sha256_file(Path(__file__)),
        "governed_sha256": sha256_file(GOVERNED),
        "runs": {name: {"path": r["path"], "sha256": r["sha256"],
                        "options": r["doc"]["options"], "seconds": r["doc"]["seconds"]}
                 for name, r in runs.items()},
        "default_rerun_max_abs_diff": worst,
        "n_cells": len(cells),
        "n_unresolved": sum(not c["resolved"] for c in cells.values()),
        "by_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
        "confusion_recount": {
            "all": recount(lambda name: True),
            "fit_identity": recount(lambda name: not name.endswith("_derived")),
        },
        "surrogate_max_test_error": surrogates,
        "cells": cells,
    }


def markdown(result: dict) -> str:
    lines = ["# Bayes Oracle Sensitivity", "",
             f"Rule: `{RULE_FILE}`, Part B. Cells outside R0: {result['n_cells']}; "
             f"unresolved: {result['n_unresolved']}. Default rerun max difference: "
             f"{result['default_rerun_max_abs_diff']}.", "",
             "| Rung @ shots | Cells | Information | Undetermined | Unresolved |",
             "|---|---|---|---|---|"]
    for stratum, c in result["by_stratum"].items():
        lines.append(f"| {stratum} | {c.get('cells', 0)} | {c.get('information', 0)} | "
                     f"{c.get('undetermined', 0)} | {c.get('unresolved', 0)} |")
    lines += ["", "## Confusion Recount", ""]
    for scope, counts in result["confusion_recount"].items():
        lines.append(f"- {scope}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True, help="NAME=PATH")
    p.add_argument("--frozen-rule", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = p.parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.md.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.out}: {result['n_cells']} cells, {result['n_unresolved']} unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
