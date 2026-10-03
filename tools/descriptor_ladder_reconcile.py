"""Reconcile the two independent analyses of the descriptor-information experiment.

The frozen rule (docs/frozen-rules/2026-10-02-descriptor-information.md, as
amended) requires the two analysis scripts to agree on every point estimate and
interval bound to 1e-12, and on every cell label and rung statement.

    python tools/descriptor_ladder_reconcile.py \
        artifacts/descriptor-information/analysis-a.json \
        artifacts/descriptor-information/analysis-b.json \
        --out artifacts/descriptor-information/reconcile.json

Exit code 0 when the rule's agreement criterion holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOLERANCE = 1e-12
LABELS = {"Measurement adds": "measurement_adds", "Measurement hurts": "measurement_hurts",
          "Not distinguished": "not_distinguished"}
REFERENCE_RUNGS = ("R0", "B-partial", "NC-R0")


def row_a(a: dict, cell: dict) -> dict:
    seed = cell["dataset_seed"]
    if cell["part"] == "NC":
        return a["parts"]["NC"]["rows"][f"nc-s{seed}-n640"]["rungs"][cell["rung"]]
    if cell["part"] == "B":
        return a["parts"]["B"]["rows"][f"qaoa-s{seed}-n640"]["rungs"][cell["rung"]]
    return a["parts"]["A"]["rows"][f"shipped-s{seed}-n640/{cell['family']}"]["rungs"][cell["rung"]]


def point(x):
    return x["point"] if x and x.get("status") == "estimated" else None


def bound(x, i):
    return x["interval"]["lower" if i == 0 else "upper"] if x and x.get("status") == "estimated" else None


def pairs(ra: dict, cb: dict):
    m = ra["means"]
    yield "mean C", point(m.get("C")), cb.get("mean_c")
    yield "mean F", point(m.get("F")), cb.get("mean_f")
    yield "mean A", point(m.get("A")), cb.get("mean_a")
    yield "mean R", point(m.get("R")), cb.get("mean_r")
    yield "mean M", point(m.get("M")), cb.get("mean_m")
    for name, ka, kb, pb in [("D", "D", "d", "mean_d"), ("D/C", "D_over_C", "d_over_c", "mean_d_over_c"),
                             ("E", "E", "e", "e"), ("S", "S", "s", "s")]:
        ci = cb.get(kb + "_ci_95") or [None, None]
        yield name, point(ra.get(ka)), cb.get(pb)
        yield name + " lower", bound(ra.get(ka), 0), ci[0]
        yield name + " upper", bound(ra.get(ka), 1), ci[1]
    if cb["rung"] not in REFERENCE_RUNGS:
        cr, pc = ra.get("contrast_vs_reference") or {}, cb.get("paired_contrast") or {}
        for name, ka, kb in [("delta D", "delta_D", "delta_d"), ("delta D/C", "delta_D_over_C", "delta_d_over_c")]:
            ci = pc.get(kb + "_ci_95") or [None, None]
            yield name, point(cr.get(ka)), pc.get(kb)
            yield name + " lower", bound(cr.get(ka), 0), ci[0]
            yield name + " upper", bound(cr.get(ka), 1), ci[1]
    sec = (ra.get("learner_fixed_secondary") or {}).get("D")
    ci = cb.get("secondary_d_ci_95") or [None, None]
    yield "MLP D", point(sec), cb.get("secondary_d")
    yield "MLP D lower", bound(sec, 0), ci[0]
    yield "MLP D upper", bound(sec, 1), ci[1]
    for sev, value in (cb.get("d_per_severity") or {}).items():
        yield "D " + sev, point((ra.get("D_by_severity_descriptive") or {}).get(sev)), value


def statements_a(a: dict) -> dict:
    out = {}
    for part, body in a["parts"].items():
        for fam, rungs in (body.get("rung_statements") or {}).items():
            fam = {"NC": "near_clifford", "B": "qaoa"}.get(part, fam) if fam == "all" else fam
            for rung, lab in (rungs or {}).items():
                out[f"{fam}/{rung}"] = lab["statement"] if isinstance(lab, dict) else lab
    return out


def statements_b(b: dict) -> dict:
    out = {}
    for part, rungs in b["rung_statements"].items():
        for rung, fams in rungs.items():
            if isinstance(fams, str):
                fams = {"qaoa" if part == "B" else "near_clifford": fams}
            for fam, lab in fams.items():
                fam = {"B": "qaoa", "NC": "near_clifford"}.get(part, fam)
                out[f"{fam}/{rung}"] = LABELS.get(lab, lab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("analysis_a", type=Path)
    ap.add_argument("analysis_b", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    a, b = json.loads(args.analysis_a.read_text()), json.loads(args.analysis_b.read_text())
    worst, one_sided, label_mismatch, n_compared = {}, [], [], 0
    for cell in b["cells"]:
        where = f"{cell['part']}/{cell['row_key']}/{cell['rung']}"
        ra = row_a(a, cell)
        for name, va, vb in pairs(ra, cell):
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                one_sided.append({"cell": where, "quantity": name, "a": va, "b": vb})
                continue
            n_compared += 1
            diff = abs(va - vb)
            if diff >= worst.get(name, {"max_abs_diff": -1.0})["max_abs_diff"]:
                worst[name] = {"max_abs_diff": diff, "at": where}
        la = (ra.get("classification") or {}).get("label")
        lb = LABELS.get(cell.get("classification_label"), cell.get("classification_label"))
        if la != lb:
            label_mismatch.append({"cell": where, "a": la, "b": lb})
    sa, sb = statements_a(a), statements_b(b)
    statement_mismatch = {k: [sa.get(k), sb.get(k)] for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)}
    max_diff = max((v["max_abs_diff"] for v in worst.values()), default=0.0)
    passed = max_diff <= TOLERANCE and not one_sided and not label_mismatch and not statement_mismatch
    report = {
        "schema": "descriptor-information-reconcile-v1",
        "criterion": "points and interval bounds agree to 1e-12; labels and rung statements identical",
        "tolerance": TOLERANCE, "cells": len(b["cells"]), "values_compared": n_compared,
        "max_abs_diff": max_diff, "by_quantity": dict(sorted(worst.items())),
        "one_sided": one_sided, "label_mismatches": label_mismatch,
        "rung_statements_compared": len(set(sa) | set(sb)), "rung_statement_mismatches": statement_mismatch,
        "passed": passed,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"cells {report['cells']}, values {n_compared}, max |A-B| {max_diff:.3e}, "
          f"labels {'identical' if not label_mismatch else 'DIFFER'}, statements "
          f"{'identical' if not statement_mismatch else 'DIFFER'} -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
