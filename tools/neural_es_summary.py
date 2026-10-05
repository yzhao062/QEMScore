#!/usr/bin/env python3
"""Validation and test errors of the early-stopped MLP against the original MLP (part C).

Follows rule docs/frozen-rules/2026-10-04-crossed-follow-ups.md (part C, "Reported"). Reads fit
records only; fits nothing and decides nothing. For each dataset seed, family, and rung, it
averages over learner seeds the MLP candidate's validation and test family errors in arm C
(`candidate_validation_family_mae["mlp"]`, `candidate_test_family_mae["mlp"]`) for the
`--neural-es` fits and for the original fits at the same level, and summarizes epochs run and the
best epoch of the early-stopped fits. Every record read is listed with its SHA-256.

Usage:
    PYTHONPATH=. python tools/neural_es_summary.py --fits RUN/fits --original-fits SWEEP/fits \\
        --rungs R0 N1 N2 R5 --out OUT_JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

SCHEMA = "neural-es-summary-v1"
FAMILIES = ("tfi", "heisenberg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(fits: Path, original: Path, rungs: list[str], seeds: list[int],
              learner_seeds: list[int], arm: str = "C") -> dict:
    inputs: dict[str, str] = {}
    cells: dict[str, dict] = {}
    for s in seeds:
        key = f"shipped-s{s}-n640"
        for rung in rungs:
            recs = {"new": [], "orig": []}
            for k in learner_seeds:
                for tag, d in (("new", fits), ("orig", original)):
                    path = d / f"{key}__{rung}__k{k:02d}__{arm}.json"
                    if not path.is_file():
                        raise SystemExit(f"missing fit record {path}")
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    if meta.get("rung") != rung or meta.get("arm") != arm or meta.get("learner_seed") != k:
                        raise SystemExit(f"{path}: record names another job")
                    if tag == "new" and not meta.get("neural_es"):
                        raise SystemExit(f"{path}: not a --neural-es fit")
                    if tag == "orig" and meta.get("neural_es"):
                        raise SystemExit(f"{path}: an original fit must not be a --neural-es fit")
                    inputs[f"{tag}/{path.name}"] = sha256(path)
                    recs[tag].append(meta)
            for fam in FAMILIES:
                val = {t: statistics.fmean(m["candidate_validation_family_mae"]["mlp"][fam] for m in recs[t])
                       for t in recs}
                test = {t: statistics.fmean(m["candidate_test_family_mae"]["mlp"][fam] for m in recs[t])
                        for t in recs}
                epochs = [m["epochs_run"] for m in recs["new"]]
                best = [m["best_epoch"] for m in recs["new"]]
                cells[f"{key}/{fam}/{rung}"] = {
                    "validation_mlp_mean": {"early_stopped": val["new"], "original": val["orig"],
                                            "ratio": val["new"] / val["orig"]},
                    "test_mlp_mean": {"early_stopped": test["new"], "original": test["orig"],
                                      "ratio": test["new"] / test["orig"]},
                    "epochs_run": {"min": min(epochs), "max": max(epochs),
                                   "mean": statistics.fmean(epochs)},
                    "best_epoch": {"min": min(best), "max": max(best), "mean": statistics.fmean(best)},
                    "selected_mlp_in_C": sum(m.get("selected_model") == "mlp" for m in recs["new"]),
                }
    return {"schema": SCHEMA, "arm": arm, "rungs": rungs, "dataset_seeds": seeds,
            "learner_seeds": learner_seeds, "cells": cells, "inputs_sha256": inputs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, required=True, help="--neural-es fits directory")
    parser.add_argument("--original-fits", type=Path, required=True,
                        help="original fits at the same level")
    parser.add_argument("--rungs", nargs="+", default=["R0", "N1", "N2", "R5"])
    parser.add_argument("--dataset-seeds", type=int, nargs="+", default=[101, 211, 307])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.fits, args.original_fits, args.rungs, args.dataset_seeds,
                       list(range(1, 21)))
    result["script_sha256"] = sha256(Path(__file__).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(result['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
