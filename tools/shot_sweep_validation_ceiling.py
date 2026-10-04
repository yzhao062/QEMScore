#!/usr/bin/env python3
"""Shot-sweep ceiling computed from validation rows only, with two baselines (post hoc).

Post hoc and descriptive; it decides nothing and changes no frozen result. The
frozen rule ``docs/frozen-rules/2026-10-03-shot-sweep.md`` computes its ceiling
D/C* from two test-set errors: c_C, the control's test error, and c_m, the test
error of the de-attenuated estimate. Prediction and outcome therefore share
test circuits. This script recomputes the ceiling from validation rows alone:

- c_C^val, C's validation error per rung and strength-by-observable cell,
  averaged over learner seeds 1 to 20 of the sweep's 2,048-shot C fits (C reads
  no r, so these fits serve every level). C chose between its two candidates on
  these rows, so c_C^val is optimistic;
- c_m^cf, the two-fold cross-fitted error of the de-attenuated estimate on the
  validation rows, which the predictions file already holds for every level
  and cell.

Every other input is the rule's: the macro, the ``adds`` thresholds max(0.10, 2h)
and 0.05, the reference labels, and the matching convention. The script swaps
the ceiling into a copy of the predictions file and calls the rule's own
``shot_sweep_hypotheses.evaluate_pipeline``. It also reports two baselines that
use no error scale: shot rank for H1(a), and persistence of the 2,048-shot
label for H2.

Usage:
    PYTHONPATH=. python tools/shot_sweep_validation_ceiling.py \
        --predictions artifacts/descriptor-information/shot-sweep/predictions.json \
        --c-fits RUNS/shots-2048/fits --cache RUNS/shots-2048/cache \
        --analysis-a 256=.../levels/256/analysis-a.json ... exact=... \
        --out artifacts/descriptor-information/shot-sweep/posthoc-validation-ceiling.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

import shot_sweep_hypotheses as hyp  # noqa: E402
from qemscore.campaign.analysis import _family_mae  # noqa: E402

SCHEMA = "posthoc-shot-sweep-validation-ceiling-v1"
SEEDS = hyp.SEEDS
FAMILIES = hyp.FAMILIES
RUNGS = hyp.RUNGS
LEVELS = hyp.CANONICAL_LEVELS
CELLS_DEF = tuple((s, o) for s in ("L1", "L3") for o in ("z_mid", "zz_mid"))
NOT_DISTINGUISHED_MAX = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def cell_indices(items: list[dict], family: str, severity: str, observable: str) -> np.ndarray:
    return np.asarray([i for i, r in enumerate(items)
                       if str(r["family"]) == family and str(r["severity"]) == severity
                       and str(r["observable"]) == observable], dtype=int)


def validation_c_C(c_fits: Path, cache_dir: Path) -> dict[str, float]:
    """c_C^val per row/rung/cell: mean over learner seeds of C's validation MAE."""
    out: dict[str, float] = {}
    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        with open(cache_dir / f"{key}.pkl", "rb") as handle:
            validation = pickle.load(handle)["validation"]
        y_val = np.asarray([float(r["ideal_expectation"]) for r in validation])
        index = {(fam, s, o): cell_indices(validation, fam, s, o)
                 for fam in FAMILIES for s, o in CELLS_DEF}
        for rung in RUNGS:
            per_cell: dict[tuple, list[float]] = {k: [] for k in index}
            for k in range(1, 21):
                stem = f"{key}__{rung}__k{k:02d}__C"
                meta = json.loads((c_fits / f"{stem}.json").read_text(encoding="utf-8"))
                with np.load(c_fits / f"{stem}.npz") as data:
                    pred = np.asarray(data["validation"], dtype=float)
                # The prediction array must follow the cache's validation order.
                recomputed = _family_mae(validation, pred, artifact_id=meta["dataset_hash"])
                for fam in FAMILIES:
                    diff = abs(recomputed[fam] - meta["validation_family_mae"][fam])
                    if diff > 1e-12:
                        raise SystemExit(f"{stem} {fam}: validation MAE differs by {diff}")
                for cell, idx in index.items():
                    per_cell[cell].append(
                        math.fsum(np.abs(pred[idx] - y_val[idx]).tolist()) / len(idx))
            for (fam, s, o), values in per_cell.items():
                out[f"{key}/{fam}/{rung}/{s}/{o}"] = float(np.mean(values))
    return out


def ceiling(c_C: list[float], c_m: list[float]) -> float:
    comb = [cc if math.isinf(cm) else (cc ** -2.0 + cm ** -2.0) ** -0.5
            for cc, cm in zip(c_C, c_m)]
    return 1.0 - float(np.mean(comb)) / float(np.mean(c_C))


def predicted_label(value: float, adds_threshold: float) -> str | None:
    if value >= adds_threshold:
        return "measurement_adds"
    if value <= NOT_DISTINGUISHED_MAX:
        return "not_distinguished"
    return None


def swap_ceiling(predictions: dict, values: dict[str, float]) -> dict:
    """A copy of the predictions file with another ceiling and the rule's labels."""
    alt = copy.deepcopy(predictions)
    for key, cell in alt["classification_cells"].items():
        cell["D_over_C_star"] = values[key]
        label = predicted_label(values[key], cell["adds_threshold"])
        cell["predicted_label"] = label
        cell["label_changing"] = bool(label is not None
                                      and label != cell["reference_label_baseline"])
    return alt


def matches(predicted: str, observed: str) -> bool:
    """The rule's matching: a predicted 'not distinguished' also matches 'hurts'."""
    if predicted == "measurement_adds":
        return observed == "measurement_adds"
    return observed in ("not_distinguished", "measurement_hurts")


def persistence(evaluation_cells: dict) -> dict:
    """H2 counts if every predicted cell kept its 2,048-shot (reference) label."""
    def count(cells):
        return {"n_cells": len(cells),
                "matches": sum(matches(c["reference_baseline"], c["observed_label"])
                               for c in cells)}
    cells = list(evaluation_cells.values())
    predicted = [c for c in cells if c["predicted_label"] is not None]
    changing = [c for c in predicted if c["label_changing"]]
    return {"all_eligible": count(cells), "predicted": count(predicted),
            "label_changing": count(changing)}


def h2_cell_view(evaluation: dict, predictions: dict) -> dict:
    out = {}
    for key, cell in evaluation["h2"]["cells"].items():
        ref = predictions["classification_cells"][key]["reference_label_baseline"]
        out[key] = {**cell, "reference_baseline": ref}
    return out


def shot_rank_spearman(obs_by_level: dict) -> dict:
    out = {}
    for rung in ("N1", "N2"):
        for seed in SEEDS:
            for fam in FAMILIES:
                row = f"shipped-s{seed}-n640/{fam}"
                observed = [obs_by_level[lvl][row][rung]["observed_D_over_C"] for lvl in LEVELS]
                corr, _ = hyp.compute_spearman(list(range(len(LEVELS))), observed)
                out[f"{row}/{rung}"] = corr
    return out


def summarize(evaluation: dict) -> dict:
    h1, h2 = evaluation["h1"], evaluation["h2"]
    return {
        "h1_a_n1_pass": h1["part_a"]["n1_pass_count"],
        "h1_a_n2_pass": h1["part_a"]["n2_pass_count"],
        "h1_qualifying_cells": h1["qualifying_cell_count"],
        "h1_b_median_ratio": h1["part_b"]["median_ratio"],
        "h1_c_exceeding": h1["part_c"]["exceeding_count"],
        "h1_holds": h1["holds"],
        "h2_label_changing": [h2["criterion_label_changing"]["matches"],
                              h2["criterion_label_changing"]["n_cells"]],
        "h2_all_predicted": [h2["criterion_all_predicted"]["matches"],
                             h2["criterion_all_predicted"]["n_cells"]],
        "h2_abstentions": sum(1 for c in h2["cells"].values() if c["predicted_label"] is None),
        "h2_holds": h2["holds"],
        "series_monotone_in_shots": {
            f"{row}/{rung}": bool(np.all(np.diff(series["dc_star_points"]) > 0))
            for part, rung in (("series_n1", "N1"), ("series_n2", "N2"))
            for row, series in h1["part_a"][part].items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--c-fits", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--analysis-a", nargs="+", required=True, metavar="LEVEL=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    analysis_paths = hyp.parse_level_args(args.analysis_a)
    obs_by_level = {lvl: hyp.extract_analysis_a(p) for lvl, p in analysis_paths.items()}

    c_C_val = validation_c_C(args.c_fits, args.cache)
    c_C_test = {k: v["c_C"] for k, v in predictions["c_C_by_rung_cell"].items()}
    rule_cells = predictions["rule_cells"]

    reproduced, val_ceiling, mixed_ceiling = {}, {}, {}
    for key, cell in predictions["classification_cells"].items():
        level, row, rung = cell["level"], cell["row"], cell["rung"]
        cells = [(f"{row}/{rung}/{s}/{o}", f"{level}/{row}/{s}/{o}") for s, o in CELLS_DEF]
        c_m = [rule_cells[rc]["c_m"] for _, rc in cells]
        c_m_cf = [rule_cells[rc]["c_m_cross_fitted"] for _, rc in cells]
        reproduced[key] = ceiling([c_C_test[cc] for cc, _ in cells], c_m)
        val_ceiling[key] = ceiling([c_C_val[cc] for cc, _ in cells], c_m_cf)
        mixed_ceiling[key] = ceiling([c_C_test[cc] for cc, _ in cells], c_m_cf)
    worst = max(abs(reproduced[k] - c["D_over_C_star"])
                for k, c in predictions["classification_cells"].items())
    if worst > 1e-12:
        raise SystemExit(f"the frozen ceiling is not reproduced (largest difference {worst})")

    frozen = hyp.evaluate_pipeline(predictions, obs_by_level)
    variants = {}
    for name, values in (("validation_only", val_ceiling),
                         ("test_c_C_cross_fitted_c_m", mixed_ceiling)):
        alt = swap_ceiling(predictions, values)
        evaluation = hyp.evaluate_pipeline(alt, obs_by_level)
        variants[name] = {
            "summary": summarize(evaluation),
            "persistence": persistence(h2_cell_view(evaluation, alt)),
            "cells": {k: {"D_over_C_star": values[k],
                          "predicted_label": alt["classification_cells"][k]["predicted_label"],
                          "label_changing": alt["classification_cells"][k]["label_changing"]}
                      for k in sorted(values)},
        }

    c_C_ratio = {
        rung: float(np.mean([c_C_val[k] / c_C_test[k] for k in c_C_val
                             if k.split("/")[2] == rung]))
        for rung in RUNGS}

    payload = {
        "schema": SCHEMA,
        "status": "post hoc, descriptive; decides nothing",
        "script": "tools/shot_sweep_validation_ceiling.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "predictions": display(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "c_fits": display(args.c_fits),
            "cache": display(args.cache),
            "analysis_a": {lvl: display(p) for lvl, p in analysis_paths.items()},
        },
        "frozen_ceiling_reproduced_max_abs_diff": worst,
        "mean_ratio_validation_to_test_c_C": c_C_ratio,
        "frozen": {"summary": summarize(frozen),
                   "persistence": persistence(h2_cell_view(frozen, predictions))},
        "variants": variants,
        "shot_rank_spearman_observed": shot_rank_spearman(obs_by_level),
        "c_C_validation": c_C_val,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"frozen": payload["frozen"],
                      **{k: {"summary": v["summary"], "persistence": v["persistence"]}
                         for k, v in variants.items()},
                      "c_C_ratio": c_C_ratio}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
