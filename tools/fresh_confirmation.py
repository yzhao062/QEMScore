#!/usr/bin/env python3
"""Locked predictions of the fresh-circuit confirmation panel: write once, score once.

Frozen rule: ``docs/frozen-rules/2026-10-04-fresh-confirmation.md``. Two
subcommands:

``predict`` writes the predictions file of P6 and P7 from validation rows only.
For each shot level, row (dataset seed and family), and rung (R0, N1, N2, R5)
it computes the ceiling D/C*_val = 1 - macro(c_comb) / macro(c_C^val), with
c_comb = ((c_C^val)^-2 + (c_m^cf)^-2)^-1/2 in each strength-by-observable cell.
c_C^val is C's validation error, averaged over learner seeds 1 to 20 of the
original-candidate C fits at 2,048 shots. c_m^cf is the two-fold cross-fitted
error of the de-attenuated estimate on the level's validation rows, computed
as ``tools/shot_sweep_predict.py`` computes it. The step drops every cache's
test rows on loading and never opens a test prediction. At N1 and N2 it gives
each cell a predicted label with the rule's thresholds. It refuses to
overwrite an existing predictions file and appends the file's SHA-256 and time
to the run log.

``score`` reads the predictions file and the analyses of the fresh fits and
scores P1 to P8 once. Each prediction is "pass", "fail", or "not_testable". A
missing analysis cell is an input error: the script stops.

Run on the earlier dataset seeds (101, 211, 307) with their released trees,
both subcommands give the earlier column the rule quotes; ``predict`` then
reproduces ``posthoc-validation-ceiling.json`` (variant ``validation_only``).

Usage:
    PYTHONPATH=. python tools/fresh_confirmation.py predict --dataset-seeds 401 503 607 \
        --level 256=RUNS/shots-256 ... --level exact=RUNS/shots-exact \
        --c-fits RUNS/shots-2048/fits --out RUNS/predictions-fresh.json --run-log run.log
    PYTHONPATH=. python tools/fresh_confirmation.py score --dataset-seeds 401 503 607 \
        --predictions RUNS/predictions-fresh.json \
        --original 256=.../analysis-a.json ... --strong 2048=.../analysis-a.json \
        --stacking .../posthoc-stacking-fresh.json --stacking-set fresh-exact \
        --out RUNS/fresh-confirmation.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

from qemscore.campaign.analysis import _family_mae  # noqa: E402

SCHEMA_PREDICT = "fresh-confirmation-predictions-v1"
SCHEMA_SCORE = "fresh-confirmation-score-v1"
RULE = "docs/frozen-rules/2026-10-04-fresh-confirmation.md"
LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")
OTHER_LEVELS = tuple(level for level in LEVELS if level != "2048")
FAMILIES = ("tfi", "heisenberg")
CELLS_DEF = tuple((s, o) for s in ("L1", "L3") for o in ("z_mid", "zz_mid"))
PREDICT_RUNGS = ("R0", "N1", "N2", "R5")
ADDS_THRESHOLD = {"N1": 0.189, "N2": 0.100}
NOT_DISTINGUISHED_MAX = 0.05
QUALIFY_MIN = 0.10
LADDER = ("R0", "N1", "N2", "N3", "N4")
FIRST_RUNG = {"tfi": "N2", "heisenberg": "N3"}
ADDS, NONE, HURTS = "measurement_adds", "not_distinguished", "measurement_hurts"


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


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def parse_pairs(values: list[str] | None, name: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        level, sep, path = value.partition("=")
        if not sep or level not in LEVELS:
            raise SystemExit(f"{name} {value!r}: expected LEVEL=PATH with LEVEL in {LEVELS}")
        if level in out:
            raise SystemExit(f"{name}: level {level} given twice")
        out[level] = Path(path)
    return out


def row_key(seed: int, family: str) -> str:
    return f"shipped-s{seed}-n640/{family}"


def cell_index(items: list[dict], family: str, severity: str, observable: str) -> np.ndarray:
    return np.asarray([i for i, r in enumerate(items)
                       if str(r["family"]) == family and str(r["severity"]) == severity
                       and str(r["observable"]) == observable], dtype=int)


# ---------------------------------------------------------------- predict

def load_validation(cache_path: Path) -> dict:
    """The cache's validation rows and identity; its test rows are dropped unread."""
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    return {"key": cache["key"], "dataset_hash": cache["dataset_hash"],
            "validation": cache["validation"]}


def cross_fitted_c_m(y_v: np.ndarray, r_v: np.ndarray) -> float:
    """Two-fold cross-fitted error of (r - a) / b on validation rows (shot_sweep_predict (c))."""
    n_v = len(y_v)
    half = n_v // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        if half == 0:
            b, a = np.polyfit(y_v, r_v, 1)
            if abs(b) <= 1e-12:
                return float("inf")
            return float(math.fsum(np.abs((r_v - a) / b - y_v).tolist()) / n_v)
        b1, a1 = np.polyfit(y_v[:half], r_v[:half], 1)
        pred2 = np.full(n_v - half, np.inf) if abs(b1) <= 1e-12 else (r_v[half:] - a1) / b1
        b2, a2 = np.polyfit(y_v[half:], r_v[half:], 1)
        pred1 = np.full(half, np.inf) if abs(b2) <= 1e-12 else (r_v[:half] - a2) / b2
    pred = np.concatenate([pred1, pred2])
    if np.any(np.isinf(pred)):
        return float("inf")
    return float(math.fsum(np.abs(pred - y_v).tolist()) / n_v)


def ceiling(c_C: list[float], c_m: list[float]) -> float:
    comb = [cc if math.isinf(cm) else (cc ** -2.0 + cm ** -2.0) ** -0.5
            for cc, cm in zip(c_C, c_m)]
    return 1.0 - float(np.mean(comb)) / float(np.mean(c_C))


def predicted_label(rung: str, value: float) -> str | None:
    if rung not in ADDS_THRESHOLD:
        return None
    if value >= ADDS_THRESHOLD[rung]:
        return ADDS
    if value <= NOT_DISTINGUISHED_MAX:
        return NONE
    return None


def validation_c_C(seeds: list[int], c_fits: Path, caches: dict[int, dict]) -> tuple[dict, str]:
    """c_C^val per row, rung, and cell from the original-candidate C fits at 2,048 shots."""
    out: dict[str, float] = {}
    digest = hashlib.sha256()
    for seed in seeds:
        cache = caches[seed]
        key = f"shipped-s{seed}-n640"
        if cache["key"] != key:
            raise SystemExit(f"cache for dataset seed {seed} names {cache['key']}")
        validation = cache["validation"]
        y_val = np.asarray([float(r["ideal_expectation"]) for r in validation])
        index = {(fam, s, o): cell_index(validation, fam, s, o)
                 for fam in FAMILIES for s, o in CELLS_DEF}
        for (fam, s, o), idx in index.items():
            if idx.size == 0:
                raise SystemExit(f"{key}: no validation rows for {fam}/{s}/{o}")
        for rung in PREDICT_RUNGS:
            per_cell: dict[tuple, list[float]] = {k: [] for k in index}
            for k in range(1, 21):
                stem = f"{key}__{rung}__k{k:02d}__C"
                record, arrays = c_fits / f"{stem}.json", c_fits / f"{stem}.npz"
                if not record.exists() or not arrays.exists():
                    raise SystemExit(f"{c_fits}: {stem} is missing; every learner seed 1 to 20 is required")
                meta = json.loads(record.read_text(encoding="utf-8"))
                expected = {"key": key, "dataset_seed": seed, "rung": rung, "arm": "C",
                            "learner_seed": k, "dataset_hash": cache["dataset_hash"],
                            "strength_indicator": True}
                for field, value in expected.items():
                    if meta.get(field) != value:
                        raise SystemExit(f"{stem}: {field} {meta.get(field)!r} differs from {value!r}")
                if meta.get("strong_learners") or meta.get("neural_es") or meta.get("train_size"):
                    raise SystemExit(f"{stem}: not an original-candidate fit")
                with np.load(arrays) as data:
                    pred = np.asarray(data["validation"], dtype=float)
                recomputed = _family_mae(validation, pred, artifact_id=cache["dataset_hash"])
                for fam in FAMILIES:
                    diff = abs(recomputed[fam] - meta["validation_family_mae"][fam])
                    if diff > 1e-12:
                        raise SystemExit(f"{stem} {fam}: validation MAE differs by {diff}")
                for name in (record, arrays):
                    digest.update(name.name.encode())
                    digest.update(sha256_file(name).encode())
                for cell, idx in index.items():
                    per_cell[cell].append(math.fsum(np.abs(pred[idx] - y_val[idx]).tolist()) / len(idx))
            for (fam, s, o), values in per_cell.items():
                out[f"{key}/{fam}/{rung}/{s}/{o}"] = float(np.mean(values))
    return out, digest.hexdigest()


def run_predict(seeds: list[int], levels: dict[str, Path], c_fits: Path) -> dict:
    if set(levels) != set(LEVELS):
        raise SystemExit(f"predict needs all seven levels {LEVELS}; got {sorted(levels)}")
    caches: dict[str, dict[int, dict]] = {}
    cache_sha: dict[str, dict[str, str]] = {}
    for level, tree in levels.items():
        caches[level], cache_sha[level] = {}, {}
        for seed in seeds:
            path = tree / "cache" / f"shipped-s{seed}-n640.pkl"
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            caches[level][seed] = load_validation(path)
            cache_sha[level][str(seed)] = sha256_file(path)
    for seed in seeds:
        ref = [r["item_id"] for r in caches["2048"][seed]["validation"]]
        for level in LEVELS:
            if [r["item_id"] for r in caches[level][seed]["validation"]] != ref:
                raise SystemExit(f"level {level}, seed {seed}: validation rows differ in order from 2,048 shots")
    c_C_val, c_fits_digest = validation_c_C(seeds, c_fits, caches["2048"])

    rule_cells: dict[str, dict] = {}
    cells: dict[str, dict] = {}
    for level in LEVELS:
        for seed in seeds:
            validation = caches[level][seed]["validation"]
            for fam in FAMILIES:
                rk = row_key(seed, fam)
                c_m = []
                for s, o in CELLS_DEF:
                    idx = cell_index(validation, fam, s, o)
                    y_v = np.asarray([float(validation[i]["ideal_expectation"]) for i in idx])
                    r_v = np.asarray([float(validation[i]["noisy_expectation"]) for i in idx])
                    value = cross_fitted_c_m(y_v, r_v)
                    c_m.append(value)
                    rule_cells[f"{level}/{rk}/{s}/{o}"] = {
                        "c_m_cross_fitted": value if math.isfinite(value) else "inf",
                        "n_validation_rows": int(idx.size)}
                for rung in PREDICT_RUNGS:
                    c_C = [c_C_val[f"{rk}/{rung}/{s}/{o}"] for s, o in CELLS_DEF]
                    value = ceiling(c_C, c_m)
                    cells[f"{level}/{rk}/{rung}"] = {
                        "level": level, "row": rk, "rung": rung,
                        "D_over_C_star_val": value,
                        "adds_threshold": ADDS_THRESHOLD.get(rung),
                        "predicted_label": predicted_label(rung, value)}
    rule_path = _REPO / RULE
    return {
        "schema": SCHEMA_PREDICT,
        "frozen_rule": RULE,
        "rule_file_sha256": sha256_file(rule_path) if rule_path.exists() else None,
        "script": display(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "dataset_seeds": seeds,
        "thresholds": {"adds": ADDS_THRESHOLD, "not_distinguished_max": NOT_DISTINGUISHED_MAX},
        "inputs": {"levels": {k: display(v) for k, v in levels.items()},
                   "cache_sha256": cache_sha, "c_fits": display(c_fits),
                   "c_fits_digest": c_fits_digest},
        "reads": "validation rows and C validation predictions only; test rows dropped on load",
        "c_C_validation": c_C_val,
        "rule_cells": rule_cells,
        "cells": cells,
    }


# ---------------------------------------------------------------- score

def load_analysis(path: Path, seeds: list[int]) -> dict:
    analysis = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = analysis["parts"]["A"]["rows"]
    expected = {row_key(seed, fam) for seed in seeds for fam in FAMILIES}
    if set(rows) != expected:
        raise SystemExit(f"{path}: rows {sorted(rows)} differ from {sorted(expected)}")
    return analysis


def cell(analysis: dict, row: str, rung: str, path: Path) -> dict:
    entry = analysis["parts"]["A"]["rows"][row]["rungs"].get(rung)
    if entry is None or entry["classification"].get("status") != "estimated":
        raise SystemExit(f"{path}: {row}/{rung} is missing or not estimated")
    return entry


def label(entry: dict) -> str:
    return entry["classification"]["label"]


def normalized(value: str) -> str:
    return NONE if value == HURTS else value


def matches(predicted: str, observed: str) -> bool:
    if predicted == ADDS:
        return observed == ADDS
    return observed in (NONE, HURTS)


def first_adds_rung(analysis: dict, family: str) -> str | None:
    statements = analysis["parts"]["A"]["rung_statements"][family]
    for rung in LADDER:
        statement = statements.get(rung)
        statement = statement.get("statement") if isinstance(statement, dict) else statement
        if statement == ADDS:
            return rung
    return None


def outcome(ok: bool) -> str:
    return "pass" if ok else "fail"


def run_score(seeds: list[int], predictions_path: Path, original: dict[str, Path],
              strong: dict[str, Path], stacking_path: Path, stacking_set: str,
              ladder_path: Path | None = None) -> dict:
    """Score P1 to P8. ``ladder_path`` is the original-candidate analysis A whose
    2,048-shot fits cover the whole ladder; it defaults to ``original['2048']``."""
    if set(original) != set(LEVELS):
        raise SystemExit(f"score needs the original-candidate analysis A at all seven levels; got {sorted(original)}")
    if "2048" not in strong:
        raise SystemExit("score needs the strong-candidate analysis A at 2,048 shots")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if predictions.get("schema") != SCHEMA_PREDICT or predictions.get("dataset_seeds") != seeds:
        raise SystemExit(f"{predictions_path}: not a predictions file for dataset seeds {seeds}")
    orig = {level: load_analysis(path, seeds) for level, path in original.items()}
    ladder_path = ladder_path or original["2048"]
    ladder = load_analysis(ladder_path, seeds)
    strg = {level: load_analysis(path, seeds) for level, path in strong.items()}
    rows = [row_key(seed, fam) for seed in seeds for fam in FAMILIES]
    result: dict[str, dict] = {}

    # P1: original candidates, R0, no row reads "F beats C" at any level.
    p1 = {f"{level}/{row}": label(cell(orig[level], row, "R0", original[level]))
          for level in LEVELS for row in rows}
    hits = sorted(k for k, v in p1.items() if v == ADDS)
    result["P1"] = {"outcome": outcome(not hits), "n_cells": len(p1), "cells_reading_adds": hits,
                    "labels": p1}

    # P2: strong candidates, R0, 2,048 shots, upper D/C limit below 0.01 on every row.
    p2 = {row: cell(strg["2048"], row, "R0", strong["2048"])["D_over_C"]["interval"]["upper"]
          for row in rows}
    result["P2"] = {"outcome": outcome(all(v < 0.01 for v in p2.values())),
                    "upper_limits": p2, "bound": 0.01}

    # P3 and P4: first rung reading "F beats C" along R0, N1, N2, N3, N4.
    for name, analysis in (("P3", ladder), ("P4", strg["2048"])):
        first = {fam: first_adds_rung(analysis, fam) for fam in FAMILIES}
        result[name] = {"outcome": outcome(first == FIRST_RUNG), "first_rung": first,
                        "predicted": dict(FIRST_RUNG)}

    # P5: positive controls.
    p5 = {}
    for row in rows:
        fam = row.rsplit("/", 1)[1]
        rungs = (["R3-TFI"] if fam == "tfi" else []) + ["R4", "R5"]
        for rung in rungs:
            p5[f"2048/{row}/{rung}"] = label(cell(ladder, row, rung, ladder_path))
        for level in OTHER_LEVELS:
            p5[f"{level}/{row}/R5"] = label(cell(orig[level], row, "R5", original[level]))
    misses = sorted(k for k, v in p5.items() if v != ADDS)
    result["P5"] = {"outcome": outcome(not misses), "n_cells": len(p5), "cells_not_adds": misses,
                    "labels": p5}

    # P6 and P7: the shot rule at N1 and N2 on the six levels other than 2,048.
    p6_cells, p7_cells = {}, {}
    for level in OTHER_LEVELS:
        for row in rows:
            for rung in ("N1", "N2"):
                key = f"{level}/{row}/{rung}"
                pred = predictions["cells"][key]
                observed = cell(orig[level], row, rung, original[level])
                reference = normalized(label(cell(orig["2048"], row, rung, original["2048"])))
                star = pred["D_over_C_star_val"]
                if pred["predicted_label"] is not None:
                    p6_cells[key] = {
                        "predicted_label": pred["predicted_label"],
                        "observed_label": label(observed),
                        "reference_label_2048": reference,
                        "label_changing": pred["predicted_label"] != reference,
                        "match": matches(pred["predicted_label"], label(observed)),
                        "kept_label_match": matches(reference, label(observed)),
                        "D_over_C_star_val": star}
                if star >= QUALIFY_MIN:
                    point = observed["D_over_C"]["point"]
                    lower = observed["D_over_C"]["interval"]["lower"]
                    p7_cells[key] = {"D_over_C_star_val": star, "observed_D_over_C": point,
                                     "ratio": point / star, "lower": lower,
                                     "exceeds": bool(lower > star)}
    predicted = list(p6_cells.values())
    changing = [c for c in predicted if c["label_changing"]]
    changing_rows = {k.split("/", 1)[1].rsplit("/", 1)[0] for k, c in p6_cells.items() if c["label_changing"]}
    count = lambda cs, f: sum(1 for c in cs if c[f])  # noqa: E731
    if len(changing) < 6 or len(changing_rows) < 2:
        p6_outcome = "not_testable"
    else:
        p6_outcome = outcome(count(predicted, "match") >= 0.8 * len(predicted)
                             and count(changing, "match") >= 0.8 * len(changing))
    result["P6"] = {
        "outcome": p6_outcome,
        "predicted": [count(predicted, "match"), len(predicted)],
        "label_changing": [count(changing, "match"), len(changing)],
        "label_changing_rows": sorted(changing_rows),
        "kept_label_baseline": {"predicted": [count(predicted, "kept_label_match"), len(predicted)],
                                "label_changing": [count(changing, "kept_label_match"), len(changing)]},
        "abstentions": sum(1 for level in OTHER_LEVELS for row in rows for rung in ("N1", "N2")
                           if predictions["cells"][f"{level}/{row}/{rung}"]["predicted_label"] is None),
        "cells": p6_cells}
    qualifying = list(p7_cells.values())
    q_rows = {k.split("/", 1)[1].rsplit("/", 1)[0] for k in p7_cells}
    if len(qualifying) < 10 or len(q_rows) < 2:
        p7_outcome, median = "not_testable", None
    else:
        median = float(np.median([c["ratio"] for c in qualifying]))
        p7_outcome = outcome(0.6 <= median <= 1.1
                             and count(qualifying, "exceeds") <= 0.10 * len(qualifying))
    result["P7"] = {"outcome": p7_outcome, "n_cells": len(qualifying), "rows": sorted(q_rows),
                    "median_ratio": median, "exceeding": count(qualifying, "exceeds"),
                    "cells": p7_cells}

    # P8: relative stacking increment at R0 and the exact level above zero on every row.
    stacking = json.loads(stacking_path.read_text(encoding="utf-8"))
    entry = stacking["sets"].get(stacking_set)
    if entry is None:
        raise SystemExit(f"{stacking_path}: no set {stacking_set}")
    if entry.get("analysis_a_sha256") != sha256_file(original["exact"]):
        raise SystemExit(f"{stacking_path}: set {stacking_set} was not computed from the exact-level analysis A given")
    p8 = {}
    for row in rows:
        stack_cell = entry["cells"].get(f"{row}/R0")
        if stack_cell is None:
            raise SystemExit(f"{stacking_path}: {row}/R0 missing from set {stacking_set}")
        p8[row] = stack_cell["relative_increment"]
    result["P8"] = {"outcome": outcome(all(v["interval"]["lower"] > 0 for v in p8.values())),
                    "relative_increment": p8}

    return {
        "schema": SCHEMA_SCORE,
        "frozen_rule": RULE,
        "script": display(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "dataset_seeds": seeds,
        "inputs": {
            "predictions": display(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "original": {k: {"path": display(v), "sha256": sha256_file(v)} for k, v in original.items()},
            "strong": {k: {"path": display(v), "sha256": sha256_file(v)} for k, v in strong.items()},
            "ladder": {"path": display(ladder_path), "sha256": sha256_file(ladder_path)},
            "stacking": {"path": display(stacking_path), "sha256": sha256_file(stacking_path),
                         "set": stacking_set}},
        "predictions": result,
        "summary": {name: value["outcome"] for name, value in result.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--dataset-seeds", type=int, nargs="+", required=True)
    p.add_argument("--level", action="append", required=True, help="LEVEL=TREE (a prepared tree with cache/)")
    p.add_argument("--c-fits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-log", type=Path)
    s = sub.add_parser("score")
    s.add_argument("--dataset-seeds", type=int, nargs="+", required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--original", action="append", required=True, help="LEVEL=analysis-a.json")
    s.add_argument("--strong", action="append", required=True, help="LEVEL=analysis-a.json")
    s.add_argument("--stacking", type=Path, required=True)
    s.add_argument("--stacking-set", required=True)
    s.add_argument("--ladder", type=Path, help="analysis A with every rung at 2,048 shots (default: --original 2048)")
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    seeds = list(args.dataset_seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--dataset-seeds repeats a seed")
    if args.command == "predict":
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; the predictions file is written once")
        payload = run_predict(seeds, parse_pairs(args.level, "--level"), args.c_fits)
        payload["created_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(args.out, payload)
        digest = sha256_file(args.out)
        if args.run_log is not None:
            with open(args.run_log, "a", encoding="utf-8") as handle:
                handle.write(f"{payload['created_utc']}\tpredict\tpredictions {display(args.out)} "
                             f"sha256={digest}\n")
        print(f"wrote {args.out} sha256={digest}")
    else:
        payload = run_score(seeds, args.predictions, parse_pairs(args.original, "--original"),
                            parse_pairs(args.strong, "--strong"), args.stacking, args.stacking_set,
                            args.ladder)
        write_json(args.out, payload)
        print(json.dumps(payload["summary"]))


if __name__ == "__main__":
    main()
