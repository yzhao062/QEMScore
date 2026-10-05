#!/usr/bin/env python3
"""Analysis A of the descriptor-information experiment (Parts A and B).

Frozen rule: docs/frozen-rules/2026-10-02-descriptor-information.md
(Estimands, Uncertainty, Classification, Deviations). This is one of the two
independent analysis scripts the rule's Deviations section asks for.

Inputs
------
``--fits DIR`` (repeatable): directories of per-fit files in the format of
tools/descriptor_common.py, ``<stem>.json`` plus ``<stem>.npz``, where the
NPZ holds the selected candidate's test predictions under ``test`` and, when
the fit stored them, each candidate's under ``test__random_forest`` and
``test__mlp``, all aligned to the cached test-row order. Stems follow the
ladder naming of tools/descriptor_ladder.py and tools/qaoa_intermediate.py:

- ``<key>__<rung>__k<NN>__<arm>`` for the Liao arms F, C, P and M at learner
  seed NN; only learner seeds 1 to 20 enter the analysis, and a k-tagged fit
  outside them (Appendix N's anchor, re-exported as NC-R0 at learner seed =
  dataset seed) is listed as an anchor;
- ``<key>__<rung>__orig__<arm>`` for the original-seed anchor fits, which are
  listed but are not part of the 20-seed analysis (``--use-orig-fits`` swaps
  them in as a one-seed sample, for smoke tests only);
- ``<key>__<rung>__<arm>`` for the seedless arms A (affine control) and GBT
  (the Part B reference).

``--cache DIR`` (repeatable): ``<key>.pkl`` caches written by the fit tools,
or ``<key>.json.gz`` compact caches as released for Appendix M.4. Only the
``test`` rows are read: item_id, circuit_id, family, noise_family, severity,
observable, ideal_expectation, noisy_expectation.

Parts, rows and rungs
---------------------
- Part A (keys ``shipped-s<seed>-n<N>``): one row per dataset seed and family
  (tfi, heisenberg). Rungs R0, N1 to N4, R3-TFI (TFI row only), R3-Heis
  (Heisenberg row only), R4, R5. M is F at R5. Contrasts are against R0.
- Part B (keys ``qaoa[-q<n>]-s<seed>-n<N>``): one row per dataset seed. Rungs
  B-partial, B-complete, B-none. M is F at B-none. Contrasts are against
  B-partial. GBT at B-complete is reported as a reference, never as an arm.
- Near-Clifford (keys ``nc-s<seed>-n<N>``): one row per dataset seed. Rungs
  NC-R0 (the Appendix N re-export) and NC-none (arms M and C). M is the NC-none
  M fit. Contrasts are against NC-R0. The rule asks for M on these datasets;
  the rung-level statement here is descriptive.

Estimands per row and rung
--------------------------
Every error is the macro mean absolute error over the row's cells (noise
family x severity x observable): within a cell the mean absolute error over
its items, then the equal-weight mean over cells. At unit weights this is the
campaign's ``_family_mae`` (``math.fsum`` per cell and over cells), and the
per-seed point values are checked against each fit record's
``test_family_mae`` to 1e-12.

- mean C, F, P, M: the mean over learner seeds of the per-seed macro MAE;
  mean A, mean R (raw noisy estimate) and mean GBT are single values.
- D = C - F, the mean over learner seeds of the per-seed difference.
- D/C, a ratio of means: mean D / mean C, and inside each draw the draw's
  mean D over the draw's mean C.
- E = M - F, paired by learner seed (M at the M rung, F at this rung).
- S = (A - C)/(A - F), a ratio of means inside each draw, reported only when
  A - F (the draw's A minus the draw's mean F) is positive in every draw and
  at the point estimate; otherwise ``not_reported``.
- Paired contrasts against the reference rung: delta D and delta (D/C), on the
  learner seeds both rungs share, with the same seed and circuit indices in
  each draw.
- Learner-fixed secondary D: C - F with both arms' MLP candidate predictions
  (``test__mlp``), plus mean C and F of the MLP candidate.
- Selection counts per cell (row x rung): the selected candidate of each fit,
  by arm, plus convergence-flag counts.
- D per severity (descriptive): the macro over the row's cells at that
  severity, with its interval; D per single cell, point only.
- Sensitivity (Appendix M.4 convention): D on the seeds whose C and F fits are
  both unflagged, under the literal flag and under the secondary flag that
  ignores the platform's spurious matmul warnings. All fits remain the
  decision basis.
- Classification of the cell from the all-fits interval for mean D:
  ``measurement_adds`` (wholly above zero), ``measurement_hurts`` (wholly
  below zero), else ``not_distinguished`` with upper(D)/mean(C).
- Rung-level statement per family (Part A) and for Part B (and near-Clifford):
  the shared label when all three dataset seeds (101, 211, 307) carry one
  label, ``mixed`` when they differ, and ``incomplete`` when a seed's cell is
  absent (no statement is made then).

Uncertainty
-----------
Two-stage percentile bootstrap, ``--draws`` draws (default 10,000). For every
row and every quantity the draws come from a fresh
``numpy.random.default_rng(--bootstrap-seed)``; each draw calls
``integers(0, n_seeds, n_seeds)`` and then ``integers(0, n_circuits,
n_circuits)``. The resamples are stored as counts (``np.bincount``): learner
seeds by multiplicity, test circuits as whole units by multiplicity, so every
item of a resampled circuit enters its cell with the circuit's multiplicity.
Inside a draw a cell's MAE is the multiplicity-weighted mean absolute error of
its items and the macro is the equal-weight mean over cells; a seed-averaged
statistic is ``sum(seed_counts * per_seed_value) / n_seeds``. This is the
arithmetic of tools/two_stage_bootstrap.py (Appendix M.4) operation for
operation, so R0 reproduces Appendix M.4 bit for bit. Intervals are the 2.5
and 97.5 percentiles (numpy linear interpolation), pointwise.

Because each quantity's generator is fresh and seeded the same way, two
quantities with the same seed count and circuit count see the same draws; the
script therefore memoizes the draws by (n_seeds, n_circuits, draws, seed),
which returns exactly what a fresh generator returns (the self-test checks
this). A seedless quantity (A, R, GBT) makes the same two calls per draw with
the rung's D seed count (20 when D is absent) and ignores the seed resample,
so its circuit resamples are those of the rung's D.

Usage
-----
  PYTHONPATH=. python tools/descriptor_ladder_analysis.py \
      --fits RUNS/partA/fits --fits RUNS/partA/gate/fits --fits RUNS/partB/fits \
      --cache RUNS/partA/cache --cache RUNS/partB/cache --out RUNS/analysis_a.json
  PYTHONPATH=. python tools/descriptor_ladder_analysis.py --self-test
"""

from __future__ import annotations

import argparse
from collections import Counter
import functools
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import time

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

RULE_FILE = "docs/frozen-rules/2026-10-02-descriptor-information.md"
RULE_SEED = 20261002
DEFAULT_DRAWS = 10_000
PERCENTILES = (2.5, 97.5)
CELL_FIELDS = ("noise_family", "severity", "observable")
DATASET_SEEDS = (101, 211, 307)
LEARNER_SEEDS = tuple(range(1, 21))
SEEDED_ARMS = ("F", "C", "P", "M")
SEEDLESS_ARMS = ("A", "GBT")
LABEL_ADDS = "measurement_adds"
LABEL_HURTS = "measurement_hurts"
LABEL_NONE = "not_distinguished"
POINT_CHECK_TOLERANCE = 1e-12

STEM = re.compile(
    r"^(?P<key>[A-Za-z0-9.\-]+)__(?P<rung>[A-Za-z0-9.+\-]+)"
    r"(?:__(?P<tag>k\d{2,}|orig))?__(?P<arm>A|F|C|P|M|GBT)$")

_LADDER = ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
PARTS: dict[str, dict] = {
    "A": {
        "title": "Part A: spin chains, same circuits",
        "key_pattern": re.compile(r"^shipped-s(\d+)-n\d+$"),
        "families": ("tfi", "heisenberg"),
        "rungs": _LADDER,
        "rung_families": {"R3-TFI": ("tfi",), "R3-Heis": ("heisenberg",)},
        "reference": "R0",
        "m_rung": "R5",
        "expected_arms": {rung: ("A", "F", "C", "P") for rung in _LADDER},
        "m_note": "M is F at R5 (the same fits)",
    },
    "B": {
        "title": "Part B: QAOA-MaxCut, random graphs",
        "key_pattern": re.compile(r"^qaoa(?:-q\d+)?-s(\d+)-n\d+$"),
        "families": None,
        "rungs": ("B-partial", "B-complete", "B-none"),
        "rung_families": {},
        "reference": "B-partial",
        "m_rung": "B-none",
        "expected_arms": {"B-partial": ("A", "F", "C", "P"),
                          "B-complete": ("A", "F", "C", "P", "GBT"),
                          "B-none": ("A", "F", "C", "P")},
        "m_note": "M is F at B-none (the same fits)",
    },
    "NC": {
        "title": "Near-Clifford datasets of Appendix N (arm M)",
        "key_pattern": re.compile(r"^nc-s(\d+)-n\d+$"),
        "families": None,
        "rungs": ("NC-R0", "NC-none"),
        "rung_families": {},
        "reference": "NC-R0",
        "m_rung": "NC-none",
        "expected_arms": {"NC-R0": ("F", "C", "P"), "NC-none": ("M", "C")},
        "m_note": "M is the NC-none M fit; F at NC-none is that same fit",
    },
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


@functools.lru_cache(maxsize=None)
def bootstrap_draws(n_seeds: int, n_circuits: int, draws: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The resample counts a fresh ``default_rng(seed)`` gives (rule's call order).

    Memoized: a fresh generator with the same seed and sizes returns the same
    stream, so reuse is identical to drawing again. The arrays are read-only.
    """
    rng = np.random.default_rng(seed)
    seed_counts = np.zeros((draws, n_seeds))
    circuit_counts = np.zeros((draws, n_circuits))
    for b in range(draws):
        seed_counts[b] = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
        circuit_counts[b] = np.bincount(
            rng.integers(0, n_circuits, n_circuits), minlength=n_circuits)
    seed_counts.flags.writeable = False
    circuit_counts.flags.writeable = False
    return seed_counts, circuit_counts


def macro_under_weights(errors: np.ndarray, item_circuit: np.ndarray,
                        members: list[np.ndarray], cells: tuple[int, ...],
                        circuit_counts: np.ndarray) -> np.ndarray:
    """(draws x seeds) macro MAE of item errors (seeds x items) under circuit counts.

    The arithmetic of Appendix M.4's ``_macro_under_weights``; ``cells``
    selects which cells enter the macro (all cells for the headline). The
    platform's Accelerate BLAS raises spurious floating-point flags in matmul
    on finite inputs (Appendix M.4); they are silenced here and the result is
    required to be finite instead.
    """
    total = np.zeros((circuit_counts.shape[0], errors.shape[0]))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for cell in cells:
            index = members[cell]
            weights = circuit_counts[:, item_circuit[index]]
            total += (weights @ errors[:, index].T) / weights.sum(axis=1, keepdims=True)
    result = total / len(cells)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite macro MAE under bootstrap weights")
    return result


def point_macro(errors_row: np.ndarray, members: list[np.ndarray],
                cells: tuple[int, ...]) -> float:
    """Unit-weight macro MAE with the campaign's exact summation (math.fsum)."""
    maes = [math.fsum(errors_row[members[cell]].tolist()) / len(members[cell])
            for cell in cells]
    return math.fsum(maes) / len(maes)


def interval_entry(values: np.ndarray, point: float, n_seeds: int | None,
                   seeds: list | None = None) -> dict:
    lower, upper = np.percentile(values, PERCENTILES)
    entry = {"status": "estimated", "point": float(point),
             "interval": {"lower": float(lower), "upper": float(upper)},
             "excludes_zero_above": bool(lower > 0.0),
             "excludes_zero_below": bool(upper < 0.0),
             "draw_sd": float(values.std(ddof=1)) if values.size > 1 else None}
    if n_seeds is not None:
        entry["n_seeds"] = int(n_seeds)
    if seeds is not None:
        entry["learner_seeds"] = list(seeds)
    return entry


def classify(d_entry: dict, mean_c: float | None) -> dict:
    """The rule's mutually exclusive label from the all-fits interval for mean D."""
    if d_entry.get("status") != "estimated":
        return {"label": None, "status": "not_estimable",
                "reason": d_entry.get("reason", "D not estimated")}
    lower = d_entry["interval"]["lower"]
    upper = d_entry["interval"]["upper"]
    out = {"status": "estimated", "interval_D": {"lower": lower, "upper": upper},
           "mean_D": d_entry["point"], "mean_C": mean_c}
    if lower > 0.0:
        out["label"] = LABEL_ADDS
    elif upper < 0.0:
        out["label"] = LABEL_HURTS
    else:
        out["label"] = LABEL_NONE
        out["largest_reduction_not_excluded"] = (
            None if not mean_c else _finite(upper / mean_c))
        out["largest_reduction_definition"] = "upper(D) / mean(C)"
    return out


def rung_statement(labels: dict[int, str | None],
                   dataset_seeds: tuple[int, ...] = DATASET_SEEDS) -> dict:
    present = {seed: label for seed, label in labels.items() if label is not None}
    missing = [seed for seed in dataset_seeds if seed not in present]
    if missing:
        statement = "incomplete"
    elif len(set(present.values())) == 1:
        statement = next(iter(present.values()))
    else:
        statement = "mixed"
    return {"statement": statement,
            "labels": {str(seed): present.get(seed) for seed in dataset_seeds},
            "missing_dataset_seeds": missing}


# --------------------------------------------------------------------------
# Inputs: fits and caches
# --------------------------------------------------------------------------


class FitStore:
    """Every parsable fit under the --fits directories, grouped by (key, rung, arm)."""

    def __init__(self, fit_dirs: list[Path], use_orig: bool) -> None:
        self.use_orig = use_orig
        self.groups: dict[tuple[str, str, str], dict] = {}
        self.unparsed: list[str] = []
        self.with_errors: list[dict] = []
        self.n_files = 0
        self._arrays: dict[tuple[str, str], np.ndarray | None] = {}
        seen: dict[str, str] = {}
        digest = hashlib.sha256()
        for fit_dir in fit_dirs:
            fit_dir = Path(fit_dir)
            if not fit_dir.is_dir():
                raise SystemExit(f"--fits {fit_dir} is not a directory")
            for path in sorted(fit_dir.glob("*.json")):
                match = STEM.match(path.stem)
                if match is None:
                    self.unparsed.append(str(path))
                    continue
                stem = path.stem
                if stem in seen:
                    raise SystemExit(f"fit {stem} appears in {seen[stem]} and {fit_dir}")
                seen[stem] = str(fit_dir)
                self.n_files += 1
                key, rung, tag, arm = (match.group("key"), match.group("rung"),
                                       match.group("tag"), match.group("arm"))
                meta = json.loads(path.read_text(encoding="utf-8"))
                npz = path.with_suffix(".npz")
                record = {"stem": stem, "json": str(path), "npz": str(npz),
                          "key": key, "rung": rung, "arm": arm, "tag": tag, "meta": meta}
                self._check_record(record)
                digest.update(stem.encode())
                digest.update(sha256_file(path).encode())
                if npz.exists():
                    digest.update(sha256_file(npz).encode())
                if meta.get("error") or not npz.exists():
                    self.with_errors.append({
                        "stem": stem, "error": meta.get("error") or "npz missing"})
                    continue
                group = self.groups.setdefault(
                    (key, rung, arm), {"seeded": {}, "anchors": {}, "seedless": None})
                if arm in SEEDLESS_ARMS:
                    group["seedless"] = record
                elif tag == "orig":
                    seed = int(meta.get("learner_seed", -1))
                    (group["seeded"] if use_orig else group["anchors"])[seed] = record
                elif int(tag[1:]) not in LEARNER_SEEDS:
                    # A k-tagged fit outside learner seeds 1 to 20 is an anchor
                    # (Appendix N's anchor seed is re-exported as NC-R0 k<dataset
                    # seed>); the rule's analysis uses learner seeds 1 to 20 only.
                    group["anchors"][int(tag[1:])] = record
                elif not use_orig:
                    group["seeded"][int(tag[1:])] = record
        self.inputs_digest = digest.hexdigest()

    @staticmethod
    def _check_record(record: dict) -> None:
        meta, stem, arm, tag = record["meta"], record["stem"], record["arm"], record["tag"]
        if arm in SEEDLESS_ARMS and tag is not None:
            raise SystemExit(f"{stem}: seedless arm {arm} carries a seed tag")
        if arm in SEEDED_ARMS and tag is None:
            raise SystemExit(f"{stem}: arm {arm} has no learner-seed tag")
        for field, expected in (("key", record["key"]), ("rung", record["rung"]),
                                ("arm", arm)):
            if field in meta and meta[field] is not None and str(meta[field]) != expected:
                raise SystemExit(f"{stem}: record {field}={meta[field]!r} disagrees "
                                 f"with the file name ({expected!r})")
        if tag is not None and tag.startswith("k") and meta.get("learner_seed") is not None:
            if int(meta["learner_seed"]) != int(tag[1:]):
                raise SystemExit(f"{stem}: learner_seed {meta['learner_seed']} "
                                 f"disagrees with the file name")

    def seeded(self, key: str, rung: str, arm: str) -> dict[int, dict]:
        group = self.groups.get((key, rung, arm))
        return dict(group["seeded"]) if group else {}

    def anchors(self, key: str, rung: str, arm: str) -> dict[int, dict]:
        group = self.groups.get((key, rung, arm))
        return dict(group["anchors"]) if group else {}

    def seedless(self, key: str, rung: str, arm: str) -> dict | None:
        group = self.groups.get((key, rung, arm))
        return group["seedless"] if group else None

    def array(self, record: dict, name: str) -> np.ndarray | None:
        cache_key = (record["stem"], name)
        if cache_key not in self._arrays:
            with np.load(record["npz"]) as handle:
                self._arrays[cache_key] = (np.asarray(handle[name], dtype=float)
                                           if name in handle.files else None)
        return self._arrays[cache_key]

    def keys(self) -> set[str]:
        return {key for key, _, _ in self.groups} | {
            STEM.match(Path(e["stem"]).name).group("key") for e in self.with_errors}

    def rungs_for(self, key: str) -> set[str]:
        return {rung for k, rung, _ in self.groups if k == key}


def discover_caches(cache_dirs: list[Path]) -> tuple[dict[str, Path], list[str]]:
    found: dict[str, Path] = {}
    shadowed: list[str] = []
    for cache_dir in cache_dirs:
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            raise SystemExit(f"--cache {cache_dir} is not a directory")
        for path in sorted(cache_dir.iterdir()):
            name = path.name
            if name.endswith(".pkl"):
                key = name[:-len(".pkl")]
            elif name.endswith(".json.gz"):
                key = name[:-len(".json.gz")]
            else:
                continue
            if key in found:
                shadowed.append(str(path))
                continue
            found[key] = path
    return found, shadowed


def load_cache(path: Path) -> dict:
    if path.name.endswith(".pkl"):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    if "test" not in data:
        raise SystemExit(f"{path}: no test rows")
    return data


class Row:
    """The test items of one row and their circuit and cell structure."""

    def __init__(self, data: dict, family: str | None, label: str) -> None:
        test = data["test"]
        self.label = label
        self.family = family
        self.n_test_total = len(test)
        self.index = np.asarray(
            [i for i, row in enumerate(test)
             if family is None or str(row["family"]) == family], dtype=int)
        rows = [test[i] for i in self.index]
        if not rows:
            raise ValueError(f"{label}: no test items")
        self.families = sorted({str(row["family"]) for row in rows})
        self.ideal = np.asarray([float(row["ideal_expectation"]) for row in rows])
        self.noisy = np.asarray([float(row["noisy_expectation"]) for row in rows])
        circuits = sorted({str(row["circuit_id"]) for row in rows})
        position = {name: i for i, name in enumerate(circuits)}
        self.n_circuits = len(circuits)
        self.item_circuit = np.asarray([position[str(row["circuit_id"])] for row in rows])
        self.cells = sorted({tuple(str(row[f]) for f in CELL_FIELDS) for row in rows})
        cell_position = {cell: i for i, cell in enumerate(self.cells)}
        self.item_cell = np.asarray(
            [cell_position[tuple(str(row[f]) for f in CELL_FIELDS)] for row in rows])
        self.members = [np.flatnonzero(self.item_cell == c) for c in range(len(self.cells))]
        self.all_cells = tuple(range(len(self.cells)))
        incidence = np.zeros((self.n_circuits, len(self.cells)), dtype=int)
        np.add.at(incidence, (self.item_circuit, self.item_cell), 1)
        if not np.all(incidence >= 1):
            raise ValueError(f"{label}: a circuit lacks an item in some cell; the "
                             "circuit-weighted macro needs every cell in every circuit")
        self.n_items = len(rows)

    def severities(self) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = {}
        for i, cell in enumerate(self.cells):
            groups.setdefault(cell[1], []).append(i)
        return {sev: tuple(idx) for sev, idx in sorted(groups.items())}


# --------------------------------------------------------------------------
# Estimation for one row
# --------------------------------------------------------------------------


class RowEstimator:
    """All quantities of one row; draws and per-arm macros are memoized."""

    def __init__(self, row: Row, store: FitStore, draws: int, seed: int,
                 nominal_seeds: int) -> None:
        self.row = row
        self.store = store
        self.draws = draws
        self.seed = seed
        self.nominal_seeds = nominal_seeds
        self._errors: dict = {}
        self._macro: dict = {}
        self.point_check_max = 0.0
        self.point_checks = 0

    # ---- inputs ----

    def errors(self, records: tuple[dict, ...], name: str) -> np.ndarray | None:
        key = (tuple(r["stem"] for r in records), name)
        if key not in self._errors:
            out = np.empty((len(records), self.row.n_items))
            for s, record in enumerate(records):
                values = self.store.array(record, name)
                if values is None:
                    self._errors[key] = None
                    return None
                if values.shape != (self.row.n_test_total,):
                    raise SystemExit(f"{record['stem']}: {name} has shape {values.shape}, "
                                     f"the cache has {self.row.n_test_total} test rows")
                out[s] = np.abs(values[self.row.index] - self.row.ideal)
            self._errors[key] = out
        return self._errors[key]

    def raw_errors(self) -> np.ndarray:
        return np.abs(self.row.noisy - self.row.ideal)[None, :]

    def point_values(self, records: tuple[dict, ...], name: str,
                     cells: tuple[int, ...]) -> np.ndarray | None:
        errors = self.errors(records, name)
        if errors is None:
            return None
        values = np.asarray([point_macro(errors[s], self.row.members, cells)
                             for s in range(len(records))])
        if cells == self.row.all_cells and self.row.family is not None or (
                cells == self.row.all_cells and len(self.row.families) == 1):
            family = self.row.families[0]
            for s, record in enumerate(records):
                meta = record["meta"]
                recorded = None
                if name == "test":
                    recorded = (meta.get("test_family_mae") or {}).get(family)
                elif name.startswith("test__"):
                    recorded = ((meta.get("candidate_test_family_mae") or {})
                                .get(name[len("test__"):]) or {}).get(family)
                if recorded is not None:
                    diff = abs(float(recorded) - values[s])
                    self.point_check_max = max(self.point_check_max, diff)
                    self.point_checks += 1
                    if diff > POINT_CHECK_TOLERANCE:
                        raise SystemExit(
                            f"{record['stem']}: macro MAE {values[s]!r} disagrees with "
                            f"the fit record {recorded!r} ({self.row.label}); the "
                            "predictions and the cached rows are misaligned")
        return values

    def macro(self, records: tuple[dict, ...] | None, name: str, n_seeds: int,
              cells: tuple[int, ...]) -> np.ndarray | None:
        """(draws x len(records)) macro MAE; records=None means the raw estimate."""
        stems = ("__raw__",) if records is None else tuple(r["stem"] for r in records)
        key = (stems, name, n_seeds, cells)
        if key not in self._macro:
            errors = self.raw_errors() if records is None else self.errors(records, name)
            if errors is None:
                self._macro[key] = None
            else:
                _, circuit_counts = bootstrap_draws(
                    n_seeds, self.row.n_circuits, self.draws, self.seed)
                self._macro[key] = macro_under_weights(
                    errors, self.row.item_circuit, self.row.members, cells, circuit_counts)
        return self._macro[key]

    def seed_counts(self, n_seeds: int) -> np.ndarray:
        return bootstrap_draws(n_seeds, self.row.n_circuits, self.draws, self.seed)[0]

    # ---- quantities ----

    @staticmethod
    def _records(seeded: dict[int, dict], seeds: list[int]) -> tuple[dict, ...]:
        return tuple(seeded[s] for s in seeds)

    def mean_arm(self, seeded: dict[int, dict], name: str = "test",
                 cells: tuple[int, ...] | None = None) -> tuple[dict, np.ndarray | None]:
        cells = self.row.all_cells if cells is None else cells
        seeds = sorted(seeded)
        if not seeds:
            return {"status": "absent"}, None
        records = self._records(seeded, seeds)
        points = self.point_values(records, name, cells)
        if points is None:
            return {"status": "not_available", "reason": f"no {name} predictions"}, None
        n = len(seeds)
        macro = self.macro(records, name, n, cells)
        draws = np.sum(self.seed_counts(n) * macro, axis=1) / n
        return interval_entry(draws, float(points.mean()), n, seeds), draws

    def mean_single(self, record: dict | None, n_seeds: int, *,
                    raw: bool = False) -> dict:
        """A seedless arm (A, GBT) or the raw estimate R."""
        if not raw and record is None:
            return {"status": "absent"}
        cells = self.row.all_cells
        if raw:
            point = point_macro(self.raw_errors()[0], self.row.members, cells)
            macro = self.macro(None, "raw", n_seeds, cells)
        else:
            points = self.point_values((record,), "test", cells)
            if points is None:
                return {"status": "not_available", "reason": "no test predictions"}
            point = float(points[0])
            macro = self.macro((record,), "test", n_seeds, cells)
        entry = interval_entry(macro[:, 0], point, None)
        entry["draws_seed_count"] = int(n_seeds)
        return entry

    def difference(self, first: dict[int, dict], second: dict[int, dict], *,
                   name: str = "test", cells: tuple[int, ...] | None = None,
                   seeds: list[int] | None = None) -> tuple[dict, dict | None]:
        """Mean over seeds of first - second (Appendix M.4 arithmetic), and parts."""
        cells = self.row.all_cells if cells is None else cells
        common = sorted(set(first) & set(second)) if seeds is None else list(seeds)
        if not common:
            return {"status": "not_estimable", "reason": "no learner seed with both arms",
                    "seeds_first": sorted(first), "seeds_second": sorted(second)}, None
        n = len(common)
        rec_a, rec_b = self._records(first, common), self._records(second, common)
        point_a = self.point_values(rec_a, name, cells)
        point_b = self.point_values(rec_b, name, cells)
        if point_a is None or point_b is None:
            return {"status": "not_available",
                    "reason": f"{name} predictions missing in at least one fit"}, None
        macro_a = self.macro(rec_a, name, n, cells)
        macro_b = self.macro(rec_b, name, n, cells)
        seed_counts = self.seed_counts(n)
        draws = np.sum(seed_counts * (macro_a - macro_b), axis=1) / n
        mean_a = np.sum(seed_counts * macro_a, axis=1) / n
        point = float((point_a - point_b).mean())
        entry = interval_entry(draws, point, n, common)
        dropped = sorted((set(first) | set(second)) - set(common))
        if dropped and seeds is None:
            entry["seeds_without_both_arms"] = dropped
        parts = {"draws": draws, "first_mean_draws": mean_a, "point": point,
                 "first_point": float(point_a.mean()),
                 "second_point": float(point_b.mean()), "seeds": common, "n": n}
        return entry, parts

    def ratio(self, parts: dict | None) -> dict:
        """D/C from a D = C - F difference: ratio of means, inside each draw."""
        if parts is None:
            return {"status": "not_estimable", "reason": "D not estimated"}
        denominator = parts["first_mean_draws"]
        if parts["first_point"] <= 0.0 or np.any(denominator <= 0.0):
            return {"status": "undefined", "reason": "mean C is not positive in every draw"}
        draws = parts["draws"] / denominator
        return interval_entry(draws, parts["point"] / parts["first_point"], parts["n"],
                              parts["seeds"])


# --------------------------------------------------------------------------
# One cell (row x rung)
# --------------------------------------------------------------------------


def _flag_subset(seeded_a: dict, seeded_b: dict, field: str) -> list[int]:
    common = sorted(set(seeded_a) & set(seeded_b))
    return [s for s in common
            if not (seeded_a[s]["meta"].get(field) or seeded_b[s]["meta"].get(field))]


def _selection(seeded: dict[int, dict]) -> dict:
    counts = Counter(str(r["meta"].get("selected_model")) for r in seeded.values())
    return {"n_fits": len(seeded), "selected": dict(sorted(counts.items())),
            "flagged_literal": int(sum(bool(r["meta"].get("flagged_non_converged"))
                                       for r in seeded.values())),
            "flagged_secondary": int(sum(
                bool(r["meta"].get("flagged_excluding_spurious_matmul_fpe"))
                for r in seeded.values()))}


def arms_at(store: FitStore, spec: dict, key: str, rung: str) -> dict:
    """The seeded arms of one rung; F falls back to M at the M rung (NC-none)."""
    arms = {arm: store.seeded(key, rung, arm) for arm in ("F", "C", "P")}
    if not arms["F"] and rung == spec["m_rung"]:
        arms["F"] = store.seeded(key, rung, "M")
    m_fits = store.seeded(key, spec["m_rung"], "M") or store.seeded(key, spec["m_rung"], "F")
    arms["M"] = m_fits
    return arms


def analyze_cell(est: RowEstimator, store: FitStore, spec: dict, key: str,
                 rung: str, *, orig_est: RowEstimator | None = None,
                 orig_store: FitStore | None = None) -> dict:
    arms = arms_at(store, spec, key, rung)
    entry: dict = {"rung": rung}
    entry["fits_present"] = {
        arm: sorted(arms[arm]) for arm in ("F", "C", "P", "M")}
    entry["fits_present"]["A"] = store.seedless(key, rung, "A") is not None
    if rung == spec["m_rung"] and not store.seeded(key, rung, "F"):
        entry["note"] = "F at this rung is the M fit"

    # D first: its seed count sets the draws of the seedless arms.
    d_entry, d_parts = est.difference(arms["C"], arms["F"])
    n_ref = d_parts["n"] if d_parts else est.nominal_seeds

    means = {}
    for arm in ("C", "F", "P", "M"):
        means[arm], _ = est.mean_arm(arms[arm])
    means["A"] = est.mean_single(store.seedless(key, rung, "A"), n_ref)
    means["R"] = est.mean_single(None, n_ref, raw=True)
    if store.seedless(key, rung, "GBT") is not None:
        means["GBT_reference"] = est.mean_single(store.seedless(key, rung, "GBT"), n_ref)
        means["GBT_reference"]["note"] = "a reference, not an arm"
    entry["means"] = means
    entry["D"] = d_entry
    entry["D_over_C"] = est.ratio(d_parts)

    e_entry, _ = est.difference(arms["M"], arms["F"])
    if e_entry.get("status") == "estimated":
        e_entry["M_rung"] = spec["m_rung"]
    entry["E"] = e_entry
    entry["S"] = _s_entry(est, store, key, rung, arms, d_parts)

    reference = spec["reference"]
    if rung == reference:
        entry["contrast_vs_reference"] = {"status": "is_reference", "reference": reference}
    else:
        entry["contrast_vs_reference"] = _contrast(est, store, spec, key, rung, arms)

    if orig_est is not None and orig_store is not None:
        entry["comparison_vs_original"] = _compare_original(
            est, orig_est, store, orig_store, spec, key, rung, arms)

    mlp, mlp_parts = est.difference(arms["C"], arms["F"], name="test__mlp")
    learner_fixed = {"candidate": "mlp", "D": mlp}
    if mlp_parts is not None:
        learner_fixed["mean_C_mlp"] = mlp_parts["first_point"]
        learner_fixed["mean_F_mlp"] = mlp_parts["second_point"]
        learner_fixed["classification_descriptive"] = classify(
            mlp, mlp_parts["first_point"]).get("label")
    entry["learner_fixed_secondary"] = learner_fixed

    entry["selection_counts"] = {arm: _selection(arms[arm]) for arm in ("F", "C", "P")}
    entry["selection_counts"]["M"] = _selection(arms["M"])

    by_severity = {}
    for severity, cells in est.row.severities().items():
        sev_entry, _ = est.difference(arms["C"], arms["F"], cells=cells)
        sev_entry["cells"] = [list(est.row.cells[c]) for c in cells]
        by_severity[severity] = sev_entry
    entry["D_by_severity_descriptive"] = by_severity
    by_cell = {}
    common = sorted(set(arms["C"]) & set(arms["F"]))
    if common:
        for c, cell in enumerate(est.row.cells):
            pc = est.point_values(est._records(arms["C"], common), "test", (c,))
            pf = est.point_values(est._records(arms["F"], common), "test", (c,))
            by_cell["/".join(cell)] = {"point_D": float((pc - pf).mean()),
                                       "point_C": float(pc.mean()),
                                       "point_F": float(pf.mean())}
    entry["D_by_cell_point_descriptive"] = by_cell

    sensitivity = {}
    for label, field in (("excluding_flagged_literal", "flagged_non_converged"),
                         ("excluding_flagged_secondary",
                          "flagged_excluding_spurious_matmul_fpe")):
        kept = _flag_subset(arms["C"], arms["F"], field)
        if not common:
            sensitivity[label] = {"status": "not_estimable", "reason": "D not estimated"}
        elif not kept:
            sensitivity[label] = {"status": "not_estimable",
                                  "reason": "no seed with both C and F unflagged",
                                  "n_seeds": 0}
        else:
            sens, sens_parts = est.difference(arms["C"], arms["F"], seeds=kept)
            sens["classification"] = classify(
                sens, sens_parts["first_point"] if sens_parts else None).get("label")
            sensitivity[label] = sens
    entry["sensitivity"] = sensitivity
    entry["classification"] = classify(
        d_entry, d_parts["first_point"] if d_parts else None)
    return entry


def _s_entry(est: RowEstimator, store: FitStore, key: str, rung: str, arms: dict,
             d_parts: dict | None) -> dict:
    a_record = store.seedless(key, rung, "A")
    if a_record is None:
        return {"status": "not_available", "reason": "no arm A fit at this rung"}
    if d_parts is None:
        return {"status": "not_available", "reason": "C and F not both present"}
    seeds, n = d_parts["seeds"], d_parts["n"]
    a_points = est.point_values((a_record,), "test", est.row.all_cells)
    if a_points is None:
        return {"status": "not_available", "reason": "arm A has no test predictions"}
    a_draws = est.macro((a_record,), "test", n, est.row.all_cells)[:, 0]
    rec_c = est._records(arms["C"], seeds)
    rec_f = est._records(arms["F"], seeds)
    seed_counts = est.seed_counts(n)
    c_draws = np.sum(seed_counts * est.macro(rec_c, "test", n, est.row.all_cells),
                     axis=1) / n
    f_draws = np.sum(seed_counts * est.macro(rec_f, "test", n, est.row.all_cells),
                     axis=1) / n
    a_point = float(a_points[0])
    denominator = a_draws - f_draws
    point_denominator = a_point - d_parts["second_point"]
    nonpositive = int(np.sum(denominator <= 0.0))
    if nonpositive or point_denominator <= 0.0:
        return {"status": "not_reported",
                "reason": "A - F is not positive in every draw",
                "n_draws_A_minus_F_nonpositive": nonpositive,
                "point_A_minus_F": point_denominator, "n_seeds": n}
    draws = (a_draws - c_draws) / denominator
    point = (a_point - d_parts["first_point"]) / point_denominator
    entry = interval_entry(draws, point, n, seeds)
    entry["definition"] = "(A - mean C) / (A - mean F), ratio of means inside each draw"
    return entry


def _contrast(est: RowEstimator, store: FitStore, spec: dict, key: str, rung: str,
              arms: dict) -> dict:
    reference = spec["reference"]
    ref_arms = arms_at(store, spec, key, reference)
    seeds = sorted(set(arms["C"]) & set(arms["F"]) & set(ref_arms["C"]) & set(ref_arms["F"]))
    if not (ref_arms["C"] and ref_arms["F"]):
        return {"status": "not_available", "reference": reference,
                "reason": "reference rung C and F not both present"}
    if not seeds:
        return {"status": "not_estimable", "reference": reference,
                "reason": "no learner seed shared by both rungs' C and F"}
    here, here_parts = est.difference(arms["C"], arms["F"], seeds=seeds)
    there, there_parts = est.difference(ref_arms["C"], ref_arms["F"], seeds=seeds)
    if here_parts is None or there_parts is None:
        return {"status": "not_available", "reference": reference,
                "reason": "predictions missing"}
    n = len(seeds)
    delta_d = here_parts["draws"] - there_parts["draws"]
    out = {"status": "estimated", "reference": reference, "n_seeds": n,
           "learner_seeds": seeds,
           "delta_D": interval_entry(delta_d, here_parts["point"] - there_parts["point"],
                                     n)}
    if (np.any(here_parts["first_mean_draws"] <= 0.0)
            or np.any(there_parts["first_mean_draws"] <= 0.0)):
        out["delta_D_over_C"] = {"status": "undefined",
                                 "reason": "mean C not positive in every draw"}
    else:
        ratio_draws = (here_parts["draws"] / here_parts["first_mean_draws"]
                       - there_parts["draws"] / there_parts["first_mean_draws"])
        ratio_point = (here_parts["point"] / here_parts["first_point"]
                       - there_parts["point"] / there_parts["first_point"])
        out["delta_D_over_C"] = interval_entry(ratio_draws, ratio_point, n)
    return out


def _compare_original(est: RowEstimator, orig_est: RowEstimator, store: FitStore,
                      orig_store: FitStore, spec: dict, key: str, rung: str,
                      arms: dict) -> dict:
    orig_arms = arms_at(orig_store, spec, key, rung)
    seeds = sorted(set(arms["C"]) & set(arms["F"]) & set(orig_arms["C"]) & set(orig_arms["F"]))
    if not (orig_arms["C"] and orig_arms["F"]):
        return {"status": "not_available", "reason": "original C and F fits not both present"}
    if not seeds:
        return {"status": "not_estimable", "reason": "no learner seed shared by both fits"}
    here, here_parts = est.difference(arms["C"], arms["F"], seeds=seeds)
    there, there_parts = orig_est.difference(orig_arms["C"], orig_arms["F"], seeds=seeds)
    if here_parts is None or there_parts is None:
        return {"status": "not_available", "reason": "predictions missing"}
    n = len(seeds)
    delta_d = here_parts["draws"] - there_parts["draws"]
    out = {"status": "estimated", "n_seeds": n, "learner_seeds": seeds,
           "delta_D": interval_entry(delta_d, here_parts["point"] - there_parts["point"],
                                     n)}
    if (np.any(here_parts["first_mean_draws"] <= 0.0)
            or np.any(there_parts["first_mean_draws"] <= 0.0)):
        out["delta_D_over_C"] = {"status": "undefined",
                                 "reason": "mean C not positive in every draw"}
    else:
        ratio_draws = (here_parts["draws"] / here_parts["first_mean_draws"]
                       - there_parts["draws"] / there_parts["first_mean_draws"])
        ratio_point = (here_parts["point"] / here_parts["first_point"]
                       - there_parts["point"] / there_parts["first_point"])
        out["delta_D_over_C"] = interval_entry(ratio_draws, ratio_point, n)
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def inventory(store: FitStore, part: str, spec: dict, keys_by_seed: dict[int, str],
              use_orig: bool, dataset_seeds: tuple[int, ...] = DATASET_SEEDS) -> dict:
    expected_seeds = None if use_orig else set(LEARNER_SEEDS)
    absent, incomplete, anchors = [], [], []
    for dataset_seed in dataset_seeds:
        key = keys_by_seed.get(dataset_seed)
        if key is None:
            absent.append({"dataset_seed": dataset_seed, "what": "no fits for this dataset"})
            continue
        for rung in spec["rungs"]:
            for arm in spec["expected_arms"][rung]:
                if arm in SEEDLESS_ARMS:
                    if store.seedless(key, rung, arm) is None:
                        absent.append({"key": key, "rung": rung, "arm": arm})
                    continue
                seeded = store.seeded(key, rung, arm)
                if not seeded:
                    absent.append({"key": key, "rung": rung, "arm": arm})
                elif expected_seeds is not None and set(seeded) != expected_seeds:
                    incomplete.append({"key": key, "rung": rung, "arm": arm,
                                       "n_seeds": len(seeded),
                                       "missing_seeds": sorted(expected_seeds - set(seeded))})
                for seed, record in sorted(store.anchors(key, rung, arm).items()):
                    anchors.append(f"{record['stem']} (learner seed {seed})")
    unknown = sorted({f"{key}/{rung}" for key in keys_by_seed.values()
                      for rung in store.rungs_for(key) if rung not in spec["rungs"]})
    return {"absent": absent, "incomplete_seeds": incomplete,
            "anchor_fits_not_in_analysis": anchors, "unknown_rungs": unknown}


def analyze(fit_dirs: list[Path], cache_dirs: list[Path], *, draws: int, seed: int,
            use_orig: bool = False, verbose: bool = True,
            original_fit_dirs: list[Path] | None = None,
            dataset_seeds: Sequence[int] | None = None) -> dict:
    started = time.perf_counter()
    ds = tuple(dataset_seeds) if dataset_seeds is not None else DATASET_SEEDS
    store = FitStore(fit_dirs, use_orig)
    orig_store = FitStore(original_fit_dirs, use_orig) if original_fit_dirs else None
    caches, shadowed = discover_caches(cache_dirs)
    rule = _REPO / RULE_FILE
    result: dict = {
        "schema": "descriptor-information-analysis-a-v1",
        "frozen_rule": RULE_FILE,
        "rule_file_sha256": sha256_file(rule) if rule.exists() else None,
        "script_sha256": sha256_file(Path(__file__)),
        "bootstrap": {
            "seed": seed, "draws": draws, "percentiles": list(PERCENTILES),
            "rng_scheme": ("fresh numpy default_rng(seed) per row and per quantity; per "
                           "draw integers(0, n_seeds, n_seeds) then integers(0, "
                           "n_circuits, n_circuits); test circuits resampled as whole "
                           "units; Appendix M.4 arithmetic"),
            "rule_default_seed": RULE_SEED,
        },
        "mode": ("smoke: original-seed (orig) fits as a one-seed sample" if use_orig
                 else "learner seeds from k-tagged fits"),
        "inputs": {"fits": [str(Path(d).resolve()) for d in fit_dirs],
                   "cache": [str(Path(d).resolve()) for d in cache_dirs],
                   "n_fit_files": store.n_files,
                   "fits_sha256_digest": store.inputs_digest,
                   "caches_used": {}, "caches_shadowed": shadowed},
        "versions": {"numpy": np.__version__, "python": sys.version.split()[0]},
        "fits_with_errors": store.with_errors,
        "unparsed_json_files": store.unparsed,
        "skipped": [],
        "parts": {},
    }
    if original_fit_dirs:
        result["inputs"]["original_fits"] = [str(Path(d).resolve()) for d in original_fit_dirs]
        result["inputs"]["n_original_fit_files"] = orig_store.n_files
    nominal = 1 if use_orig else len(LEARNER_SEEDS)
    point_check_max, point_checks = 0.0, 0
    fit_keys = store.keys()
    # Keys that match no part are listed, never dropped silently.
    result["unassigned_fit_keys"] = sorted(
        key for key in fit_keys
        if not any(spec["key_pattern"].match(key) for spec in PARTS.values()))
    for part, spec in PARTS.items():
        keys = sorted(k for k in fit_keys if spec["key_pattern"].match(k))
        part_out: dict = {"title": spec["title"], "reference_rung": spec["reference"],
                          "m_rung": spec["m_rung"], "m_note": spec["m_note"]}
        if not keys:
            part_out["status"] = "absent"
            part_out["reason"] = "no fits for this part under --fits"
            result["parts"][part] = part_out
            continue
        keys_by_seed: dict[int, str] = {}
        for key in keys:
            dataset_seed = int(spec["key_pattern"].match(key).group(1))
            if dataset_seed in keys_by_seed:
                raise SystemExit(f"two keys for dataset seed {dataset_seed}: "
                                 f"{keys_by_seed[dataset_seed]} and {key}")
            keys_by_seed[dataset_seed] = key
        part_out["status"] = "analyzed"
        part_out["inventory"] = inventory(store, part, spec, keys_by_seed, use_orig, dataset_seeds=ds)
        rows_out: dict = {}
        for dataset_seed, key in sorted(keys_by_seed.items()):
            if key not in caches:
                result["skipped"].append({"key": key, "reason": "no cache for this key "
                                          "under --cache"})
                continue
            data = load_cache(caches[key])
            result["inputs"]["caches_used"][key] = {
                "path": str(caches[key]), "sha256": sha256_file(caches[key]),
                "dataset_hash": data.get("dataset_hash")}
            for record in [r for g in store.groups.values() if g
                           for r in [*g["seeded"].values(), *g["anchors"].values(),
                                     *([g["seedless"]] if g["seedless"] else [])]
                           if r["key"] == key]:
                recorded = record["meta"].get("dataset_hash")
                if recorded and data.get("dataset_hash") and recorded != data["dataset_hash"]:
                    raise SystemExit(f"{record['stem']}: dataset hash differs from the cache")
            families = spec["families"] or (None,)
            for family in families:
                label = f"{key}/{family}" if family else key
                try:
                    row = Row(data, family, label)
                except ValueError as exc:
                    result["skipped"].append({"row": label, "reason": str(exc)})
                    continue
                est = RowEstimator(row, store, draws, seed, nominal)
                orig_est = RowEstimator(row, orig_store, draws, seed, nominal) if orig_store else None
                row_out = {"key": key, "dataset_seed": dataset_seed, "family": family,
                           "families_in_row": row.families,
                           "n_test_items": row.n_items, "n_test_circuits": row.n_circuits,
                           "cells": [list(c) for c in row.cells],
                           # Part B: cells the fit tool dropped before any fit
                           # (training-label SD below 0.01), as its cache records.
                           "dropped_cells_in_cache": list(data.get("dropped_cells") or []),
                           "rungs": {}, "rungs_absent": []}
                present = store.rungs_for(key)
                for rung in spec["rungs"]:
                    allowed = spec["rung_families"].get(rung)
                    if allowed is not None and family not in allowed:
                        continue
                    if rung not in present:
                        row_out["rungs_absent"].append(rung)
                        continue
                    row_out["rungs"][rung] = analyze_cell(
                        est, store, spec, key, rung, orig_est=orig_est, orig_store=orig_store)
                    if verbose:
                        _print_cell(label, rung, row_out["rungs"][rung])
                point_check_max = max(point_check_max, est.point_check_max)
                point_checks += est.point_checks
                rows_out[label] = row_out
        part_out["rows"] = rows_out
        part_out["rung_statements"] = _statements(spec, rows_out, dataset_seeds=ds)
        result["parts"][part] = part_out
    result["point_check_against_fit_records"] = {
        "n_checked": point_checks, "max_abs_diff": point_check_max,
        "tolerance": POINT_CHECK_TOLERANCE}
    result["seconds"] = time.perf_counter() - started
    return result


def _statements(spec: dict, rows_out: dict, dataset_seeds: tuple[int, ...] = DATASET_SEEDS) -> dict:
    groups = spec["families"] or (None,)
    statements: dict = {}
    for family in groups:
        name = family or "all"
        statements[name] = {}
        for rung in spec["rungs"]:
            allowed = spec["rung_families"].get(rung)
            if allowed is not None and family not in allowed:
                continue
            labels: dict[int, str | None] = {}
            for row in rows_out.values():
                if row["family"] != family:
                    continue
                cell = row["rungs"].get(rung)
                labels[row["dataset_seed"]] = (cell["classification"].get("label")
                                               if cell else None)
            if labels and all(label is None for label in labels.values()):
                # No row of this family has fits at this rung (a reduced rung
                # selection, as in the shot sweep): omit the statement, as
                # analysis B does. A rung fitted in some rows stays "incomplete".
                continue
            statements[name][rung] = rung_statement(labels, dataset_seeds=dataset_seeds)
    return statements


def _fmt(entry: dict, key: str = "point") -> str:
    if entry.get("status") != "estimated":
        return entry.get("status", "-")
    value = entry[key]
    return f"{value:+.6f}"


def _print_cell(label: str, rung: str, cell: dict) -> None:
    d = cell["D"]
    interval = (f"[{d['interval']['lower']:+.6f}, {d['interval']['upper']:+.6f}]"
                if d.get("status") == "estimated" else "")
    comp = cell.get("comparison_vs_original")
    comp_str = ""
    if comp and comp.get("status") == "estimated":
        comp_str = f" [vs orig: dD={_fmt(comp['delta_D'])} d(D/C)={_fmt(comp['delta_D_over_C'])}]"
    print(f"{label:34s} {rung:10s} n={d.get('n_seeds', '-')!s:>3} "
          f"C={_fmt(cell['means']['C'])} F={_fmt(cell['means']['F'])} "
          f"D={_fmt(d)} {interval} D/C={_fmt(cell['D_over_C'])} "
          f"-> {cell['classification'].get('label')}{comp_str}", flush=True)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _naive_two_stage_d(err_c: np.ndarray, err_f: np.ndarray, item_circuit: np.ndarray,
                       item_cell: np.ndarray, n_cells: int, n_circuits: int,
                       draws: int, seed: int) -> np.ndarray:
    """Literal resampling: explicit seed and circuit indices, items by multiplicity."""
    rng = np.random.default_rng(seed)
    n_seeds = err_c.shape[0]
    out = np.empty(draws)
    for b in range(draws):
        seed_index = rng.integers(0, n_seeds, n_seeds)
        circuit_index = rng.integers(0, n_circuits, n_circuits)
        chosen = np.concatenate([np.flatnonzero(item_circuit == c) for c in circuit_index])
        values = []
        for s in seed_index:
            macro_c = np.mean([err_c[s, chosen[item_cell[chosen] == cell]].mean()
                               for cell in range(n_cells)])
            macro_f = np.mean([err_f[s, chosen[item_cell[chosen] == cell]].mean()
                               for cell in range(n_cells)])
            values.append(macro_c - macro_f)
        out[b] = np.mean(values)
    return out


def _synthetic_rows(family: str, dataset_seed: int, n_circuits: int, cells, rng) -> list:
    rows = []
    for c in range(n_circuits):
        circuit = f"circuit-{family}-{dataset_seed}-{c:03d}"
        for noise_family, severity, observable in cells:
            ideal = float(rng.uniform(-1.0, 1.0))
            rows.append({
                "item_id": f"item-{family}-{dataset_seed}-{c:03d}-{severity}-{observable}",
                "circuit_id": circuit, "family": family, "noise_family": noise_family,
                "severity": severity, "observable": observable,
                "ideal_expectation": ideal, "noisy_expectation": 0.8 * ideal,
                "split": "test"})
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def _predictions(rows: list, error_of) -> np.ndarray:
    """ideal +/- error, the sign alternating by item so errors do not cancel."""
    values = np.empty(len(rows))
    for i, row in enumerate(rows):
        sign = 1.0 if i % 2 == 0 else -1.0
        values[i] = float(row["ideal_expectation"]) + sign * error_of(row)
    return values


def _write_fit(fit_dir: Path, key: str, rung: str, arm: str, seed: int | None,
               rows: list, error_of, *, mlp_error_of=None, selected: str = "mlp",
               flagged: bool = False, flagged_secondary: bool = False,
               tag: str | None = None, error: str | None = None) -> None:
    if tag is None:
        tag = None if seed is None else f"k{seed:02d}"
    stem = f"{key}__{rung}__{tag}__{arm}" if tag else f"{key}__{rung}__{arm}"
    arrays = {"test": _predictions(rows, error_of)}
    if mlp_error_of is not None:
        arrays["test__mlp"] = _predictions(rows, mlp_error_of)
        arrays["test__random_forest"] = arrays["test"]
    np.savez_compressed(fit_dir / f"{stem}.npz", **arrays)
    families = sorted({str(r["family"]) for r in rows})
    family_mae = {}
    for family in families:
        index = [i for i, r in enumerate(rows) if r["family"] == family]
        cells = sorted({tuple(str(rows[i][f]) for f in CELL_FIELDS) for i in index})
        maes = []
        for cell in cells:
            members = [i for i in index
                       if tuple(str(rows[i][f]) for f in CELL_FIELDS) == cell]
            maes.append(math.fsum(abs(arrays["test"][i] - rows[i]["ideal_expectation"])
                                  for i in members) / len(members))
        family_mae[family] = math.fsum(maes) / len(maes)
    meta = {"schema": "descriptor-information-fit-v1", "key": key, "rung": rung,
            "arm": arm, "learner_seed": seed, "error": error,
            "dataset_hash": f"synthetic-{key}", "selected_model": selected,
            "flagged_non_converged": flagged,
            "flagged_excluding_spurious_matmul_fpe": flagged_secondary,
            "test_family_mae": family_mae}
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def _build_synthetic(root: Path) -> None:
    rng = np.random.default_rng(7)
    fits = root / "fits"
    cache = root / "cache"
    fits.mkdir(parents=True)
    cache.mkdir(parents=True)
    spin_cells = [("depolarizing_readout", sev, obs)
                  for sev in ("L1", "L3") for obs in ("z_mid", "zz_mid")]
    qaoa_cells = [("depolarizing_readout", sev, "zz_mid") for sev in ("L1", "L3")]
    seeds = (1, 2, 3, 4)

    def const(value):
        return lambda row: value

    for dataset_seed in DATASET_SEEDS:
        key = f"shipped-s{dataset_seed}-n64"
        rows = (_synthetic_rows("tfi", dataset_seed, 12, spin_cells, rng)
                + _synthetic_rows("heisenberg", dataset_seed, 12, spin_cells, rng))
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        with open(cache / f"{key}.pkl", "wb") as handle:
            pickle.dump({"key": key, "dataset_hash": f"synthetic-{key}", "test": rows},
                        handle)
        for k in seeds:
            # R0: F 0.01, C 0.03 -> D 0.02 (adds); MLP: C 0.05, F 0.02 -> 0.03.
            _write_fit(fits, key, "R0", "F", k, rows, const(0.01),
                       mlp_error_of=const(0.02), selected="mlp")
            _write_fit(fits, key, "R0", "C", k, rows, const(0.03),
                       mlp_error_of=const(0.05), selected="random_forest",
                       flagged=(k == 1))
            _write_fit(fits, key, "R0", "P", k, rows, const(0.035))
            # R5: M = F 0.04, C 0.045.
            _write_fit(fits, key, "R5", "F", k, rows, const(0.04))
            _write_fit(fits, key, "R5", "C", k, rows, const(0.045))
            # N1: F 0.03, C 0.01 (hurts), except Heisenberg at seed 307 (adds).
            flip = dataset_seed == 307
            _write_fit(fits, key, "N1", "F", k, rows,
                       lambda r, flip=flip: 0.01 if flip and r["family"] == "heisenberg"
                       else 0.03)
            _write_fit(fits, key, "N1", "C", k, rows,
                       lambda r, flip=flip: 0.03 if flip and r["family"] == "heisenberg"
                       else 0.01)
            # N2: F 0.02, C 0.02 +/- 0.004 by seed parity (not distinguished).
            _write_fit(fits, key, "N2", "F", k, rows, const(0.02))
            _write_fit(fits, key, "N2", "C", k, rows,
                       const(0.024 if k % 2 else 0.016))
            # R4: F 0.01; C 0.02 at L1 and 0.05 at L3.
            _write_fit(fits, key, "R4", "F", k, rows, const(0.01))
            _write_fit(fits, key, "R4", "C", k, rows,
                       lambda r: 0.02 if r["severity"] == "L1" else 0.05)
            # R3-TFI: the ladder fits every item; only the TFI row is read.
            _write_fit(fits, key, "R3-TFI", "F", k, rows, const(0.01))
            _write_fit(fits, key, "R3-TFI", "C", k, rows, const(0.02))
        _write_fit(fits, key, "R0", "A", None, rows, const(0.05))
        _write_fit(fits, key, "N1", "A", None, rows, const(0.02))
        _write_fit(fits, key, "R0", "F", dataset_seed, rows, const(0.5), tag="orig")

        qkey = f"qaoa-s{dataset_seed}-n64"
        qrows = _synthetic_rows("qaoa", dataset_seed, 10, qaoa_cells, rng)
        with open(cache / f"{qkey}.pkl", "wb") as handle:
            pickle.dump({"key": qkey, "dataset_hash": f"synthetic-{qkey}", "test": qrows,
                         "n_qubits": 10,
                         "dropped_cells": (["depolarizing_readout/L9/zz_mid"]
                                           if dataset_seed == 307 else [])}, handle)
        for k in seeds:
            _write_fit(fits, qkey, "B-partial", "F", k, qrows, const(0.01))
            _write_fit(fits, qkey, "B-partial", "C", k, qrows, const(0.03))
            _write_fit(fits, qkey, "B-complete", "F", k, qrows, const(0.01))
            _write_fit(fits, qkey, "B-complete", "C", k, qrows, const(0.02))
            _write_fit(fits, qkey, "B-none", "F", k, qrows, const(0.05))
            _write_fit(fits, qkey, "B-none", "C", k, qrows, const(0.06))
        _write_fit(fits, qkey, "B-partial", "A", None, qrows, const(0.05))
        _write_fit(fits, qkey, "B-complete", "GBT", None, qrows, const(0.025))
        for k in seeds:
            # An errored fit is excluded and listed; P at B-none keeps seeds 2-4.
            _write_fit(fits, qkey, "B-none", "P", k, qrows, const(0.07),
                       error="fit: ValueError: synthetic" if k == 1 else None)

        # Near-Clifford: NC-R0 F 0.01, C 0.03; NC-none M 0.04, C 0.05.
        nkey = f"nc-s{dataset_seed}-n64"
        nrows = _synthetic_rows("near_clifford", dataset_seed, 8, spin_cells, rng)
        with open(cache / f"{nkey}.pkl", "wb") as handle:
            pickle.dump({"key": nkey, "dataset_hash": f"synthetic-{nkey}",
                         "test": nrows}, handle)
        for k in seeds:
            _write_fit(fits, nkey, "NC-R0", "F", k, nrows, const(0.01))
            _write_fit(fits, nkey, "NC-R0", "C", k, nrows, const(0.03))
            _write_fit(fits, nkey, "NC-R0", "P", k, nrows, const(0.02))
            _write_fit(fits, nkey, "NC-none", "M", k, nrows, const(0.04))
            _write_fit(fits, nkey, "NC-none", "C", k, nrows, const(0.05))
        # Appendix N's anchor, re-exported at learner seed = dataset seed (k101 and
        # so on): listed as an anchor, never a 21st learner seed. Its C is far off
        # so that its inclusion would show in D.
        _write_fit(fits, nkey, "NC-R0", "F", dataset_seed, nrows, const(0.01))
        _write_fit(fits, nkey, "NC-R0", "C", dataset_seed, nrows, const(0.5))
    # A fit key that matches no part: listed, not silently ignored.
    _write_fit(fits, "other-s101-n64", "R0", "F", 1, nrows, const(0.01))


def self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    def close(a, b, tol=1e-12) -> bool:
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol

    # 1. Memoized draws equal a fresh generator's stream in the rule's call order.
    seed_counts, circuit_counts = bootstrap_draws(5, 7, 50, 123)
    rng = np.random.default_rng(123)
    fresh_ok = True
    for b in range(50):
        s = np.bincount(rng.integers(0, 5, 5), minlength=5)
        c = np.bincount(rng.integers(0, 7, 7), minlength=7)
        fresh_ok &= np.array_equal(s, seed_counts[b]) and np.array_equal(c, circuit_counts[b])
    check("draws: memoized stream equals a fresh default_rng, seeds then circuits", fresh_ok)

    # 2. Vectorized two-stage D equals literal resampling on random data.
    rng = np.random.default_rng(11)
    n_seeds, n_circuits, n_cells = 6, 9, 3
    item_circuit = np.repeat(np.arange(n_circuits), n_cells)
    item_cell = np.tile(np.arange(n_cells), n_circuits)
    perm = rng.permutation(item_circuit.size)
    item_circuit, item_cell = item_circuit[perm], item_cell[perm]
    err_c = rng.uniform(0, 0.1, (n_seeds, item_circuit.size))
    err_f = rng.uniform(0, 0.1, (n_seeds, item_circuit.size))
    members = [np.flatnonzero(item_cell == c) for c in range(n_cells)]
    cells = tuple(range(n_cells))
    draws = 300
    sc, cc = bootstrap_draws(n_seeds, n_circuits, draws, 99)
    vec = np.sum(sc * (macro_under_weights(err_c, item_circuit, members, cells, cc)
                       - macro_under_weights(err_f, item_circuit, members, cells, cc)),
                 axis=1) / n_seeds
    naive = _naive_two_stage_d(err_c, err_f, item_circuit, item_cell, n_cells,
                               n_circuits, draws, 99)
    check("two-stage: vectorized D equals literal resampling draw by draw",
          np.max(np.abs(vec - naive)) < 1e-12, f"max diff {np.max(np.abs(vec - naive)):.2e}")

    # 3. Bit-identical to Appendix M.4's bootstrap_row on the same inputs.
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from tools.two_stage_bootstrap import bootstrap_row  # noqa: PLC0415
        inputs = {"cells": [[str(c)] for c in range(n_cells)], "n_circuits": n_circuits,
                  "item_circuit": item_circuit, "item_cell": item_cell,
                  "errors": {"C": err_c, "F": err_f, "P": err_f},
                  "point_mae": {arm: np.zeros(n_seeds) for arm in ("C", "F", "P")}}
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            m4 = bootstrap_row(inputs, list(range(1, n_seeds + 1)), draws=draws, seed=99)
        lower, upper = np.percentile(vec, PERCENTILES)
        same = (m4["D"]["two_stage_interval"]["lower"] == float(lower)
                and m4["D"]["two_stage_interval"]["upper"] == float(upper))
        check("two-stage: D interval bit-identical to tools/two_stage_bootstrap.py", same)
    except ImportError as exc:
        check("two-stage: comparison with tools/two_stage_bootstrap.py skipped", True,
              f"import failed: {exc}")

    # 4. End to end on synthetic fits with known values.
    with tempfile.TemporaryDirectory(prefix="ladder-analysis-a-") as tmp:
        root = Path(tmp)
        _build_synthetic(root)
        result = analyze([root / "fits"], [root / "cache"], draws=2000,
                         seed=RULE_SEED, verbose=False)
        part_a = result["parts"]["A"]
        rows = part_a["rows"]
        tfi = rows["shipped-s101-n64/tfi"]["rungs"]
        r0 = tfi["R0"]
        check("A/R0: mean F = 0.01, mean C = 0.03",
              close(r0["means"]["F"]["point"], 0.01) and close(r0["means"]["C"]["point"], 0.03))
        check("A/R0: D = 0.02 with interval [0.02, 0.02]",
              close(r0["D"]["point"], 0.02) and close(r0["D"]["interval"]["lower"], 0.02)
              and close(r0["D"]["interval"]["upper"], 0.02))
        check("A/R0: D/C = 2/3", close(r0["D_over_C"]["point"], 2 / 3))
        check("A/R0: S = (0.05 - 0.03)/(0.05 - 0.01) = 0.5",
              r0["S"].get("status") == "estimated" and close(r0["S"]["point"], 0.5),
              json.dumps(r0["S"])[:200])
        check("A/R0: E = M - F = 0.04 - 0.01 = 0.03", close(r0["E"].get("point"), 0.03))
        check("A/R0: mean A = 0.05", close(r0["means"]["A"]["point"], 0.05))
        tfi_rows = [i for i, r in enumerate(load_cache(root / "cache" /
                                                       "shipped-s101-n64.pkl")["test"])
                    if r["family"] == "tfi"]
        cache_rows = load_cache(root / "cache" / "shipped-s101-n64.pkl")["test"]
        raw_cells: dict = {}
        for i in tfi_rows:
            row = cache_rows[i]
            raw_cells.setdefault((row["severity"], row["observable"]), []).append(
                abs(row["noisy_expectation"] - row["ideal_expectation"]))
        raw_expected = math.fsum(math.fsum(v) / len(v) for v in raw_cells.values()) / 4
        check("A/R0: mean R = macro |noisy - ideal|",
              close(r0["means"]["R"]["point"], raw_expected, 1e-15))
        check("A/R0: label measurement_adds",
              r0["classification"]["label"] == LABEL_ADDS)
        check("A/R0: learner-fixed MLP D = 0.03",
              close(r0["learner_fixed_secondary"]["D"].get("point"), 0.03))
        check("A/R0: selection counts F mlp 4, C random_forest 4",
              r0["selection_counts"]["F"]["selected"] == {"mlp": 4}
              and r0["selection_counts"]["C"]["selected"] == {"random_forest": 4})
        check("A/R0: literal sensitivity drops flagged seed 1",
              r0["sensitivity"]["excluding_flagged_literal"].get("learner_seeds") == [2, 3, 4]
              and r0["sensitivity"]["excluding_flagged_secondary"].get("n_seeds") == 4)
        check("A/R0: the orig fit is listed, not analyzed",
              any("__R0__orig__F" in a
                  for a in part_a["inventory"]["anchor_fits_not_in_analysis"])
              and r0["D"]["learner_seeds"] == [1, 2, 3, 4])
        n1 = tfi["N1"]
        check("A/N1: D = -0.02, label measurement_hurts",
              close(n1["D"]["point"], -0.02) and n1["classification"]["label"] == LABEL_HURTS)
        check("A/N1: delta D vs R0 = -0.04",
              close(n1["contrast_vs_reference"]["delta_D"]["point"], -0.04)
              and close(n1["contrast_vs_reference"]["delta_D"]["interval"]["upper"], -0.04))
        check("A/N1: delta (D/C) vs R0 = -2 - 2/3",
              close(n1["contrast_vs_reference"]["delta_D_over_C"]["point"], -2 - 2 / 3,
                    1e-9))
        check("A/N1: S not reported when A - F <= 0",
              n1["S"]["status"] == "not_reported")
        n2 = tfi["N2"]
        check("A/N2: mean D = 0, interval [-0.004, 0.004], not distinguished",
              close(n2["D"]["point"], 0.0) and close(n2["D"]["interval"]["lower"], -0.004)
              and close(n2["D"]["interval"]["upper"], 0.004)
              and n2["classification"]["label"] == LABEL_NONE)
        check("A/N2: largest reduction not excluded = 0.004/0.02 = 0.2",
              close(n2["classification"]["largest_reduction_not_excluded"], 0.2, 1e-9))
        r4 = tfi["R4"]
        check("A/R4: D = 0.025; by severity L1 0.01, L3 0.04",
              close(r4["D"]["point"], 0.025)
              and close(r4["D_by_severity_descriptive"]["L1"]["point"], 0.01)
              and close(r4["D_by_severity_descriptive"]["L3"]["point"], 0.04))
        check("A/R3-TFI: TFI row only",
              "R3-TFI" in tfi
              and "R3-TFI" not in rows["shipped-s101-n64/heisenberg"]["rungs"])
        statements = part_a["rung_statements"]
        check("A: rung statements (tfi R0 adds, tfi N1 hurts, heisenberg N1 mixed, "
              "tfi N2 not distinguished)",
              statements["tfi"]["R0"]["statement"] == LABEL_ADDS
              and statements["tfi"]["N1"]["statement"] == LABEL_HURTS
              and statements["heisenberg"]["N1"]["statement"] == "mixed"
              and statements["tfi"]["N2"]["statement"] == LABEL_NONE,
              json.dumps({f: {r: s["statement"] for r, s in v.items()}
                          for f, v in statements.items()}))
        check("A: absent rungs listed (N3, N4, R3-Heis) and R3-Heis absent statement",
              {"N3", "N4", "R3-Heis"} <= {a.get("rung") for a in
                                          part_a["inventory"]["absent"]}
              and statements["heisenberg"]["R3-Heis"]["statement"] == "incomplete")
        part_b = result["parts"]["B"]
        b = part_b["rows"]["qaoa-s211-n64"]["rungs"]
        check("B/B-partial: D = 0.02 adds; E = 0.05 - 0.01 = 0.04",
              close(b["B-partial"]["D"]["point"], 0.02)
              and b["B-partial"]["classification"]["label"] == LABEL_ADDS
              and close(b["B-partial"]["E"]["point"], 0.04))
        check("B/B-complete: delta D vs B-partial = -0.01; GBT reference 0.025",
              close(b["B-complete"]["contrast_vs_reference"]["delta_D"]["point"], -0.01)
              and close(b["B-complete"]["means"]["GBT_reference"]["point"], 0.025))
        check("B: rung statement B-partial adds",
              part_b["rung_statements"]["all"]["B-partial"]["statement"] == LABEL_ADDS)
        nc = result["parts"]["NC"]["rows"]["nc-s307-n64"]["rungs"]
        check("NC: NC-R0 D = 0.02, E = M(NC-none) - F(NC-R0) = 0.03",
              close(nc["NC-R0"]["D"]["point"], 0.02) and close(nc["NC-R0"]["E"]["point"], 0.03))
        check("NC: NC-none D = C - M = 0.01 (F is the M fit); delta D vs NC-R0 = -0.01",
              close(nc["NC-none"]["D"]["point"], 0.01)
              and nc["NC-none"].get("note") == "F at this rung is the M fit"
              and close(nc["NC-none"]["contrast_vs_reference"]["delta_D"]["point"], -0.01))
        check("NC: the anchor fit (k<dataset seed>) is an anchor, not a learner seed",
              nc["NC-R0"]["D"]["learner_seeds"] == [1, 2, 3, 4]
              and "nc-s307-n64__NC-R0__k307__C (learner seed 307)"
              in result["parts"]["NC"]["inventory"]["anchor_fits_not_in_analysis"])
        check("B: cells dropped before any fit are disclosed per row",
              part_b["rows"]["qaoa-s307-n64"]["dropped_cells_in_cache"]
              == ["depolarizing_readout/L9/zz_mid"]
              and part_b["rows"]["qaoa-s101-n64"]["dropped_cells_in_cache"] == [])
        check("fit keys matching no part are listed",
              result["unassigned_fit_keys"] == ["other-s101-n64"])
        check("errored fit excluded and listed; its arm reported with the seeds left",
              [e["stem"] for e in result["fits_with_errors"]]
              == [f"qaoa-s{s}-n64__B-none__k01__P" for s in DATASET_SEEDS]
              and b["B-none"]["means"]["P"]["learner_seeds"] == [2, 3, 4])
        check("incomplete learner seeds listed (5 to 20 missing)",
              any(e["rung"] == "R0" and e["arm"] == "F"
                  and e["missing_seeds"] == list(range(5, 21))
                  for e in part_a["inventory"]["incomplete_seeds"]))
        check("learner-fixed D not available without test__mlp",
              tfi["N1"]["learner_fixed_secondary"]["D"]["status"] == "not_available")
        check("point estimates match the fit records' test_family_mae",
              result["point_check_against_fit_records"]["n_checked"] > 0
              and result["point_check_against_fit_records"]["max_abs_diff"] <= 1e-15)
        json.dumps(result, allow_nan=False)
        check("output serializes without NaN or infinity", True)

        # 5. The interval responds to a known D under item-level noise.
        rng = np.random.default_rng(5)
        noisy_root = root / "noisy"
        (noisy_root / "fits").mkdir(parents=True)
        (noisy_root / "cache").mkdir(parents=True)
        spin_cells = [("depolarizing_readout", sev, obs)
                      for sev in ("L1", "L3") for obs in ("z_mid", "zz_mid")]
        outcomes = {}
        for gap, name in ((0.01, "pos"), (-0.01, "neg"), (0.0, "zero")):
            key = f"shipped-s101-n{900 + len(outcomes)}"
            rows = (_synthetic_rows("tfi", 101, 40, spin_cells, rng)
                    + _synthetic_rows("heisenberg", 101, 40, spin_cells, rng))
            with open(noisy_root / "cache" / f"{key}.pkl", "wb") as handle:
                pickle.dump({"key": key, "test": rows}, handle)
            for k in range(1, 9):
                f_err = {r["item_id"]: float(rng.uniform(0.01, 0.03)) for r in rows}
                c_err = {r["item_id"]: max(0.0, f_err[r["item_id"]] + gap
                                           + float(rng.normal(0, 0.002))) for r in rows}
                _write_fit(noisy_root / "fits", key, "R0", "F", k, rows,
                           lambda r, e=f_err: e[r["item_id"]])
                _write_fit(noisy_root / "fits", key, "R0", "C", k, rows,
                           lambda r, e=c_err: e[r["item_id"]])
            outcomes[name] = key
        # Three keys share dataset seed 101 by name pattern only for this check, so
        # analyze each key separately.
        labels = {}
        for name, key in outcomes.items():
            sub = noisy_root / f"only-{name}"
            (sub / "fits").mkdir(parents=True)
            (sub / "cache").mkdir(parents=True)
            for path in (noisy_root / "fits").glob(f"{key}__*"):
                os.link(path, sub / "fits" / path.name)
            os.link(noisy_root / "cache" / f"{key}.pkl", sub / "cache" / f"{key}.pkl")
            out = analyze([sub / "fits"], [sub / "cache"], draws=1000, seed=RULE_SEED,
                          verbose=False)
            cell = out["parts"]["A"]["rows"][f"{key}/tfi"]["rungs"]["R0"]
            labels[name] = (cell["classification"]["label"], cell["D"]["point"])
        check("classification responds: +0.01 adds, -0.01 hurts, 0 not distinguished",
              labels["pos"][0] == LABEL_ADDS and labels["neg"][0] == LABEL_HURTS
              and labels["zero"][0] == LABEL_NONE, json.dumps(labels))

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print(f"self-test: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", action="append", type=Path, default=[],
                        help="a directory of <stem>.json + <stem>.npz fits (repeatable)")
    parser.add_argument("--cache", action="append", type=Path, default=[],
                        help="a directory of <key>.pkl or <key>.json.gz caches (repeatable)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=RULE_SEED)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--use-orig-fits", action="store_true",
                        help="smoke test only: analyze the original-seed (orig) fits as "
                             "a one-seed sample instead of k01..k20")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--original-fits", action="append", type=Path, default=[],
                        help="optional: directories of original fits to perform paired comparison (descriptive)")
    parser.add_argument("--follow-up-rule", type=Path, default=None,
                        help="optional: a later frozen rule that governs these fits "
                             "(for example docs/frozen-rules/2026-10-03-strength-indicator.md); "
                             "its path and SHA-256 are recorded beside the estimator rule")
    parser.add_argument("--dataset-seeds", nargs="+", type=int, default=list(DATASET_SEEDS),
                        help=f"dataset seeds to evaluate (default: {' '.join(str(s) for s in DATASET_SEEDS)})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.fits or not args.cache or args.out is None:
        parser.error("--fits, --cache and --out are required unless --self-test")
    if args.bootstrap_seed != RULE_SEED:
        print(f"NOTE: bootstrap seed {args.bootstrap_seed} differs from the rule's "
              f"{RULE_SEED}", flush=True)
    result = analyze(args.fits, args.cache, draws=args.draws, seed=args.bootstrap_seed,
                     use_orig=args.use_orig_fits, verbose=not args.quiet,
                     original_fit_dirs=args.original_fits,
                     dataset_seeds=args.dataset_seeds)
    if args.follow_up_rule is not None:
        path = args.follow_up_rule
        found = path if path.is_absolute() or path.exists() else _REPO / path
        result["follow_up_rule"] = {"path": str(path), "sha256": sha256_file(found)}
    _write_json(args.out, result)
    for part, value in result["parts"].items():
        if value.get("status") != "analyzed":
            print(f"part {part}: {value.get('status')} ({value.get('reason', '')})")
            continue
        inventory_ = value["inventory"]
        print(f"part {part}: absent {len(inventory_['absent'])} rung/arm groups, "
              f"incomplete {len(inventory_['incomplete_seeds'])}, "
              f"anchors listed {len(inventory_['anchor_fits_not_in_analysis'])}")
        for group, rungs in value["rung_statements"].items():
            print(f"  {group}: " + ", ".join(f"{rung}={s['statement']}"
                                             for rung, s in rungs.items()))
    if result["skipped"]:
        print(f"skipped: {json.dumps(result['skipped'])}")
    if result["unassigned_fit_keys"]:
        print(f"fit keys matching no part (not analyzed): {result['unassigned_fit_keys']}")
    dropped = {row: value["dropped_cells_in_cache"]
               for part in result["parts"].values() for row, value in
               part.get("rows", {}).items() if value["dropped_cells_in_cache"]}
    if dropped:
        print(f"cells dropped before any fit (from the caches): {json.dumps(dropped)}")
    if result["fits_with_errors"]:
        print(f"fits with errors (excluded): {len(result['fits_with_errors'])}")
    print(f"wrote {args.out} in {result['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
