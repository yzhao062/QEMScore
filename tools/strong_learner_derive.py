"""Derived baseline fits from strong-learner fits (rule 2026-10-03-strong-learners.md).

For each strong fit in --fits, writes a baseline fit into --out under the same
name, in the format ``tools.descriptor_common._write_fit`` writes with the
option off:

- Liao fits (arms F, C, P): rebuilt from the fit's ``option_off`` object, which
  holds every value the option-off code would have written (selection,
  eligible models, threshold, validation scores of the random forest and the
  MLP, both flags, finite checks, warning counts and rows, family errors,
  ``selected_equals_candidate_predictions``). The selection is re-derived by
  ``select_one_standard_error`` on the random-forest and MLP validation scores
  alone and must equal the recorded one. The ``validation`` and ``test`` arrays
  are that candidate's stored predictions. The strong-learner fields, the hgbr
  and poly5_ridge entries of the candidate error maps, the strong candidates'
  timing keys, and their warning rows are dropped.
- Arm A and GBT fits: the strong-learner fields are dropped and nothing else.

A fit without ``strong_learners: true`` (or a Liao fit without ``option_off``)
is an error, not something to pass through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from qemscore.baselines.liao import (  # noqa: E402
    LiaoValidationScore, select_one_standard_error)
from tools import descriptor_common as common  # noqa: E402

STRONG_FIELDS = ("strong_learners", "candidates", "hgbr_grid_point",
                 "poly5_ridge_alpha", "follow_up_rule", "option_off")
STRONG_PHASES = ("candidate_hgbr", "candidate_poly5_ridge")
STRONG_TIMING_KEYS = ("candidate_hgbr_seconds", "candidate_poly5_ridge_seconds")
BASELINE_CANDIDATES = ("random_forest", "mlp")
# option_off key -> fit JSON key
RESTORED = {
    "selected_model": "selected_model",
    "eligible_models": "eligible_models",
    "one_standard_error_threshold": "one_standard_error_threshold",
    "validation_scores": "validation_scores",
    "flagged": "flagged_non_converged",
    "flagged_secondary": "flagged_excluding_spurious_matmul_fpe",
    "finite": "finite_checks",
    "n_runtime_warnings": "n_runtime_warnings",
    "n_runtime_warnings_not_matmul_fpe": "n_runtime_warnings_not_matmul_fpe",
    "warnings_aggregated": "warnings_aggregated",
    "test_family_mae": "test_family_mae",
    "validation_family_mae": "validation_family_mae",
    "selected_equals_candidate_predictions": "selected_equals_candidate_predictions",
}


def derive_liao_fit(meta: dict, npz_data: dict[str, np.ndarray]
                    ) -> tuple[dict, dict[str, np.ndarray]]:
    """Derive the option-off Liao fit from a strong Liao fit."""
    name = f"{meta.get('key')}/{meta.get('rung')}/{meta.get('arm')}/k{meta.get('learner_seed')}"
    off = meta.get("option_off")
    if not isinstance(off, dict):
        raise ValueError(f"{name}: a strong Liao fit needs its option_off object")
    missing = sorted(set(RESTORED) - set(off))
    if missing:
        raise ValueError(f"{name}: option_off lacks {missing}")
    out_meta = {k: v for k, v in meta.items() if k not in STRONG_FIELDS}
    for off_key, meta_key in RESTORED.items():
        out_meta[meta_key] = off[off_key]

    selected = off["selected_model"]
    if off["validation_scores"] is not None:
        scores = [LiaoValidationScore(
            name=s["name"], validation_mae=float(s["validation_mae"]),
            standard_error=float(s["standard_error"]),
            total_excess_absolute_loss=float(s["total_excess_absolute_loss"]),
            circuit_evaluations=int(s["circuit_evaluations"]),
            simplicity_rank=int(s["simplicity_rank"]))
            for s in off["validation_scores"]]
        if [s.name for s in scores] != list(BASELINE_CANDIDATES):
            raise ValueError(f"{name}: option_off validation scores are not "
                             f"{list(BASELINE_CANDIDATES)}")
        chosen, threshold, eligible = select_one_standard_error(scores)
        if (chosen != selected or list(eligible) != list(off["eligible_models"])
                or float(threshold) != float(off["one_standard_error_threshold"])):
            raise ValueError(f"{name}: select_one_standard_error on the random-forest "
                             f"and MLP scores disagrees with option_off")

    for key in ("candidate_test_family_mae", "candidate_validation_family_mae"):
        if key in out_meta and isinstance(out_meta[key], dict):
            out_meta[key] = {k: v for k, v in out_meta[key].items()
                             if k in BASELINE_CANDIDATES}
    if isinstance(out_meta.get("candidate_prediction_warnings"), list):
        out_meta["candidate_prediction_warnings"] = [
            w for w in out_meta["candidate_prediction_warnings"]
            if w.get("phase") not in STRONG_PHASES]
    if isinstance(out_meta.get("timing"), dict):
        out_meta["timing"] = {k: v for k, v in out_meta["timing"].items()
                              if k not in STRONG_TIMING_KEYS}

    arrays: dict[str, np.ndarray] = {}
    if npz_data:
        arrays["validation"] = np.array(npz_data[f"validation__{selected}"])
        arrays["test"] = np.array(npz_data[f"test__{selected}"])
        for cand in BASELINE_CANDIDATES:
            for role in ("validation", "test"):
                arrays[f"{role}__{cand}"] = np.array(npz_data[f"{role}__{cand}"])
    return out_meta, arrays


def derive_other_fit(meta: dict, npz_data: dict[str, np.ndarray]
                     ) -> tuple[dict, dict[str, np.ndarray]]:
    """Arm A and GBT fits: drop the strong-learner fields only."""
    out_meta = {k: v for k, v in meta.items() if k not in STRONG_FIELDS}
    return out_meta, {k: np.array(v) for k, v in npz_data.items()}


def derive_directory(fit_dirs: list[Path], out_dir: Path, *, dry_run: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for fit_dir in fit_dirs:
        for json_path in sorted(fit_dir.glob("*.json")):
            if json_path.name.endswith(".tmp.json"):
                continue
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("strong_learners") is not True:
                raise SystemExit(f"{json_path} is not a strong-learner fit")
            npz_path = json_path.with_suffix(".npz")
            npz_data: dict[str, np.ndarray] = {}
            if npz_path.exists():
                with np.load(npz_path) as npz:
                    npz_data = {k: npz[k] for k in npz.files}
            if meta.get("fit_arm") in common.LIAO_ARMS:
                derived_meta, arrays = derive_liao_fit(meta, npz_data)
            else:
                derived_meta, arrays = derive_other_fit(meta, npz_data)
            if not dry_run:
                # The writer the fitting code uses: same NPZ and JSON format.
                common._write_fit({"fit_dir": str(out_dir), "stem": json_path.stem},
                                  derived_meta, arrays)
            count += 1
            if count % 50 == 0:
                print(f"Derived {count} fits...", flush=True)
    print(f"Successfully derived {count} baseline fits into {out_dir}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, action="append", required=True,
                        help="Input fits directory (may be repeated)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for derived baseline fits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute derivations without writing to disk")
    args = parser.parse_args()
    derive_directory(args.fits, args.out, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
