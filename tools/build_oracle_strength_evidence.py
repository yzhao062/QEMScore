"""Record which oracle variant each fit set needs, from the fits' own records.

Round-9 rule `docs/frozen-rules/2026-10-06-round9-follow-ups.md`, Part C: each
mapping entry states whether its fits read the noise-strength indicator, checked
against the fits' recorded feature names. This script reads one C fit record and
one F fit record per fit set (dataset seed key of the first row, rung R0, learner
seed 1). It writes their file names, SHA-256 values, recorded `strength_indicator`
fields, and whether `noise_strength_L3` is among their feature names to
`tools/bayes_oracle_strength_evidence.json`. Derived fit sets read their own derived
fit records. The fits are the released assets; their local
roots are arguments, so the evidence file names each asset rather than a local
path. `--r8-root` holds the unpacked round-6 to round-8 fits; `--fresh-root` holds
the R0 learner-seed-1 records extracted from `descriptor-information-fresh-v1`.

    python tools/build_oracle_strength_evidence.py --r8-root ~/qemscore-r8 --fresh-root <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STRENGTH_FEATURE = "noise_strength_L3"
LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_sources() -> dict[str, tuple[str, str]]:
    """Fit set name to (release asset label, path below the round-8 root)."""
    out = {
        "original_2048": ("descriptor-information-v1: fits", "assets/descriptor-information-v1/fits"),
        "strength_indicator_2048": ("descriptor-information-strength-v1: fits",
                                    "assets/descriptor-information-strength-v1/fits"),
        "strong_learners_2048": ("descriptor-information-strong-learners-v1: partA-fits",
                                 "assets/strong-learners-v1/partA-fits"),
        "early_stopped_mlp_2048": ("descriptor-information-crossed-follow-ups-v1: C/shots-2048/fits",
                                   "follow-dai/follow/C/shots-2048/fits"),
    }
    for level in LEVELS:
        out[f"sweep_{level}"] = (f"descriptor-information-shot-sweep-v1: runs/shots-{level}/fits",
                                 f"assets/shot-sweep-v1/runs/shots-{level}/fits")
        out[f"fresh_orig_{level}"] = (f"descriptor-information-fresh-v1: fresh/orig/shots-{level}/fits",
                                      f"FRESH:fresh/orig/shots-{level}/fits")
        out[f"fresh_strong_{level}"] = (f"descriptor-information-fresh-v1: fresh/strong/shots-{level}/fits",
                                        f"FRESH:fresh/strong/shots-{level}/fits")
        out[f"fresh_strong_{level}_derived"] = (
            f"descriptor-information-fresh-v1: fresh/strong/shots-{level}/derived/fits",
            f"FRESH:fresh/strong/shots-{level}/derived/fits")
        if level != "2048":
            out[f"follow_b_{level}"] = (f"descriptor-information-crossed-follow-ups-v1: B/shots-{level}/fits",
                                        f"follow-dai/follow/B/shots-{level}/fits")
            out[f"follow_b_{level}_derived"] = (
                f"descriptor-information-crossed-follow-ups-v1: B/shots-{level}/derived/fits",
                f"follow-dai/follow/B/shots-{level}/derived/fits")
    return out


def record_for(fits_dir: Path, key: str, arm: str) -> dict:
    """Read the fit record of one arm at rung R0, learner seed 1."""
    path = fits_dir / f"{key}__R0__k01__{arm}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing evidence fit record {path}")
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    names = doc.get("feature_names")
    if not isinstance(names, list):
        raise KeyError(f"{path} records no feature_names")
    recorded = doc.get("strength_indicator")
    has_feature = STRENGTH_FEATURE in names
    if recorded is not None and bool(recorded) != has_feature:
        raise ValueError(f"{path}: strength_indicator={recorded} disagrees with its feature names")
    return {"file": path.name, "sha256": sha256_file(path), "arm": arm,
            "strength_indicator_field": recorded, "has_strength_feature": has_feature,
            "feature_names_sha256": doc.get("feature_names_sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--r8-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("tools/bayes_oracle_mapping.json"))
    parser.add_argument("--out", type=Path, default=Path("tools/bayes_oracle_strength_evidence.json"))
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    sources = fit_sources()
    evidence = {}
    for entry in mapping["fit_sets"]:
        name = entry["fit_set"]
        if name not in sources:
            raise KeyError(f"no fit source declared for {name}")
        label, rel = sources[name]
        if rel.startswith("FRESH:"):
            fits_dir = args.fresh_root.expanduser() / rel[len("FRESH:"):]
        else:
            fits_dir = args.r8_root.expanduser() / rel
        key = f"shipped-s{int(entry['dataset_seeds'][0])}-n640"
        records = [record_for(fits_dir, key, arm) for arm in ("C", "F")]
        flags = {r["has_strength_feature"] for r in records}
        if len(flags) != 1:
            raise ValueError(f"{name}: C and F disagree on the strength feature")
        evidence[name] = {"source": label, "dataset_key": key, "records": records,
                          "strength_observed": flags.pop()}
    out = {"schema": "bayes-oracle-strength-evidence-v1",
           "strength_feature": STRENGTH_FEATURE,
           "mapping_sha256": sha256_file(args.mapping),
           "fit_sets": evidence}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} for {len(evidence)} fit sets")


if __name__ == "__main__":
    main()
