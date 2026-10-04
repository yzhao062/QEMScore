#!/usr/bin/env python3
"""Pin the fit files the stacking increment reads to the published release assets.

Post hoc support for ``tools/stacking_increment_all.py``. The script streams
each release asset (a ``.tar.xz`` archive) once, without unpacking it, and
hashes the C and F fit records and prediction arrays that the stacking reads
for each fit set: learner seeds 1 to 20 at every Part A rung whose first C fit
exists for all three dataset seeds. For each fit set it writes the SHA-256 of
every consumed file and their digest in the form ``fits_digest`` computes,
together with the archive's own SHA-256 and the SHA-256 of the analysis A
whose labels the stacking copies. ``stacking_increment_all.py --pins`` then
refuses any fit directory whose consumed files differ from these pins, so a
reordered or edited prediction array cannot reach a record.

Usage:
    PYTHONPATH=. python tools/stacking_pins.py \
        --asset descriptor-information-v1=PATH/descriptor-information-v1.tar.xz \
        --set original descriptor-information-v1 descriptor-information-v1/fits \
            artifacts/descriptor-information/analysis-a.json \
        ... \
        --out artifacts/descriptor-information/posthoc-stacking-pins.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))

import stacking_increment_all as sia  # noqa: E402

SCHEMA = "posthoc-stacking-pins-v1"
CONSUMED = re.compile(r"^shipped-s(?P<seed>\d+)-n640__(?P<rung>[^_]+(?:_[^_]+)*)"
                      r"__k(?P<k>\d{2})__(?P<arm>[CF])\.(?P<ext>json|npz)$")


def stream_hashes(archive: Path, prefixes: set[str]) -> dict[str, dict[str, str]]:
    """SHA-256 of every consumable fit file under each prefix, read from the archive."""
    out: dict[str, dict[str, str]] = {prefix: {} for prefix in prefixes}
    with tarfile.open(archive, mode="r|xz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            prefix = str(path.parent)
            if prefix not in out or CONSUMED.match(path.name) is None:
                continue
            digest = hashlib.sha256()
            handle = tar.extractfile(member)
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            out[prefix][path.name] = digest.hexdigest()
    return out


def rungs_present(names: set[str]) -> dict[str, tuple[str, ...]]:
    """The rungs ``rungs_with_fits`` would select, from file names alone."""
    out = {}
    for fam, rungs in sia.mf.RUNGS_PER_FAMILY.items():
        out[fam] = tuple(
            rung for rung in rungs
            if all(f"shipped-s{seed}-n640__{rung}__k01__C.json" in names
                   for seed in sia.mf.SEEDS))
    return out


def consumed_names(rungs: dict[str, tuple[str, ...]]) -> list[str]:
    return sorted({
        f"shipped-s{seed}-n640__{rung}__k{k:02d}__{arm}.{ext}"
        for fam_rungs in rungs.values() for rung in fam_rungs
        for seed in sia.mf.SEEDS for k in range(1, 21)
        for arm in ("C", "F") for ext in ("json", "npz")})


def digest_of(names: list[str], hashes: dict[str, str]) -> str:
    """The digest ``stacking_increment_all.fits_digest`` computes over the same files."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(hashes[name].encode())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", action="append", required=True, metavar="NAME=ARCHIVE")
    parser.add_argument("--set", nargs=4, action="append", required=True,
                        metavar=("NAME", "ASSET", "FITS_PREFIX", "ANALYSIS_A"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    assets = dict(spec.split("=", 1) for spec in args.asset)
    prefixes: dict[str, set[str]] = {}
    for _, asset, prefix, _ in args.set:
        if asset not in assets:
            raise SystemExit(f"--set names an asset not given with --asset: {asset}")
        prefixes.setdefault(asset, set()).add(prefix)
    hashes = {}
    asset_records = {}
    for asset, archive in assets.items():
        print(f"[{asset}] {archive}", flush=True)
        archive = Path(archive)
        asset_records[asset] = {"file": archive.name, "sha256": sia.sha256_file(archive)}
        hashes[asset] = stream_hashes(archive, prefixes.get(asset, set()))

    payload = {"schema": SCHEMA,
               "status": "post hoc; pins the inputs of tools/stacking_increment_all.py",
               "script": "tools/stacking_pins.py",
               "script_sha256": sia.sha256_file(Path(__file__).resolve()),
               "assets": asset_records, "sets": {}}
    for name, asset, prefix, analysis in args.set:
        found = hashes[asset][prefix]
        rungs = rungs_present(set(found))
        names = consumed_names(rungs)
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"[{name}] {len(missing)} consumed files missing from "
                             f"{asset}:{prefix}, for example {missing[0]}")
        payload["sets"][name] = {
            "asset": asset, "fits_prefix": prefix,
            "rungs": {fam: list(r) for fam, r in rungs.items()},
            "n_files": len(names),
            "fits_digest": digest_of(names, found),
            "files_sha256": {n: found[n] for n in names},
            "analysis_a": sia.display(Path(analysis)),
            "analysis_a_sha256": sia.sha256_file(Path(analysis)),
        }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
