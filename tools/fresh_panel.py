#!/usr/bin/env python3
"""Fresh-circuit confirmation panel dataset generation and verification.

Generates fresh campaign split datasets for unreleased seeds and records metadata,
hashes, and environment provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path[:1]:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import qiskit  # noqa: E402
import qiskit_aer  # noqa: E402
import scipy  # noqa: E402

from qemscore.campaign import design  # noqa: E402
from qemscore.campaign.design import SEEDS, campaign_split_spec  # noqa: E402
from qemscore.datasets import split_generate  # noqa: E402
from qemscore.datasets.split_generate import generate_split  # noqa: E402
from qemscore.validation import validate_split_artifact  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_single_seed(
    seed: int,
    out_dir: Path,
    size: int = 640,
    counts: dict[str, int] | None = None,
) -> dict:
    """Generate one fresh dataset, validate it, and write fresh_dataset.json."""
    if seed in SEEDS:
        raise ValueError(
            f"Refusal: seed {seed} is in qemscore.campaign.design.SEEDS ({SEEDS}). "
            "Fresh panel requires unreleased seeds."
        )

    target_dir = out_dir / f"regen-shipped-s{seed}-n{size}"
    if target_dir.exists():
        raise FileExistsError(f"Refusal: output directory {target_dir} already exists.")

    spec = campaign_split_spec("shipped", size, counts=counts)
    started = time.perf_counter()
    generate_split(spec, target_dir, master_seed=seed)
    wall_time = time.perf_counter() - started

    rows, manifest = validate_split_artifact(target_dir)

    split_gen_file = Path(split_generate.__file__).resolve()
    design_file = Path(design.__file__).resolve()
    fresh_panel_file = Path(__file__).resolve()

    items_sha256 = sha256_file(target_dir / "items.jsonl")
    manifest_sha256 = sha256_file(target_dir / "manifest.json")
    split_gen_sha256 = sha256_file(split_gen_file)
    design_sha256 = sha256_file(design_file)
    fresh_panel_sha256 = sha256_file(fresh_panel_file)

    meta = {
        "target_dir": str(target_dir),
        "seed": seed,
        "size": size,
        "counts": counts if counts is not None else manifest.get("counts"),
        "dataset_hash": str(manifest["dataset_hash"]),
        "items_jsonl_sha256": items_sha256,
        "manifest_json_sha256": manifest_sha256,
        "split_generate_sha256": split_gen_sha256,
        "design_sha256": design_sha256,
        "fresh_panel_sha256": fresh_panel_sha256,
        "sha256": {
            "items.jsonl": items_sha256,
            "manifest.json": manifest_sha256,
            "split_generate.py": split_gen_sha256,
            "design.py": design_sha256,
            "fresh_panel.py": fresh_panel_sha256,
            "items_jsonl": items_sha256,
            "manifest_json": manifest_sha256,
            "split_generate_py": split_gen_sha256,
            "design_py": design_sha256,
            "fresh_panel_py": fresh_panel_sha256,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit.__version__,
            "qiskit-aer": qiskit_aer.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "wall_time_seconds": wall_time,
    }

    # Write fresh_dataset.json beside items.jsonl/manifest.json inside target_dir,
    # and also in out_dir named by the dataset folder.
    fresh_json_path = target_dir / "fresh_dataset.json"
    fresh_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{target_dir.name}.fresh_dataset.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Generated fresh dataset for seed {seed} (size {size}) in {target_dir}: "
        f"dataset_hash={manifest['dataset_hash']}, rows={len(rows)}, "
        f"wall_time={wall_time:.2f}s",
        flush=True,
    )
    return meta


def generate_fresh_datasets(
    seeds: Sequence[int],
    out_dir: Path,
    size: int = 640,
    counts: tuple[int, int, int] | dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts_dict = None
    if isinstance(counts, (list, tuple)):
        counts_dict = {"train": int(counts[0]), "validation": int(counts[1]), "test": int(counts[2])}
    elif isinstance(counts, dict):
        counts_dict = counts
    results = []
    for seed in seeds:
        res = generate_single_seed(seed, out, size=size, counts=counts_dict)
        results.append(res)
    return results


def verify_determinism(
    seed: int,
    dataset_dir: Path,
    target_dir: Path | None = None,
    size: int = 640,
    counts: tuple[int, int, int] | dict[str, int] | None = None,
) -> dict[str, Any]:
    first_dir = Path(dataset_dir).resolve()
    if not first_dir.exists():
        raise FileNotFoundError(f"First dataset directory does not exist: {first_dir}")
    counts_dict = None
    if isinstance(counts, (list, tuple)):
        counts_dict = {"train": int(counts[0]), "validation": int(counts[1]), "test": int(counts[2])}
    elif isinstance(counts, dict):
        counts_dict = counts
    if target_dir is None:
        second_dir = first_dir.parent / f"verify-regen-shipped-s{seed}-n{size}"
    else:
        second_dir = Path(target_dir).resolve()
    if second_dir.exists():
        shutil.rmtree(second_dir)
    spec = campaign_split_spec("shipped", size, counts=counts_dict)
    generate_split(spec, second_dir, master_seed=seed)
    validate_split_artifact(second_dir)
    items1 = (first_dir / "items.jsonl").read_bytes()
    items2 = (second_dir / "items.jsonl").read_bytes()
    if items1 != items2:
        raise ValueError(f"Determinism check failed: items.jsonl differs between {first_dir} and {second_dir}")
    return {
        "verified": True,
        "seed": seed,
        "items_byte_identical": True,
        "dir1": str(first_dir),
        "dir2": str(second_dir),
    }


def cmd_generate(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts = None
    if args.counts:
        counts = {
            "train": int(args.counts[0]),
            "validation": int(args.counts[1]),
            "test": int(args.counts[2]),
        }

    try:
        for seed in args.seeds:
            generate_single_seed(seed, out, size=args.size, counts=counts)
    except (ValueError, FileExistsError) as e:
        raise SystemExit(str(e))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    out = args.out.resolve() if args.out else None
    counts = None
    if args.counts:
        counts = {
            "train": int(args.counts[0]),
            "validation": int(args.counts[1]),
            "test": int(args.counts[2]),
        }

    if args.dataset:
        first_dir = Path(args.dataset).resolve()
        manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
        seed = int(manifest["master_seed"])
        size = int(manifest.get("split_spec", {}).get("role_counts", {}).get("train", 640)) // len(design.FAMILIES)
        if (first_dir / "fresh_dataset.json").exists():
            f_meta = json.loads((first_dir / "fresh_dataset.json").read_text(encoding="utf-8"))
            size = f_meta.get("size", size)
            if counts is None and "counts" in f_meta:
                counts = f_meta["counts"]
        if counts is None:
            counts = {
                role: count // len(design.FAMILIES)
                for role, count in manifest["split_spec"]["role_counts"].items()
            }
    elif args.seed is not None and out is not None:
        seed = args.seed
        size = args.size
        first_dir = out / f"regen-shipped-s{seed}-n{size}"
    else:
        raise SystemExit("verify requires either --dataset DIR or both --seed S and --out DIR")

    if not first_dir.exists():
        raise SystemExit(f"First dataset directory does not exist: {first_dir}")

    if args.target:
        second_dir = Path(args.target).resolve()
    else:
        second_dir = first_dir.parent / f"verify-regen-shipped-s{seed}-n{size}"

    if second_dir.exists():
        shutil.rmtree(second_dir)

    print(f"Verifying determinism for seed {seed}: regenerating into {second_dir}...", flush=True)
    spec = campaign_split_spec("shipped", size, counts=counts)
    generate_split(spec, second_dir, master_seed=seed)
    validate_split_artifact(second_dir)

    items1 = (first_dir / "items.jsonl").read_bytes()
    items2 = (second_dir / "items.jsonl").read_bytes()

    if items1 != items2:
        raise SystemExit(
            f"Determinism check failed: items.jsonl differs between {first_dir} and {second_dir}"
        )

    print(
        json.dumps(
            {
                "verified": True,
                "seed": seed,
                "items_byte_identical": True,
                "dir1": str(first_dir),
                "dir2": str(second_dir),
            },
            indent=2,
        )
    )
    return 0


def cmd_record_caches(args) -> int:
    """Write the SHA-256 of every primary cache in a prepared tree.

    The record has the layout of an analysis A record (``inputs.caches_used``),
    so ``tools/descriptor_ladder.py run --cache-record`` checks a copied tree
    against the machine that prepared it before any fit.
    """
    import hashlib

    cache_dir = Path(args.tree) / "cache"
    keys = sorted(path.stem for path in cache_dir.glob("shipped-s*-n*.pkl"))
    if not keys:
        raise SystemExit(f"{cache_dir}: no primary caches")
    used = {}
    for key in keys:
        digest = hashlib.sha256()
        with open(cache_dir / f"{key}.pkl", "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        used[key] = {"path": f"cache/{key}.pkl", "sha256": digest.hexdigest()}
    record = {"schema": "fresh-cache-record-v1", "tree": str(args.tree),
              "inputs": {"caches_used": used}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(used)} caches)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate fresh campaign datasets")
    gen.add_argument("--seeds", nargs="+", type=int, required=True, help="master seeds (must not be in design.SEEDS)")
    gen.add_argument("--out", type=Path, required=True, help="parent output directory")
    gen.add_argument("--size", type=int, default=640, help="training size (default: 640)")
    gen.add_argument(
        "--counts",
        nargs=3,
        type=int,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="custom per-family circuit counts (rehearsal only)",
    )

    ver = sub.add_parser("verify", help="regenerate seed into second dir and check byte identity")
    ver.add_argument("--seed", type=int, default=None, help="seed to verify")
    ver.add_argument("--dataset", type=Path, default=None, help="existing dataset directory to verify")
    ver.add_argument("--out", type=Path, default=None, help="parent directory containing the dataset")
    ver.add_argument("--target", type=Path, default=None, help="second directory for regeneration")
    ver.add_argument("--size", type=int, default=640, help="training size (default: 640)")
    ver.add_argument(
        "--counts",
        nargs=3,
        type=int,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="custom per-family circuit counts (rehearsal only)",
    )

    rec = sub.add_parser("record-caches", help="record the SHA-256 of a prepared tree's caches")
    rec.add_argument("--tree", type=Path, required=True)
    rec.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "record-caches":
        return cmd_record_caches(args)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "verify":
        return cmd_verify(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
