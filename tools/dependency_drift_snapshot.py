"""Record dependency-sensitive dataset structure and folded circuits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qem_bench.baselines.zne import (
    SCALE_FACTORS,
    _rebuild_circuit,
    fold_for_execution,
)
from qem_bench.runner.run import _load

PRESETS = ("t0-micro", "t0-smoke")


def _number(value: object) -> str:
    try:
        return float(value).hex()
    except (TypeError, ValueError):
        return str(value)


def _canonical_circuit(circuit: object) -> dict:
    operations = []
    for instruction in circuit.data:
        operation = instruction.operation
        operations.append(
            {
                "name": operation.name,
                "qubits": [circuit.find_bit(qubit).index for qubit in instruction.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in instruction.clbits],
                "params": [_number(param) for param in operation.params],
            }
        )
    return {
        "num_qubits": circuit.num_qubits,
        "global_phase": _number(circuit.global_phase),
        "operations": operations,
    }


def _snapshot_preset(root: Path, preset: str) -> tuple[dict, dict]:
    preset_root = root / preset
    items, manifest = _load(preset_root)
    structures = {
        item["item_id"]: {
            "measurement_group": item["measurement_group"],
            "transpiled_depth": item["transpiled_depth"],
            "two_qubit_gates": item["two_qubit_gates"],
        }
        for item in items
    }

    representatives = {}
    for item in items:
        representatives.setdefault(item["measurement_group"], item)
    folded = {}
    for group, item in sorted(representatives.items()):
        if item["family"] != "tfi":
            raise ValueError(f"frozen drift snapshot does not support {item['family']!r}")
        logical = _rebuild_circuit(item)
        for scale in SCALE_FACTORS:
            circuit = fold_for_execution(logical, scale, int(item["circuit_seed"]))
            canonical = _canonical_circuit(circuit)
            payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            folded[f"{preset}/{group}/scale-{scale}"] = {
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "circuit": canonical,
            }
    return {"dataset_hash": manifest["dataset_hash"], "items": structures}, folded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("requirements/ci-py312.lock")
    )
    args = parser.parse_args()

    datasets = {}
    folded_circuits = {}
    for preset in PRESETS:
        datasets[preset], folded = _snapshot_preset(args.dataset_root, preset)
        folded_circuits.update(folded)

    snapshot = {
        "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "datasets": datasets,
        "folded_circuits": folded_circuits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote dependency drift snapshot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
