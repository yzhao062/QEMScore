"""Digital zero-noise extrapolation with deterministic global unitary folding.

The frozen B1 protocol uses noise scale factors ``(1, 3, 5)``. Scale one reuses
the item's stored ``noisy_expectation`` so its shot noise is shared exactly with
the raw estimator. Scales three and five each require one new execution with the
item's shot count. Their sampler and transpiler seeds are derived without mutable
state from a caller-provided :class:`numpy.random.SeedSequence` root.

A barrier-free fold of the untranspiled logical circuit is cancelled by the
installed Qiskit optimization-level-one pass on basis ``[cx, rz, sx, x]``.
Barriered logical folding also survives that pass. B1 deliberately compiles the
unmeasured logical circuit first, then folds the resulting basis circuit, because
this scales the circuit that is actually executed. Barriers at global-fold
boundaries preserve those folds when ``sample_counts`` adds measurements and
transpiles again. They also inhibit cross-boundary optimization, so this
transpile-then-fold choice is part of the protocol and must accompany gate counts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from qemscore.circuits.heisenberg import HeisenbergParams, build_heisenberg_circuit
from qemscore.circuits.near_clifford import NearCliffordParams, build_near_clifford_circuit
from qemscore.circuits.qaoa import QAOAParams, build_qaoa_circuit
from qemscore.circuits.random_clifford import (
    RandomCliffordParams,
    build_random_clifford_circuit,
)
from qemscore.circuits.tfi import TFIParams, build_tfi_circuit
from qemscore.noise.models import BASIS_GATES
from qemscore.observables import z_expectation_from_counts
from qemscore.sampling import sample_counts

SCALE_FACTORS: tuple[int, int, int] = (1, 3, 5)
RICHARDSON_WEIGHTS: tuple[float, float, float] = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)

_EXTRAPOLATORS = frozenset({"richardson", "linear"})
_QISKIT_SEED_MOD = 2**31 - 1


def global_fold(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Return ``C (C^dag C)^k`` for the requested odd integer scale factor.

    Barriers separate the global segments so Qiskit's second optimization-level
    one transpilation does not simplify the identity folds. The barriers do not
    change the circuit unitary and are not noisy operations in the benchmark model.
    """
    if isinstance(scale_factor, bool) or not isinstance(scale_factor, (int, np.integer)):
        raise TypeError("scale_factor must be an odd integer")
    scale_factor = int(scale_factor)
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor must be a positive odd integer")
    if circuit.num_clbits or any(inst.operation.name == "measure" for inst in circuit.data):
        raise ValueError("global folding requires an unmeasured unitary circuit")

    folded = circuit.copy()
    inverse = circuit.inverse()
    for _ in range((scale_factor - 1) // 2):
        folded.barrier()
        folded.compose(inverse, inplace=True)
        folded.barrier()
        folded.compose(circuit, inplace=True)
    return folded


def fold_for_execution(
    logical_circuit: QuantumCircuit, scale_factor: int, transpile_seed: int
) -> QuantumCircuit:
    """Compile a logical circuit once, then globally fold the resulting basis circuit.

    ``sample_counts`` remains the only execution path and performs the final
    measurement-bearing transpilation. See the module docstring for why this B1
    implementation uses the transpile-then-fold fallback.
    """
    compiled = transpile(
        logical_circuit,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=int(transpile_seed % _QISKIT_SEED_MOD),
    )
    return global_fold(compiled, scale_factor)


def extrapolate_zero_noise(
    expectation_values: Sequence[float], extrapolator: str = "richardson"
) -> float:
    """Extrapolate the three expectations at scales ``(1, 3, 5)`` to zero noise."""
    values = np.asarray(expectation_values, dtype=float)
    if values.shape != (len(SCALE_FACTORS),):
        raise ValueError(f"expected {len(SCALE_FACTORS)} expectation values")
    if not np.all(np.isfinite(values)):
        raise ValueError("expectation values must be finite")
    if extrapolator == "richardson":
        return float(np.dot(np.asarray(RICHARDSON_WEIGHTS), values))
    if extrapolator == "linear":
        # Ordinary least squares on all three frozen scale factors. The constant
        # coefficient is the fitted value at zero noise.
        return float(np.polynomial.polynomial.polyfit(SCALE_FACTORS, values, deg=1)[0])
    raise ValueError(f"extrapolator must be one of {sorted(_EXTRAPOLATORS)}")


def _seed_root(seed_stream: np.random.SeedSequence | int) -> np.random.SeedSequence:
    if isinstance(seed_stream, np.random.SeedSequence):
        return seed_stream
    if isinstance(seed_stream, (int, np.integer)) and not isinstance(seed_stream, bool):
        return np.random.SeedSequence(int(seed_stream))
    raise TypeError("seed_stream must be an integer or numpy.random.SeedSequence")


def _scale_seeds(
    root: np.random.SeedSequence, measurement_group: str, scale_factor: int
) -> tuple[int, int]:
    """Derive order-independent seeds for one measurement group and scale."""
    digest = hashlib.sha256(measurement_group.encode("utf-8")).digest()
    group_key = tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
    child = np.random.SeedSequence(
        root.entropy,
        spawn_key=(*root.spawn_key, *group_key, int(scale_factor)),
        pool_size=root.pool_size,
    )
    state = child.generate_state(2, dtype=np.uint32)
    return int(state[0]), int(state[1])


def _require_fields(item: Mapping[str, object], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in item]
    if missing:
        raise ValueError(
            f"item {item.get('item_id', '?')} lacks circuit rebuild fields: {missing}"
        )


def _rebuild_circuit(item: Mapping[str, object]) -> QuantumCircuit:
    family = item.get("family")
    common = ("n_qubits", "circuit_seed", "instance")
    if family == "tfi":
        _require_fields(item, (*common, "steps", "j", "h", "dt"))
        return build_tfi_circuit(
            TFIParams(
                n_qubits=int(item["n_qubits"]),
                steps=int(item["steps"]),
                j=float(item["j"]),
                h=float(item["h"]),
                dt=float(item["dt"]),
                circuit_seed=int(item["circuit_seed"]),
                instance=int(item["instance"]),
            )
        )
    if family == "qaoa":
        required = (*common, "graph_class", "edges", "p", "gammas", "betas")
        _require_fields(item, required)
        edge_probability = item.get("edge_probability")
        return build_qaoa_circuit(
            QAOAParams(
                n_qubits=int(item["n_qubits"]),
                graph_class=str(item["graph_class"]),
                edges=tuple(tuple(int(q) for q in edge) for edge in item["edges"]),
                p=int(item["p"]),
                gammas=tuple(float(value) for value in item["gammas"]),
                betas=tuple(float(value) for value in item["betas"]),
                circuit_seed=int(item["circuit_seed"]),
                instance=int(item["instance"]),
                edge_probability=(
                    None if edge_probability is None else float(edge_probability)
                ),
            )
        )
    if family == "heisenberg":
        _require_fields(item, (*common, "steps", "jx", "jy", "jz", "dt"))
        return build_heisenberg_circuit(
            HeisenbergParams(
                n_qubits=int(item["n_qubits"]),
                steps=int(item["steps"]),
                jx=float(item["jx"]),
                jy=float(item["jy"]),
                jz=float(item["jz"]),
                dt=float(item["dt"]),
                circuit_seed=int(item["circuit_seed"]),
                instance=int(item["instance"]),
            )
        )
    if family == "random_clifford":
        _require_fields(item, (*common, "depth"))
        return build_random_clifford_circuit(
            RandomCliffordParams(
                n_qubits=int(item["n_qubits"]),
                depth=int(item["depth"]),
                circuit_seed=int(item["circuit_seed"]),
                instance=int(item["instance"]),
            )
        )
    if family == "near_clifford":
        _require_fields(item, (*common, "depth", "non_clifford_count", "theta"))
        return build_near_clifford_circuit(
            NearCliffordParams(
                n_qubits=int(item["n_qubits"]),
                depth=int(item["depth"]),
                non_clifford_count=int(item["non_clifford_count"]),
                theta=float(item["theta"]),
                circuit_seed=int(item["circuit_seed"]),
                instance=int(item["instance"]),
            )
        )
    raise ValueError(f"ZNE B1 does not support circuit family {family!r}")


def _execution_signature(item: Mapping[str, object]) -> tuple[object, ...]:
    family = item.get("family")
    family_fields: dict[object, tuple[str, ...]] = {
        "tfi": ("steps", "j", "h", "dt"),
        "qaoa": ("graph_class", "edges", "p", "gammas", "betas", "edge_probability"),
        "heisenberg": ("steps", "jx", "jy", "jz", "dt"),
        "random_clifford": ("depth",),
        "near_clifford": ("depth", "non_clifford_count", "theta"),
    }
    if family not in family_fields:
        raise ValueError(f"ZNE B1 does not support circuit family {family!r}")

    def freeze(value: object) -> object:
        if isinstance(value, list):
            return tuple(freeze(nested) for nested in value)
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(nested)) for key, nested in value.items()))
        return value

    fields = (
        "family",
        "n_qubits",
        "circuit_seed",
        "instance",
        "severity",
        "shots",
        *family_fields[family],
    )
    return (
        *(freeze(item.get(field)) for field in fields),
        item.get("noise_family", "depolarizing_readout"),
    )


def _z_support(item: Mapping[str, object]) -> tuple[int, ...]:
    label = item.get("pauli_label")
    n_qubits = int(item["n_qubits"])
    if not isinstance(label, str) or len(label) != n_qubits or set(label) - {"I", "Z"}:
        raise ValueError("ZNE B1 requires an n_qubits-long I/Z pauli_label")
    return tuple(n_qubits - 1 - index for index, symbol in enumerate(label) if symbol == "Z")


class ZNEMitigator:
    """Training-free B1 digital ZNE estimator.

    The default is three-point Richardson extrapolation at scales ``(1, 3, 5)``.
    That choice is frozen for B1 and is never selected using test labels. ``linear``
    is exposed only as a predeclared ablation; choosing it for an experiment must
    happen on source-domain validation data before test evaluation. ``fit`` is a
    no-op because neither option learns parameters.

    ``predict`` returns ``(predictions, extra_evals_per_group)``. Each entry of the
    second array is ``2 * shots`` for one measurement group. Scale one is not
    executed here: every row reuses its exact stored base estimate. Scales three
    and five each execute once per group, and every sibling observable derives
    from the same folded counts.
    """

    def __init__(self, extrapolator: str = "richardson") -> None:
        self.extrapolator = extrapolator

    def fit(self, train_items: Sequence[Mapping[str, object]]) -> "ZNEMitigator":
        del train_items
        return self

    def predict(
        self,
        test_items: Sequence[Mapping[str, object]],
        *,
        seed_stream: np.random.SeedSequence | int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate folded test circuits using deterministic seeds from ``seed_stream``."""
        if self.extrapolator not in _EXTRAPOLATORS:
            raise ValueError(f"extrapolator must be one of {sorted(_EXTRAPOLATORS)}")
        root = _seed_root(seed_stream)
        predictions = np.empty(len(test_items), dtype=float)
        groups: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
        for index, item in enumerate(test_items):
            measurement_group = item.get("measurement_group")
            if not isinstance(measurement_group, str) or not measurement_group:
                raise ValueError("each ZNE item requires a nonempty measurement_group")
            groups.setdefault(measurement_group, []).append((index, item))

        extra_evals = np.empty(len(groups), dtype=np.int64)
        for group_index, measurement_group in enumerate(sorted(groups)):
            entries = groups[measurement_group]
            reference = entries[0][1]
            signature = _execution_signature(reference)
            if any(_execution_signature(item) != signature for _, item in entries[1:]):
                raise ValueError(
                    f"measurement group {measurement_group} disagrees on ZNE execution fields"
                )

            shots = int(reference["shots"])
            if shots <= 0:
                raise ValueError("shots must be positive")
            logical = _rebuild_circuit(reference)
            folded_counts: list[dict[str, int]] = []
            for scale_factor in SCALE_FACTORS[1:]:
                sampler_seed, transpile_seed = _scale_seeds(
                    root, measurement_group, scale_factor
                )
                folded = fold_for_execution(logical, scale_factor, transpile_seed)
                counts, _ = sample_counts(
                    folded,
                    severity=str(reference["severity"]),
                    shots=shots,
                    sampler_seed=sampler_seed,
                    transpile_seed=transpile_seed,
                    noise_family=str(
                        reference.get("noise_family", "depolarizing_readout")
                    ),
                )
                folded_counts.append(counts)

            for index, item in entries:
                base = float(item["noisy_expectation"])
                if not np.isfinite(base):
                    raise ValueError("noisy_expectation must be finite")
                support = _z_support(item)
                values = [base]
                for counts in folded_counts:
                    estimate, _ = z_expectation_from_counts(counts, support, shots)
                    values.append(estimate)
                predictions[index] = extrapolate_zero_noise(values, self.extrapolator)

            extra_evals[group_index] = (len(SCALE_FACTORS) - 1) * shots

        return predictions, extra_evals


__all__ = [
    "RICHARDSON_WEIGHTS",
    "SCALE_FACTORS",
    "ZNEMitigator",
    "extrapolate_zero_noise",
    "fold_for_execution",
    "global_fold",
]
