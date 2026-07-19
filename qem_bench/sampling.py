"""Noisy sampling through Aer with deterministic seeds.

Measurement-group contract: all Z-type observables of one (circuit, noise, shots)
configuration share a single computational-basis measurement. The group is executed
once, every observable expectation derives from the same counts, and the ledger
charges the group's shots once. Every stochastic stage is seeded from the manifest:
transpilation uses the circuit seed and the simulator uses the group's sampler seed,
so a dataset regenerates bit-identically from its manifest.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qem_bench.noise.models import BASIS_GATES, build_noise_model

_MOD = 2**31 - 1


def sample_counts(
    circuit: QuantumCircuit,
    severity: str,
    shots: int,
    sampler_seed: int,
    transpile_seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Execute one measurement group under noise; return (counts, structure features).

    Features describe the transpiled circuit actually executed: two-qubit gate count
    and critical-path depth (including the measurement layer).
    """
    measured = circuit.copy()
    measured.measure_all()
    tcirc = transpile(
        measured,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=int(transpile_seed % _MOD),
    )
    backend = AerSimulator(
        noise_model=build_noise_model(severity),
        seed_simulator=int(sampler_seed % _MOD),
    )
    counts = backend.run(tcirc, shots=shots).result().get_counts()
    features = {
        "two_qubit_gates": int(tcirc.count_ops().get("cx", 0)),
        "transpiled_depth": int(tcirc.depth()),
    }
    return counts, features
