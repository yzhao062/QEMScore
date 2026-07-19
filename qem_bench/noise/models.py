"""Noise models for the walking-skeleton slice.

One family: depolarizing plus readout error. Severity levels L1 to L3 share the grid
shape of the benchmark design; the numeric values below are slice placeholders until
the protocol freeze fixes the calibrated grid. RZ is treated as virtual (noiseless),
which matches standard practice on transmon-style basis sets.
"""

from __future__ import annotations

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS_GATES = ["cx", "rz", "sx", "x"]

# severity -> (1q depolarizing prob, 2q depolarizing prob, readout flip prob)
SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {"p1": 0.001, "p2": 0.010, "p_ro": 0.010},
    "L2": {"p1": 0.003, "p2": 0.030, "p_ro": 0.030},
    "L3": {"p1": 0.010, "p2": 0.060, "p_ro": 0.060},
}


def build_noise_model(severity: str) -> NoiseModel:
    cfg = SEVERITY_GRID[severity]
    nm = NoiseModel(basis_gates=BASIS_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(cfg["p1"], 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(cfg["p2"], 2), ["cx"])
    p = cfg["p_ro"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm
