"""Z-type Pauli observables: labels, count-based estimation, and standard errors.

The walking-skeleton slice restricts observables to Z-type Pauli strings so a single
computational-basis measurement estimates every observable. Basis-rotated observables
arrive with the full benchmark.
"""

from __future__ import annotations

import math


def z_support_label(n_qubits: int, support: tuple[int, ...]) -> str:
    """Return the SparsePauliOp label with Z on `support`, identity elsewhere.

    Qiskit labels are little-endian: the rightmost character is qubit 0.
    """
    chars = ["I"] * n_qubits
    for q in support:
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit {q} outside register of size {n_qubits}")
        chars[n_qubits - 1 - q] = "Z"
    return "".join(chars)


def z_expectation_from_counts(
    counts: dict[str, int], support: tuple[int, ...], shots: int
) -> tuple[float, float]:
    """Estimate <Z...Z on support> from computational-basis counts.

    Returns (estimate, standard_error). Count keys follow Qiskit bit order
    (rightmost bit is qubit 0); keys may contain spaces from multiple registers.
    """
    total = 0.0
    for key, cnt in counts.items():
        bits = key.replace(" ", "")
        parity = sum(int(bits[len(bits) - 1 - q]) for q in support) % 2
        total += (-1 if parity else 1) * cnt
    est = total / shots
    stderr = math.sqrt(max(0.0, 1.0 - est * est) / shots)
    return est, stderr
