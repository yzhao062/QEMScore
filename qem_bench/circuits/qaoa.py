"""QAOA-MaxCut circuits on deterministic seeded graph instances.

The cost operator used here is ``C_ZZ = sum_(i,j) Z_i Z_j``. Qiskit's convention
is ``RZZ(theta) = exp(-i theta Z Z / 2)``, so ``RZZ(2 * gamma)`` implements
``exp(-i gamma Z Z)`` and a complete cost layer implements
``exp(-i gamma C_ZZ)``. The usual cut-value operator
``sum_(i,j) (I - Z_i Z_j) / 2`` differs by a sign, a factor of two, and an
irrelevant identity term.

Every circuit starts in the uniform superposition, then alternates cost layers
with ``RX(2 * beta)`` mixer layers. Graph edges and angles are stored directly in
the parameters, so rebuilding a sampled circuit does not replay random draws.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit

MAX_QUBITS = 12
GRAPH_CLASSES = ("path", "cycle", "erdos_renyi", "3_regular")

_GRAPH_CLASS_ALIASES = {
    "path": "path",
    "cycle": "cycle",
    "erdos_renyi": "erdos_renyi",
    "erdos-renyi": "erdos_renyi",
    "3_regular": "3_regular",
    "3-regular": "3_regular",
}


@dataclass(frozen=True)
class QAOAParams:
    n_qubits: int
    graph_class: str
    edges: tuple[tuple[int, int], ...]
    p: int
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    circuit_seed: int
    instance: int
    edge_probability: float | None = None

    def __post_init__(self) -> None:
        if type(self.n_qubits) is not int:
            raise ValueError("n_qubits must be an integer")
        if type(self.p) is not int:
            raise ValueError("p must be an integer")
        if type(self.circuit_seed) is not int or self.circuit_seed < 0:
            raise ValueError("circuit_seed must be a nonnegative integer")
        if type(self.instance) is not int or self.instance < 0:
            raise ValueError("instance must be a nonnegative integer")

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_graph_class(name: str) -> str:
    try:
        return _GRAPH_CLASS_ALIASES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown graph class {name!r}; expected one of {GRAPH_CLASSES}"
        ) from exc


def _canonical_edge(q0: int, q1: int) -> tuple[int, int]:
    return (q0, q1) if q0 < q1 else (q1, q0)


def _graph_edges(
    graph_class: str,
    n_qubits: int,
    rng: np.random.Generator,
    edge_probability: float | None,
) -> tuple[tuple[int, int], ...]:
    if graph_class == "path":
        return tuple((q, q + 1) for q in range(n_qubits - 1))

    if graph_class == "cycle":
        edges = {(q, q + 1) for q in range(n_qubits - 1)}
        edges.add((0, n_qubits - 1))
        return tuple(sorted(edges))

    if graph_class == "erdos_renyi":
        assert edge_probability is not None
        return tuple(
            edge
            for edge in combinations(range(n_qubits), 2)
            if rng.random() < edge_probability
        )

    # Pair three stubs per vertex, rejecting pairings with self-loops or repeated
    # edges. This configuration-model sampler is deterministic from the passed
    # generator and can produce distinct 3-regular topologies.
    stubs = np.repeat(np.arange(n_qubits), 3)
    for _ in range(10_000):
        pairing = rng.permutation(stubs)
        edges: set[tuple[int, int]] = set()
        valid = True
        for offset in range(0, len(pairing), 2):
            q0 = int(pairing[offset])
            q1 = int(pairing[offset + 1])
            edge = _canonical_edge(q0, q1)
            if q0 == q1 or edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            return tuple(sorted(edges))
    raise RuntimeError("failed to sample a simple 3-regular graph after 10000 pairings")


def sample_qaoa_params(
    rng: np.random.Generator,
    n_qubits_choices: list[int],
    p_choices: list[int],
    graph_classes: list[str],
    instance: int,
    circuit_seed: int,
    er_edge_probability: float | None = None,
) -> QAOAParams:
    """Draw one graph and QAOA configuration from the preset ranges.

    If ``er_edge_probability`` is omitted, an Erdos-Renyi instance first draws its
    probability uniformly from [0.25, 0.75). Each possible edge is then sampled
    independently with that probability. The passed generator controls the graph,
    probability, and angles.
    """
    if not n_qubits_choices or any(
        type(n) is not int or n < 2 or n > MAX_QUBITS
        for n in n_qubits_choices
    ):
        raise ValueError(f"QAOA supports 2 to {MAX_QUBITS} qubits")
    qubit_choices = tuple(n_qubits_choices)

    if not p_choices or any(
        type(depth) is not int or depth not in (1, 2) for depth in p_choices
    ):
        raise ValueError("QAOA p choices must be drawn from {1, 2}")
    depths = tuple(p_choices)

    if type(instance) is not int or instance < 0:
        raise ValueError("instance must be a nonnegative integer")
    if type(circuit_seed) is not int or circuit_seed < 0:
        raise ValueError("circuit_seed must be a nonnegative integer")
    if er_edge_probability is not None and (
        isinstance(er_edge_probability, (bool, np.bool_))
        or not isinstance(
            er_edge_probability, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(er_edge_probability)
        or not 0.0 <= er_edge_probability <= 1.0
    ):
        raise ValueError("er_edge_probability must be a number in [0, 1]")

    classes = tuple(_canonical_graph_class(name) for name in graph_classes)
    if not classes:
        raise ValueError("graph_classes must not be empty")
    graph_class = str(rng.choice(classes))

    eligible_qubits = qubit_choices
    if graph_class == "cycle":
        eligible_qubits = tuple(n for n in qubit_choices if n >= 3)
    elif graph_class == "3_regular":
        eligible_qubits = tuple(n for n in qubit_choices if n >= 4 and n % 2 == 0)
    if not eligible_qubits:
        raise ValueError(f"no n_qubits choice can realize graph class {graph_class!r}")

    n_qubits = int(rng.choice(eligible_qubits))
    p = int(rng.choice(depths))
    edge_probability: float | None = None
    if graph_class == "erdos_renyi":
        edge_probability = (
            float(rng.uniform(0.25, 0.75))
            if er_edge_probability is None
            else float(er_edge_probability)
        )
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("er_edge_probability must lie in [0, 1]")

    edges = _graph_edges(graph_class, n_qubits, rng, edge_probability)
    gammas = tuple(float(value) for value in rng.uniform(0.0, np.pi, size=p))
    betas = tuple(float(value) for value in rng.uniform(0.0, np.pi / 2.0, size=p))
    return QAOAParams(
        n_qubits=n_qubits,
        graph_class=graph_class,
        edges=edges,
        p=p,
        gammas=gammas,
        betas=betas,
        circuit_seed=circuit_seed,
        instance=instance,
        edge_probability=edge_probability,
    )


def build_qaoa_circuit(params: QAOAParams) -> QuantumCircuit:
    """Build a measurement-free QAOA circuit from stored graph parameters."""
    if not 2 <= params.n_qubits <= MAX_QUBITS:
        raise ValueError(f"QAOA supports 2 to {MAX_QUBITS} qubits")
    _canonical_graph_class(params.graph_class)
    if params.p not in (1, 2):
        raise ValueError("QAOA p must be 1 or 2")
    if len(params.gammas) != params.p or len(params.betas) != params.p:
        raise ValueError("QAOA angle-list lengths must equal p")
    if not all(np.isfinite(angle) for angle in (*params.gammas, *params.betas)):
        raise ValueError("QAOA angles must be finite")
    if params.edge_probability is not None and not 0.0 <= params.edge_probability <= 1.0:
        raise ValueError("edge_probability must lie in [0, 1]")

    seen_edges: set[tuple[int, int]] = set()
    for edge in params.edges:
        if len(edge) != 2:
            raise ValueError(f"invalid edge {edge!r}")
        if any(type(endpoint) is not int for endpoint in edge):
            raise ValueError(f"invalid edge {edge!r}; endpoints must be integers")
        q0, q1 = edge
        if q0 == q1 or not (0 <= q0 < params.n_qubits and 0 <= q1 < params.n_qubits):
            raise ValueError(f"invalid edge {edge!r} for {params.n_qubits} qubits")
        canonical = _canonical_edge(q0, q1)
        if canonical in seen_edges:
            raise ValueError(f"duplicate edge {canonical!r}")
        seen_edges.add(canonical)

    circ = QuantumCircuit(params.n_qubits)
    for q in range(params.n_qubits):
        circ.h(q)
    for layer in range(params.p):
        for q0, q1 in params.edges:
            circ.rzz(2.0 * params.gammas[layer], q0, q1)
        for q in range(params.n_qubits):
            circ.rx(2.0 * params.betas[layer], q)
    return circ
