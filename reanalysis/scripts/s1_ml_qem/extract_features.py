"""PHASE 1 acquisition/verification for S1 (ML-QEM), scoped experiment:
4-qubit TFIM Trotter on ibm_algiers hardware (release notebooks h36 / demos/demo2).

Runs under the old-qiskit venv (qiskit-terra 0.24.2, numpy 1.26) because the released
circuit pickles were written with that era's CircuitInstruction layout.

It does NOT fit anything. It:
  * asserts released row counts and schema,
  * runs the RELEASED encoder function bodies (lifted verbatim out of docs/tutorials/mlp.py
    by AST, executed against a numpy-backed torch shim) to build X and y,
  * records the split exactly as the released notebook builds it,
  * runs the leak checks,
  * writes an .npz for the later scoring phase.
"""
import ast
import json
import os
import pickle
import sys
import hashlib

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "repo")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "phase1")
TUT = os.path.join(REPO, "docs", "tutorials")
DATA = os.path.join(TUT, "data", "ising_init_from_qasm_hardware", "shuffled")
os.makedirs(OUT, exist_ok=True)

FAIL = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[%s] %s %s" % (status, label, detail))
    if not cond:
        FAIL.append(label + " " + detail)


# ---------------------------------------------------------------- torch shim
class _TorchShim:
    float32 = np.float32

    @staticmethod
    def zeros(shape):
        return np.zeros(tuple(shape), dtype=np.float32)

    @staticmethod
    def tensor(x, dtype=None):
        if dtype is None:
            return np.asarray(x, dtype=np.float64)
        return np.asarray(x, dtype=dtype)


def load_released_encoders():
    """Lift count_gates_by_rotation_angle and encode_data_v2_ecr verbatim from mlp.py."""
    src_path = os.path.join(TUT, "mlp.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    wanted = {"count_gates_by_rotation_angle", "encode_data_v2_ecr"}
    ns = {"np": np, "torch": _TorchShim}
    picked = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            seg = ast.get_source_segment(src, node)
            picked[node.name] = seg
            exec(compile(ast.Module(body=[node], type_ignores=[]), src_path, "exec"), ns)
    assert wanted <= set(picked), picked.keys()
    digest = hashlib.sha256(("\n".join(picked[k] for k in sorted(picked))).encode()).hexdigest()[:16]
    print("released encoder bodies lifted; sha256[:16]=%s" % digest)
    return ns["count_gates_by_rotation_angle"], ns["encode_data_v2_ecr"], digest


# ---------------------------------------------------------------- load bytes
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    _, encode_data_v2_ecr, enc_digest = load_released_encoders()

    print("\n--- released artifact digests ---")
    manifest = {}
    for rel in ["results.pk", "results_unshuffled.pk", "index_order.json"]:
        p = os.path.join(DATA, rel)
        manifest[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
        print("  %-24s %s  %d bytes" % (rel, manifest[rel]["sha256"][:16], manifest[rel]["bytes"]))

    # ---- results
    with open(os.path.join(DATA, "results.pk"), "rb") as f:
        res = pickle.load(f)
    check("results.pk keys", sorted(res.keys()) == ["ideal", "noisy", "zne_mitigated"], str(sorted(res.keys())))
    check("results.pk ideal rows == 3000", len(res["ideal"]) == 3000, "got %d" % len(res["ideal"]))
    check("results.pk ideal width == 4", all(len(r) == 4 for r in res["ideal"]))

    noisy = np.array(res["noisy"]).reshape(-1, 4)
    zne = np.array(res["zne_mitigated"]).reshape(-1, 4)
    check("noisy reshaped to (3000,4)", noisy.shape == (3000, 4), str(noisy.shape))
    check("zne reshaped to (3000,4)", zne.shape == (3000, 4), str(zne.shape))

    with open(os.path.join(DATA, "index_order.json")) as f:
        index_order = json.load(f)
    check("index_order is a permutation of 0..2999",
          sorted(index_order) == list(range(3000)), "len=%d" % len(index_order))

    def unshuffle(lst):
        out = [None] * len(lst)
        for i, index in enumerate(index_order):
            out[index] = lst[i]
        return out

    ideal_u = unshuffle([list(map(float, r)) for r in res["ideal"]])
    noisy_u = unshuffle(noisy.tolist())
    zne_u = unshuffle(zne.tolist())

    # cross-check against the released pre-unshuffled copy
    with open(os.path.join(DATA, "results_unshuffled.pk"), "rb") as f:
        resu = pickle.load(f)
    check("unshuffle(ideal) matches results_unshuffled.pk",
          np.allclose(np.array(ideal_u), np.array(resu["ideal"])))
    check("unshuffle(noisy) matches results_unshuffled.pk",
          np.allclose(np.array(noisy_u), np.array(resu["noisy"])))
    check("unshuffle(zne) matches results_unshuffled.pk",
          np.allclose(np.array(zne_u), np.array(resu["zne_mitigated"])))

    # ---- circuits
    batch_files = sorted(f for f in os.listdir(DATA) if f.startswith("batch_") and f.endswith(".pk"))
    check("60 circuit batch files", len(batch_files) == 60, "got %d" % len(batch_files))
    all_circuits, batch_meta = [], []
    for bf in batch_files:
        with open(os.path.join(DATA, bf), "rb") as f:
            loaded = pickle.load(f)
        batch_meta.append({k: loaded[k] for k in loaded if k != "circuit_batch"})
        all_circuits.extend(loaded["circuit_batch"])
    check("3000 circuits total", len(all_circuits) == 3000, "got %d" % len(all_circuits))
    check("all batches hold 50 circuits", all(m["batch_size"] == 50 for m in batch_meta))

    # batch index_order_batch should concatenate to the global index_order
    concat = []
    for m in batch_meta:
        concat.extend(list(m["index_order_batch"]))
    check("concat(index_order_batch) == index_order.json", concat == list(index_order),
          "len=%d" % len(concat))

    circuits_u = unshuffle(all_circuits)

    nq = sorted({c.num_qubits for c in circuits_u})
    print("  circuit num_qubits values:", nq)
    print("  example count_ops:", dict(circuits_u[0].count_ops()))

    # ---- released split, exactly as h36 / demo2 build it
    all_ideal, all_noisy, all_zne, all_circ, all_steps, all_srcrow = [], [], [], [], [], []
    for step, i in enumerate(range(0, 2000, 200)):
        all_ideal.extend(ideal_u[i:i + 200]); all_noisy.extend(noisy_u[i:i + 200])
        all_zne.extend(zne_u[i:i + 200]); all_circ.extend(circuits_u[i:i + 200])
        all_steps.extend([step] * 200); all_srcrow.extend(range(i, i + 200))
    for step, i in enumerate(range(2000, 3000, 100)):
        all_ideal.extend(ideal_u[i:i + 100]); all_noisy.extend(noisy_u[i:i + 100])
        all_zne.extend(zne_u[i:i + 100]); all_circ.extend(circuits_u[i:i + 100])
        all_steps.extend([step] * 100); all_srcrow.extend(range(i, i + 100))
    check("assembled 3000 rows", len(all_steps) == 3000, "got %d" % len(all_steps))

    order = sorted(range(len(all_steps)), key=lambda k: all_steps[k])  # stable, like the notebook
    ideal_s = [all_ideal[k] for k in order]
    noisy_s = [all_noisy[k] for k in order]
    zne_s = [all_zne[k] for k in order]
    circ_s = [all_circ[k] for k in order]
    step_s = [all_steps[k] for k in order]
    src_s = [all_srcrow[k] for k in order]

    counts = {s: step_s.count(s) for s in sorted(set(step_s))}
    check("300 rows per Trotter step", set(counts.values()) == {300}, str(counts))

    TRAIN_PER_STEP, TEST_PER_STEP = 50, 250
    train_idx, test_idx = [], []
    seen = {}
    for pos, s in enumerate(step_s):
        seen.setdefault(s, []).append(pos)
    for s, positions in seen.items():
        train_idx.extend(positions[:TRAIN_PER_STEP])
        test_idx.extend(positions[-TEST_PER_STEP:])
    check("train rows == 500", len(train_idx) == 500, "got %d" % len(train_idx))
    check("test rows == 2500", len(test_idx) == 2500, "got %d" % len(test_idx))
    check("train and test disjoint", set(train_idx).isdisjoint(test_idx))
    check("train+test covers all 3000", len(set(train_idx) | set(test_idx)) == 3000)

    # ---- released encoding
    train_c = [circ_s[i] for i in train_idx]
    test_c = [circ_s[i] for i in test_idx]
    X_train, y_train = encode_data_v2_ecr(train_c, [ideal_s[i] for i in train_idx],
                                          [noisy_s[i] for i in train_idx], obs_size=4, two_q_gate="cx")
    X_test, y_test = encode_data_v2_ecr(test_c, [ideal_s[i] for i in test_idx],
                                        [noisy_s[i] for i in test_idx], obs_size=4, two_q_gate="cx")
    print("\n  X_train", X_train.shape, "y_train", y_train.shape)
    print("  X_test ", X_test.shape, "y_test ", y_test.shape)

    n_gate, n_angle, n_obs = 5, X_train.shape[1] - 5 - 4, 4
    check("feature width = 5 gate counts + angle bins + 4 noisy",
          X_train.shape[1] == n_gate + n_angle + n_obs, "width=%d angle_bins=%d" % (X_train.shape[1], n_angle))

    # ---- LEAK CHECK 1: target must not appear in any feature column
    Xall = np.vstack([X_train, X_test])
    yall = np.vstack([y_train, y_test])
    hits = []
    for q in range(4):
        col = yall[:, q]
        for j in range(Xall.shape[1]):
            f = Xall[:, j]
            if np.allclose(f, col, atol=1e-6):
                hits.append(("identity", q, j))
            elif np.std(col) > 1e-12 and np.std(f) > 1e-12:
                r = float(np.corrcoef(f, col)[0, 1])
                if abs(r) > 0.999:
                    hits.append(("corr>0.999", q, j, r))
    check("LEAK-1 target absent from features (no identity / |r|>0.999 column)", not hits, str(hits[:6]))

    # the noisy tail must be exactly the execution-derived block, not the target
    tail = Xall[:, -4:]
    check("last 4 columns are the noisy execution-derived block",
          np.allclose(tail, np.vstack([np.array([noisy_s[i] for i in train_idx]),
                                       np.array([noisy_s[i] for i in test_idx])]).astype(np.float32),
                      atol=1e-6))
    d_ty = float(np.max(np.abs(tail - yall)))
    print("  max |noisy - ideal| over all rows: %.4f (0 would mean the target leaked)" % d_ty)

    # ---- LEAK CHECK 2: no oracle quantity in features
    # candidates that must be absent: ideal (target), zne (measured comparator, still not a feature)
    zne_all = np.vstack([np.array([zne_s[i] for i in train_idx]), np.array([zne_s[i] for i in test_idx])])
    zne_hits = []
    for q in range(4):
        for j in range(Xall.shape[1]):
            if np.allclose(Xall[:, j], zne_all[:, q], atol=1e-6):
                zne_hits.append((q, j))
    check("LEAK-2 no ZNE comparator column inside X", not zne_hits, str(zne_hits[:6]))

    # ---- LEAK CHECK 4: does the split put rows of one circuit on both sides?
    def circ_key(c):
        return hashlib.sha256(repr([(inst.operation.name,
                                     tuple(c.find_bit(q).index for q in inst.qubits),
                                     tuple(float(p) for p in inst.operation.params))
                                    for inst in c.data]).encode()).hexdigest()

    keys_train = [circ_key(c) for c in train_c]
    keys_test = [circ_key(c) for c in test_c]
    overlap = set(keys_train) & set(keys_test)
    check("LEAK-4 no identical circuit on both sides of the split",
          not overlap, "overlapping distinct circuits: %d" % len(overlap))
    dup_all = len(keys_train) + len(keys_test) - len(set(keys_train) | set(keys_test))
    print("  duplicate circuits anywhere in the 3000 rows: %d" % dup_all)

    # feature-row duplicates across the split (descriptor block only)
    def rowkey(a):
        return a.tobytes()
    desc_train = X_train[:, :-4]
    desc_test = X_test[:, :-4]
    dset = set(map(rowkey, desc_test))
    desc_overlap = sum(1 for r in desc_train if rowkey(r) in dset)
    print("  train descriptor rows byte-identical to some test descriptor row: %d / %d"
          % (desc_overlap, len(desc_train)))

    # ---- constant / degenerate columns
    stds = Xall.std(axis=0)
    n_const = int((stds < 1e-12).sum())
    print("  constant feature columns: %d of %d" % (n_const, Xall.shape[1]))
    print("  nonconstant descriptor columns: %d of %d"
          % (int((stds[:-4] >= 1e-12).sum()), Xall.shape[1] - 4))

    # ---- save
    np.savez_compressed(
        os.path.join(OUT, "s1_hw4q.npz"),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        noisy_train=np.array([noisy_s[i] for i in train_idx]),
        noisy_test=np.array([noisy_s[i] for i in test_idx]),
        zne_train=np.array([zne_s[i] for i in train_idx]),
        zne_test=np.array([zne_s[i] for i in test_idx]),
        step_train=np.array([step_s[i] for i in train_idx]),
        step_test=np.array([step_s[i] for i in test_idx]),
        srcrow_train=np.array([src_s[i] for i in train_idx]),
        srcrow_test=np.array([src_s[i] for i in test_idx]),
        n_gate=n_gate, n_angle=n_angle, n_obs=n_obs,
    )
    meta = {
        "system": "S1 ML-QEM (Liao et al., Nature Machine Intelligence 6, 1478-1486, 2024)",
        "repo": "github.com/qiskit-community/ml-qem",
        "branch": "research",
        "commit": "b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3",
        "scoped_experiment": "4-qubit TFIM Trotter on ibm_algiers hardware; RF over tabular features",
        "notebooks": ["docs/tutorials/h36_ising_4q_hardware_plot.ipynb",
                      "docs/demos/demo2_ising_4q_hardware_plot.ipynb",
                      "docs/tutorials/h35_ising_4q_hardware.ipynb (data generation)"],
        "encoder": "docs/tutorials/mlp.py::encode_data_v2_ecr(obs_size=4, two_q_gate='cx')",
        "encoder_bodies_sha256_16": enc_digest,
        "released_predictions": "docs/paper_figures/hardware_over_depth.pk -> df['rfr_list'] (2500 rows)",
        "artifact_digests": manifest,
        "rows_total": 3000, "rows_train": 500, "rows_test": 2500,
        "feature_width": int(X_train.shape[1]),
        "n_gate_count_features": n_gate, "n_angle_bin_features": int(n_angle),
        "n_noisy_expval_features": n_obs,
        "target": "ideal expectation values of Z on each of 4 spins (4-dim), y from encode_data_v2_ecr",
        "qiskit_terra_used_to_read": "0.24.2",
        "failures": FAIL,
    }
    with open(os.path.join(OUT, "s1_hw4q_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nwrote", os.path.join(OUT, "s1_hw4q.npz"))
    print("wrote", os.path.join(OUT, "s1_hw4q_meta.json"))
    print("\nFAILURES:", FAIL if FAIL else "none")


if __name__ == "__main__":
    main()
