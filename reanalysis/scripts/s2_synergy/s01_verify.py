"""
S2 phase 2, step 1: INDEPENDENT VERIFICATION of the release.

Re-derives every fact phase 2 relies on, from the released bytes, before any
arm is fitted. Runs the four leak checks of PLAN-independent-instance-score.md
as refusals. Nothing here fits an arm.

Checks:
  0. zip md5 + file inventory
  1. schema and row counts against an expectation written before loading
  2. leak check 1: the target appears in no arm feature, under no transform
  3. leak check 2: no arm reads an oracle quantity
  4. leak check 4: row-shuffled-split hazard
  5. qubits is constant (rank note for the affine arm)
  6. mz_linear identity (it reads an execution-derived input; NOT arm A)
  7. released R2 reproduction (identity of mz_predicted as arm F)
  8. determinism: mz_exact == f(theta) by exact state-vector simulation
"""
import hashlib
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REL = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "upstream", "s2_synergy"))
DEFAULT_OUT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs", "s2_synergy"))

REL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REL
ZIP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REL, "S2_release_1.4.zip")
OUT = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
os.makedirs(OUT, exist_ok=True)

PANELS = ["Fig8a", "Fig8b", "Fig8c"]
ARRAYS = ["theta", "qubits", "z_noisy", "mz_exact", "mz_zne", "mz_linear", "mz_predicted"]

# ---- expectation written BEFORE loading (leak check 3) ----------------------
EXPECT_SHAPE = {
    "Fig8a": {"theta": (100, 16), "qubits": (100, 16), "z_noisy": (100, 16),
              "mz_exact": (100,), "mz_zne": (100,), "mz_linear": (100,), "mz_predicted": (100,)},
    "Fig8b": {"theta": (100, 20), "qubits": (100, 16), "z_noisy": (100, 16),
              "mz_exact": (100,), "mz_zne": (100,), "mz_linear": (100,), "mz_predicted": (100,)},
    "Fig8c": {"theta": (100, 20), "qubits": (100, 16), "z_noisy": (100, 16),
              "mz_exact": (100,), "mz_zne": (100,), "mz_linear": (100,), "mz_predicted": (100,)},
}
EXPECT_MD5 = "c8939101e068e9e5b23599f0e5e81730"
EXPECT_NPY = 21
PAPER_R2_CNN = [0.98, 0.91, 0.98]
PAPER_R2_LIN = [0.88, 0.54, 0.94]

FAIL = []
rec = {}


def refuse(msg):
    print("\n*** REFUSAL: " + msg)
    FAIL.append(msg)


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ---- 0. zip md5 + inventory -------------------------------------------------
h = hashlib.md5()
with open(ZIP, "rb") as f:
    for blk in iter(lambda: f.read(1 << 20), b""):
        h.update(blk)
md5 = h.hexdigest()
npys = sorted(x for x in os.listdir(REL) if x.endswith(".npy"))
print("=" * 78)
print("0. RELEASE INVENTORY")
print(f"   zip md5           : {md5}  (expected {EXPECT_MD5})")
print(f"   .npy count        : {len(npys)}  (expected {EXPECT_NPY})")
print(f"   all files         : {sorted(os.listdir(REL))}")
rec["zip_md5"] = md5
rec["md5_match"] = (md5 == EXPECT_MD5)
rec["npy_count"] = len(npys)
if md5 != EXPECT_MD5:
    refuse(f"zip md5 mismatch: {md5} != {EXPECT_MD5}")
if len(npys) != EXPECT_NPY:
    refuse(f".npy count {len(npys)} != {EXPECT_NPY}")

# ---- 1. schema / row counts (leak check 3) ---------------------------------
D = {}
print("\n" + "=" * 78)
print("1. SCHEMA AND ROW COUNTS  (PLAN leak check 3: mismatch aborts)")
for p in PANELS:
    D[p] = {}
    for a in ARRAYS:
        fp = os.path.join(REL, f"{p}_{a}.npy")
        if not os.path.exists(fp):
            refuse(f"missing released array {p}_{a}.npy")
            continue
        arr = np.load(fp)
        D[p][a] = arr
        exp = EXPECT_SHAPE[p][a]
        ok = (arr.shape == exp)
        print(f"   {p}_{a:<13} shape={str(arr.shape):<10} dtype={str(arr.dtype):<8} "
              f"expected={str(exp):<10} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            refuse(f"{p}_{a} shape {arr.shape} != expected {exp}")
rec["schema_ok"] = not FAIL

# ---- 2. leak check 1: target not in any arm feature -------------------------
print("\n" + "=" * 78)
print("2. LEAK CHECK 1  -- target may not appear in any arm feature set")
FEATURE_POOL = ["theta", "qubits", "z_noisy"]   # the only arrays any arm may read
lk1 = {}
for p in PANELS:
    y = D[p]["mz_exact"]
    worst = {}
    for a in FEATURE_POOL:
        X = np.asarray(D[p][a], float)
        # (i) exact column match
        colmin = np.min(np.max(np.abs(X - y[:, None]), axis=0))
        # (ii) affine-transform match of any single column: r == +-1 exactly
        rr = []
        for j in range(X.shape[1]):
            c = X[:, j]
            if np.std(c) == 0:
                rr.append(0.0)
            else:
                rr.append(abs(np.corrcoef(c, y)[0, 1]))
        worst[a] = {"min_max_abs_col_diff": float(colmin), "max_abs_corr_any_col": float(max(rr))}
    # (iii) the row mean of z_noisy, the one natural aggregate
    mn = np.asarray(D[p]["z_noisy"], float).mean(axis=1)
    worst["mean_z_noisy"] = {"max_abs_diff": float(np.max(np.abs(mn - y))),
                             "corr": float(np.corrcoef(mn, y)[0, 1])}
    lk1[p] = worst
    print(f"   {p}:")
    for a in FEATURE_POOL:
        w = worst[a]
        print(f"     {a:<9} closest column max|diff| = {w['min_max_abs_col_diff']:.6f}   "
              f"max |corr| over columns = {w['max_abs_corr_any_col']:.6f}")
    print(f"     mean(z_noisy)  max|diff| = {worst['mean_z_noisy']['max_abs_diff']:.6f}   "
          f"corr = {worst['mean_z_noisy']['corr']:.6f}")
    for a in FEATURE_POOL:
        if worst[a]["min_max_abs_col_diff"] < 1e-9:
            refuse(f"{p}: target equals a column of {a}")
        if worst[a]["max_abs_corr_any_col"] > 1.0 - 1e-12:
            refuse(f"{p}: target is an exact affine transform of a column of {a}")
    if worst["mean_z_noisy"]["max_abs_diff"] < 1e-9:
        refuse(f"{p}: target equals mean(z_noisy)")
rec["leak1"] = lk1
print("   NOTE: mz_zne, mz_linear and mz_predicted are other methods' outputs and are")
print("         excluded from every arm's feature set by construction.")

# ---- 3. leak check 2: no oracle quantity ------------------------------------
print("\n" + "=" * 78)
print("3. LEAK CHECK 2  -- no arm may read an oracle quantity")
print("   Arm feature pool is exactly {theta, qubits, z_noisy}.")
print("   theta   : RX angles drawn before execution        -> not an oracle")
print("   qubits  : physical qubit indices                  -> not an oracle")
print("   z_noisy : FakeGuadalupe shot-averaged <Z_n>       -> execution output, not an oracle")
print("   Excluded from every arm: mz_exact (target), mz_zne, mz_linear, mz_predicted.")
# no ideal-quantity case filter can exist: every panel is a full 100 rows
for p in PANELS:
    n = D[p]["mz_exact"].shape[0]
    print(f"   {p}: {n} rows present, no case filtered (expected 100).")
    if n != 100:
        refuse(f"{p}: {n} rows, a case filter may be present")
# z_noisy is on the 10^4-shot grid -> it is a finite-shot execution output
for p in PANELS:
    z = D[p]["z_noisy"]
    g = z * 5000.0
    off = float(np.max(np.abs(g - np.round(g))))
    print(f"   {p}: z_noisy shot-grid residual (units of 2/1e4) = {off:.3e}")
    if off > 1e-6:
        print(f"   WARNING {p}: z_noisy not on the declared 1e4-shot grid")

# ---- 4. leak check 4: row-shuffled split hazard -----------------------------
print("\n" + "=" * 78)
print("4. LEAK CHECK 4  -- rows of one circuit on both sides of a split")
lk4 = {}
for p in PANELS:
    th = D[p]["theta"]
    uniq = len(set(map(tuple, np.round(th, 12))))
    ymin = float(np.min(np.abs(np.subtract.outer(D[p]["mz_exact"], D[p]["mz_exact"]) +
                               np.eye(100) * 9)))
    lk4[p] = {"unique_theta_rows": uniq, "n": int(th.shape[0]), "min_target_gap": ymin}
    print(f"   {p}: unique theta rows {uniq}/{th.shape[0]}; "
          f"min |target_i - target_j| over i!=j = {ymin:.3e}")
    if uniq != th.shape[0]:
        refuse(f"{p}: repeated circuit rows -> a row-shuffled split can straddle a circuit")
# cross-panel duplicate check
for i in range(3):
    for j in range(i + 1, 3):
        pi, pj = PANELS[i], PANELS[j]
        if D[pi]["theta"].shape[1] != D[pj]["theta"].shape[1]:
            shared = 0
        else:
            si = set(map(tuple, np.round(D[pi]["theta"], 12)))
            sj = set(map(tuple, np.round(D[pj]["theta"], 12)))
            shared = len(si & sj)
        print(f"   shared theta rows {pi} vs {pj}: {shared}")
rec["leak4"] = lk4
print("   -> one row is one circuit realization; panels are scored separately;")
print("      the hazard does not arise. RECORDED, not triggered.")

# ---- 5. qubits constant -----------------------------------------------------
print("\n" + "=" * 78)
print("5. DESCRIPTOR STRUCTURE: is `qubits` informative on the released rows?")
qrec = {}
for p in PANELS:
    q = D[p]["qubits"]
    stds = q.std(axis=0)
    same_all_rows = bool(np.all(q == q[0]))
    qrec[p] = {"per_column_std_max": float(stds.max()),
               "constant_across_rows": same_all_rows,
               "value": q[0].tolist(),
               "sha256": hashlib.sha256(q.tobytes()).hexdigest()}
    print(f"   {p}: constant across rows = {same_all_rows}; max per-column std = {stds.max():.3e}")
    print(f"        value = {q[0].tolist()}")
    print(f"        sha256 = {qrec[p]['sha256'][:16]}...")
ident = len({qrec[p]["sha256"] for p in PANELS}) == 1
print(f"   byte-identical across all three panels: {ident}")
rec["qubits"] = qrec
print("   -> zero variance: `qubits` carries no information for any arm fitted on these rows.")
print("      An affine arm must drop it for rank reasons (mechanical, not feature selection).")
print("      It is still supplied to the CNN arms as a channel, exactly as the release does.")

# ---- 6. mz_linear identity --------------------------------------------------
print("\n" + "=" * 78)
print("6. mz_linear IS NOT ARM A  -- it reads an execution-derived input")
lin = {}
for p in PANELS:
    mn = np.asarray(D[p]["z_noisy"], float).mean(axis=1)
    A = np.column_stack([np.ones_like(mn), mn])
    coef, *_ = np.linalg.lstsq(A, D[p]["mz_linear"], rcond=None)
    res = float(np.max(np.abs(A @ coef - D[p]["mz_linear"])))
    lin[p] = {"c2_intercept": float(coef[0]), "c1_slope": float(coef[1]), "max_abs_residual": res}
    print(f"   {p}: mz_linear = {coef[0]:+.10f} + {coef[1]:.10f} * mean(z_noisy);  "
          f"max|resid| = {res:.2e}")
rec["mz_linear_identity"] = lin

# ---- 7. released R2 reproduction -------------------------------------------
print("\n" + "=" * 78)
print("7. IDENTITY OF THE RELEASED PREDICTIONS (arm F)")
r2rec = {}
for k, p in enumerate(PANELS):
    y = D[p]["mz_exact"]
    row = {
        "cnn": r2(y, D[p]["mz_predicted"]),
        "linear": r2(y, D[p]["mz_linear"]),
        "zne": r2(y, D[p]["mz_zne"]),
        "noisy_mean": r2(y, np.asarray(D[p]["z_noisy"], float).mean(axis=1)),
    }
    r2rec[p] = {k2: float(v) for k2, v in row.items()}
    print(f"   {p}: R2  CNN={row['cnn']:.6f} (paper {PAPER_R2_CNN[k]:.2f})   "
          f"linear={row['linear']:.6f} (paper {PAPER_R2_LIN[k]:.2f})   "
          f"ZNE={row['zne']:+.4f}   noisy-mean={row['noisy_mean']:+.4f}")
    if abs(row["cnn"] - PAPER_R2_CNN[k]) > 0.005:
        refuse(f"{p}: CNN R2 {row['cnn']:.4f} does not match the paper's {PAPER_R2_CNN[k]}")
    if abs(row["linear"] - PAPER_R2_LIN[k]) > 0.005:
        refuse(f"{p}: linear R2 {row['linear']:.4f} does not match the paper's {PAPER_R2_LIN[k]}")
rec["released_r2"] = r2rec
print(f"   mz_predicted dtype = {D['Fig8a']['mz_predicted'].dtype} (float32, a network output)")

# ---- 8. determinism: mz_exact == f(theta) ----------------------------------
# Exact state-vector simulation of the released generator, from circ_gen.py.
# Association list, RZZ(-pi/2) on the Guadalupe spanning tree, |0>^16 input.
print("\n" + "=" * 78)
print("8. IS THE TARGET AN EXACT FUNCTION OF THE DESCRIPTORS? (saturation bound)")
ASSOC = [0, 1, 2, 3, 5, 8, 9, 11, 14, 13, 12, 15, 10, 7, 6, 4]
N, P = 16, 20
edges = []
for i in range(N - 1):
    qi = ASSOC[i]
    if qi in (8, 12, 7):
        edges.append((i, i + 1))
        if i + 2 < N:
            edges.append((i, i + 2))
    elif qi not in (9, 15, 6):
        edges.append((i, i + 1))
print(f"   RZZ edges from circ_gen.py on the full 16-qubit association list: {edges}")

dim = 1 << N
bits = ((np.arange(dim)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
sgn = 1 - 2 * bits                              # (dim, N), +1 for |0>, -1 for |1>
zz_phase = np.zeros(dim)
for (a, b) in edges:
    zz_phase += sgn[:, a] * sgn[:, b]           # sum of Z_a Z_b eigenvalues
RZZ = np.exp(-1j * (-np.pi / 2) / 2.0 * zz_phase)   # RZZ(t) = exp(-i t/2 Z Z)


def rx_all(psi, angles):
    """Apply RX(angles[k]) to qubit k for every k, in place-ish."""
    for k in range(N):
        t = angles[k]
        c, s = np.cos(t / 2.0), -1j * np.sin(t / 2.0)
        v = psi.reshape(-1, 2, 1 << k)          # axis 1 is qubit k
        a0 = v[:, 0, :].copy()
        a1 = v[:, 1, :].copy()
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = s * a0 + c * a1
        psi = v.reshape(-1)
    return psi


def mz_from_theta(theta_row, config):
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    for j in range(P):
        ang = theta_row if config == "A" else np.full(N, theta_row[j])
        psi = rx_all(psi, ang)
        psi *= RZZ
    prob = np.abs(psi) ** 2
    return float(np.sum(prob[:, None] * sgn, axis=0).mean())


det = {}
for p, cfg in [("Fig8a", "A"), ("Fig8b", "B"), ("Fig8c", "B")]:
    th = D[p]["theta"]
    pred = np.array([mz_from_theta(th[i], cfg) for i in range(th.shape[0])])
    err = np.abs(pred - D[p]["mz_exact"])
    det[p] = {"config": cfg, "max_abs_err": float(err.max()), "rmse": float(np.sqrt((err ** 2).mean())),
              "r2_of_exact_recomputation": float(r2(D[p]["mz_exact"], pred))}
    print(f"   {p} (config {cfg}): recomputed mz from theta alone -> "
          f"max|err| = {err.max():.3e}, R2 = {det[p]['r2_of_exact_recomputation']:.12f}")
rec["determinism"] = det
maxerr = max(v["max_abs_err"] for v in det.values())
print(f"   worst-case error over all 300 rows: {maxerr:.3e}")
if maxerr < 1e-9:
    print("   -> CONFIRMED: mz_exact is an exact deterministic function of theta alone.")
    print("      Information-theoretic descriptor saturation is 1.0 by construction.")
    print("      Any measured saturation below that is a finite-sample statement about")
    print("      learning a 16- or 20-dimensional function from 100 rows, NOT the plan's")
    print("      'the descriptors never carried the information' branch.")
else:
    print("   -> recomputation does NOT reproduce the target; the determinism claim is")
    print("      NOT independently confirmed and must not be relied on.")

# ---- verdict ----------------------------------------------------------------
print("\n" + "=" * 78)
if FAIL:
    print("VERDICT: REFUSAL. Phase 2 aborts. Reasons:")
    for m in FAIL:
        print("  - " + m)
else:
    print("VERDICT: all four leak checks PASS. No refusal. Fitting may proceed.")
rec["refusal_triggered"] = bool(FAIL)
rec["refusals"] = FAIL
with open(os.path.join(OUT, "s01_verify.json"), "w") as f:
    json.dump(rec, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's01_verify.json')}")
sys.exit(1 if FAIL else 0)
