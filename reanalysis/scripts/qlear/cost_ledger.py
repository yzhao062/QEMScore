"""The circuit-evaluation ledger, derived from the released generators.

Hardware (Generate Test Data Hardware.ipynb): one batch per application
circuit, reused for every output state. circuit_list holds two copies of the
transpiled circuit, one full inverse and three depth-cut circuits; every entry
runs at shots=1024.

Simulator (Generate Test Data Simulator.ipynb, Generate Training Data
Simulator.ipynb): feature collection sits inside the per-state loop, and each
call is execute_with_repitions(..., repitions=2) at shots=1024. Five such calls
per state: base, full inverse, and three depth-cut circuits.

Every count here is implied by the released source. None of it is measured
expenditure, wall-clock time, billing or retries.

The 174 release CSV and QASM files used for accounting are declared by name
and SHA-256 in release-circuit-manifest.txt beside this script. This manifest
was generated from the local release during the R2 correction. Its 112 CSV
entries match PRE-field-checksums.txt; its 62 QASM entries gain a checksum
baseline only with this manifest. Comparing against the retained manifest
checks current names and bytes against that record. It does not establish
publication correctness, pre-fit history, or measured execution. The separately
supplied field report is checked for expected seeds but is not authenticated by
this release manifest.

Usage: cost_ledger.py <unpacked-release-dir> <field-report.json> <output-dir>
"""
import glob, hashlib, json, os, sys
import numpy as np

if len(sys.argv) != 4:
    raise SystemExit(__doc__.strip().splitlines()[-1])
REL, REPORT, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
SHOTS = 1024
# Simulator generation, from the released notebooks: five feature-collection
# calls per output state at two repetitions each, plus one noisy enumeration
# that fixes the released support and one ideal simulation that supplies the
# label. The ideal run is supervision, so it is reported apart from the
# generation shots and never enters an incremental inference efficiency.
SIM_SHOTS_PER_STATE = 5 * 2 * SHOTS
SIM_NOISY_ENUMERATION_PER_CIRCUIT = SHOTS
IDEAL_SUPERVISION_PER_CIRCUIT = SHOTS
# Hardware generation: one batch of six circuits per application circuit,
# reused for every output state, so the count does not vary with state count.
HW_GENERATION_PER_CIRCUIT = 6 * SHOTS
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "release-circuit-manifest.txt")
# Declared here rather than read off the manifest. A manifest that simply omits
# a directory would otherwise switch off the check for it while the ledger went
# on reading it and counting its rows.
COVERED_DIRECTORIES = ("real_circuits_hardware", "real_circuits",
                       "training_data", "testing_data",
                       "training_circuits", "testing_circuits")


def read_manifest():
    """Read release-file name and SHA-256 declarations."""
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"manifest missing: {MANIFEST}")
    declared = {}
    for line in open(MANIFEST, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            raise SystemExit(f"manifest: cannot parse line {line!r}")
        rel = rel.replace("\\", "/")
        # A repeated path is a contradiction, not a refinement. Taking the last
        # one would let a second line silently overwrite the first.
        if rel in declared:
            raise SystemExit(f"manifest: {rel} is declared more than once")
        declared[rel] = digest
    covered = {r.split("/")[0] for r in declared}
    uncovered = sorted(set(COVERED_DIRECTORIES) - covered)
    if uncovered:
        raise SystemExit(f"manifest: declares nothing for {uncovered}, "
                         f"which this ledger reads")
    stray = sorted(covered - set(COVERED_DIRECTORIES))
    if stray:
        raise SystemExit(f"manifest: declares directories this ledger does not "
                         f"read {stray}")
    return declared


def verify_against_manifest(declared):
    """Refuse a substituted, missing or added input before counting anything.

    Names are checked per directory, so a rename that preserves the file count
    is named as the substitution it is. Hashes then detect changed bytes,
    including through a link, since the file is read rather than stat-ed. A
    byte-identical copy or link under a declared name is accepted on purpose:
    the check is on names and bytes, not on inode identity.
    """
    by_dir = {}
    for rel in declared:
        by_dir.setdefault(rel.split("/")[0], set()).add(rel.split("/", 1)[1])
    for folder in sorted(by_dir):
        want = by_dir[folder]
        path = os.path.join(REL, folder)
        if not os.path.isdir(path):
            raise SystemExit(f"{folder}: declared directory is absent")
        found = {e for e in os.listdir(path)
                 if os.path.isfile(os.path.join(path, e))}
        missing, extra = sorted(want - found), sorted(found - want)
        if missing:
            raise SystemExit(f"{folder}: missing declared files {missing}")
        if extra:
            raise SystemExit(f"{folder}: undeclared files present {extra}")
    changed = []
    for rel, digest in sorted(declared.items()):
        h = hashlib.sha256()
        with open(os.path.join(REL, *rel.split("/")), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            changed.append(rel)
    if changed:
        raise SystemExit(f"content differs from the manifest: {changed}")
    return by_dir

# The cohort is declared rather than derived, so a missing application or
# backend stops the ledger instead of quietly shrinking a mean.
EXPECTED_APPLICATIONS = ("groundstate", "pricingcall", "pricingput",
                         "qaoa", "routing", "tsp")
EXPECTED_BACKENDS = ("ibm_lagos", "ibm_nairobi", "ibm_perth", "ibmq_belem",
                     "ibmq_jakarta", "ibmq_lima", "ibmq_manila", "ibmq_quito")
EXPECTED_SEEDS = tuple(range(10))
# The corpus directories are declared the same way the evaluation cohort is.
# Counting entries is not enough: a renamed file keeps the count while dropping
# a backend, and a stray non-QASM entry adds a circuit that was never run.
TRAINING_DATA_FILES = tuple(f"training_{b}.csv" for b in EXPECTED_BACKENDS)
TESTING_DATA_FILES = tuple(f"testing_{b}.csv" for b in EXPECTED_BACKENDS)

def states_per_application(folder):
    """Mean output states per circuit, per application.

    The roster is checked by identity rather than by count. A file named for an
    unexpected application or backend, or a missing pair, stops the ledger; a
    count-only check would accept a duplicate standing in for an absent file.
    """
    seen = {}
    for f in sorted(glob.glob(os.path.join(REL, folder, "*.csv"))):
        name = os.path.basename(f)[:-4]
        app, _, backend = name.partition("_")
        n = sum(1 for _ in open(f)) - 1
        if (app, backend) in seen:
            raise SystemExit(f"{folder}: duplicate file for {app}/{backend}")
        seen[(app, backend)] = n
    expected = {(a, b) for a in EXPECTED_APPLICATIONS for b in EXPECTED_BACKENDS}
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing:
        raise SystemExit(f"{folder}: missing application/backend pairs {missing}")
    if extra:
        raise SystemExit(f"{folder}: unexpected application/backend pairs {extra}")
    return {a: float(np.mean([seen[(a, b)] for b in EXPECTED_BACKENDS]))
            for a in EXPECTED_APPLICATIONS}

def main():
    declared_by_dir = verify_against_manifest(read_manifest())
    out = {}
    for folder, panel in (("real_circuits_hardware", "hardware"), ("real_circuits", "simulator")):
        ns = states_per_application(folder)
        apps = sorted(ns)
        mean_states = float(np.mean([ns[a] for a in apps]))
        if panel == "hardware":
            per_app_C_O = {a: SHOTS * 1 for a in apps}   # second base batch
            per_app_O_F = {a: SHOTS * 3 for a in apps}   # three depth-cut batches
            released = {
                "units": "hardware shots for generation, "
                         "ideal-simulator shots for supervision",
                "formula": "6 circuits x 1024 shots, batched once per application "
                           "circuit and reused for every output state",
                "per_circuit_count_varies_by_application": False,
                "generation_shots_per_circuit": HW_GENERATION_PER_CIRCUIT,
                "ideal_supervision_shots_per_circuit": IDEAL_SUPERVISION_PER_CIRCUIT}
        else:
            per_app_C_O = {a: 2 * SHOTS * ns[a] for a in apps}   # base, 2 reps, per state
            per_app_O_F = {a: 6 * SHOTS * ns[a] for a in apps}   # 3 circuits x 2 reps, per state
            per_app_gen = {a: SIM_SHOTS_PER_STATE * ns[a]
                              + SIM_NOISY_ENUMERATION_PER_CIRCUIT for a in apps}
            gen_mean = float(np.mean([per_app_gen[a] for a in apps]))
            # Both exported counts are means, so neither is the batch of an
            # actual circuit. The per-application figure averages the eight
            # backends, since a circuit's state count differs between them; the
            # scalar then averages the six applications with equal weight,
            # matching the error weighting. An individual circuit's count is
            # always 1024 + 10 * 1024 * s for an integer state count s.
            released = {
                "units": "noisy-simulator shots for generation, "
                         "ideal-simulator shots for supervision",
                "formula": "s x (5 calls x 2 repetitions x 1024 shots) + 1024 noisy "
                           "enumeration, for a circuit with s output states",
                "per_circuit_count_varies_by_application": True,
                "generation_shots_mean_per_circuit_by_application": per_app_gen,
                "generation_shots_mean_per_circuit": gen_mean,
                "ideal_supervision_shots_per_circuit": IDEAL_SUPERVISION_PER_CIRCUIT,
                "generation_plus_supervision_mean_per_circuit":
                    gen_mean + IDEAL_SUPERVISION_PER_CIRCUIT}
        # Equal weight over applications, matching the macro error weighting.
        macro_C_O = float(np.mean([per_app_C_O[a] for a in apps]))
        macro_O_F = float(np.mean([per_app_O_F[a] for a in apps]))
        out[panel] = {"states_per_application": ns, "mean_states": mean_states,
                      "macro_added_evaluations": {"C-O": macro_C_O, "O-F": macro_O_F},
                      "per_application_added_evaluations": {"C-O": per_app_C_O, "O-F": per_app_O_F},
                      "released_path": released}
    rep = json.load(open(REPORT))
    print(f"{'panel':<11}{'step':<6}{'macro added evals':>19}{'gap (10-fit mean)':>19}{'per-eval':>13}{'ratio':>9}")
    print("-"*80)
    for panel in ("hardware", "simulator"):
        b = rep["panels"][panel]
        absent = [s for s in EXPECTED_SEEDS if str(s) not in b["per_seed"]]
        if absent:
            raise SystemExit(f"{panel}: report is missing seeds {absent}")
        gaps = {k: float(np.mean([b["per_seed"][str(s)]["gaps"][k]["estimate"]
                                  for s in EXPECTED_SEEDS]))
                for k in ("C-O", "O-F")}
        effs = {}
        for k in ("C-O", "O-F"):
            n = out[panel]["macro_added_evaluations"][k]
            effs[k] = gaps[k] / n
            print(f"{panel:<11}{k:<6}{n:19,.0f}{gaps[k]:19.4f}{effs[k]:13.3e}", end="")
            print(f"{'':>9}" if k == "C-O" else f"{effs['C-O']/effs['O-F']:9.2f}")
        out[panel]["gaps"] = gaps
        out[panel]["per_evaluation"] = effs
        out[panel]["efficiency_ratio_C_O_over_O_F"] = effs["C-O"] / effs["O-F"]
    # Generation totals for the released corpora.
    train_rows = sum(sum(1 for _ in open(f)) - 1 for f in glob.glob(os.path.join(REL, "training_data", "*.csv")))
    val_rows = sum(sum(1 for _ in open(f)) - 1 for f in glob.glob(os.path.join(REL, "testing_data", "*.csv")))
    # Circuits are counted from the declared QASM roster, so a non-QASM entry
    # cannot add one and a missing circuit cannot be masked by another entry.
    n_train_circ = sum(1 for f in declared_by_dir["training_circuits"]
                       if f.endswith(".qasm"))
    n_test_circ = sum(1 for f in declared_by_dir["testing_circuits"]
                      if f.endswith(".qasm"))
    n_backends = len(EXPECTED_BACKENDS)
    for folder, want in (("training_data", TRAINING_DATA_FILES),
                         ("testing_data", TESTING_DATA_FILES)):
        found = {os.path.basename(f)
                 for f in glob.glob(os.path.join(REL, folder, "*.csv"))}
        missing, extra = sorted(set(want) - found), sorted(found - set(want))
        if missing:
            raise SystemExit(f"{folder}: missing declared backends {missing}")
        if extra:
            raise SystemExit(f"{folder}: unexpected files {extra}")
    if n_train_circ != 56 or n_test_circ != 6:
        raise SystemExit(f"declared circuits: expected 56 training and 6 testing, "
                         f"found {n_train_circ} and {n_test_circ}")
    per_state = SIM_SHOTS_PER_STATE
    per_circuit_fixed = (SIM_NOISY_ENUMERATION_PER_CIRCUIT
                         + IDEAL_SUPERVISION_PER_CIRCUIT)
    train_total = train_rows * per_state + n_train_circ * n_backends * per_circuit_fixed
    val_total = val_rows * per_state + n_test_circ * n_backends * per_circuit_fixed
    supervision = ((n_train_circ + n_test_circ) * n_backends
                   * IDEAL_SUPERVISION_PER_CIRCUIT)
    out["corpus_generation"] = {
        "ideal_supervision_shots_included_in_totals": supervision,
        "training_rows": train_rows, "validation_rows": val_rows,
        "training_circuits": n_train_circ, "testing_circuits": n_test_circ,
        "backends": n_backends, "simulator_shots_per_state": per_state,
        "simulator_shots_per_circuit_fixed": per_circuit_fixed,
        "training_simulator_shots": train_total, "validation_simulator_shots": val_total}
    print()
    for panel in ("hardware", "simulator"):
        r = out[panel]["released_path"]
        gen = r.get("generation_shots_per_circuit",
                    r.get("generation_shots_mean_per_circuit"))
        mean = "mean " if r["per_circuit_count_varies_by_application"] else ""
        print(f"{panel} released path: {gen:,.4f} {mean}generation shots per application "
              f"circuit, plus {r['ideal_supervision_shots_per_circuit']:,} ideal-simulator "
              f"supervision shots ({r['units']})")
    print()
    print(f"training corpus: {train_rows:,} rows over {n_train_circ} circuits x {n_backends} backends "
          f"-> {train_total:,} simulator shots")
    print(f"validation corpus: {val_rows:,} rows over {n_test_circ} circuits x {n_backends} backends "
          f"-> {val_total:,} simulator shots")
    print(f"combined: {train_total + val_total:,} source-implied simulator shots, "
          f"including {supervision:,} ideal-simulator supervision shots")
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, "cost-ledger.json"), "w"), indent=1)
    print("\nwrote cost-ledger.json")

main()
