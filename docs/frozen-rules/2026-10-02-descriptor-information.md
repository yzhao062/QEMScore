# Frozen Rule: Descriptor-Information Experiment (QEMScore, Post-Campaign)

Written 2026-10-02, before any fit on a new rung or a new dataset. It was reviewed by three independent plan reviews (one Gemini, two Claude) before freezing. The R0 fits it reuses were analyzed on 2026-10-01 (Appendix M.4), so their intervals were known when this rule was written.

## Question

Between the two constructed extremes of the paper, does the capacity-matched comparison respond to how much the descriptors identify the label? Two cases are tested:
- **Missing information:** the descriptors are degraded on the same circuits (Part A).
- **Complete but high-dimensional:** descriptors on a family where the label depends on the graph (Part B).

## Part A: Spin Chains, Same Circuits

**Data.** The three regenerated primary datasets (shipped steps, 640 training circuits per family, dataset seeds 101, 211, 307). Circuits, noisy estimates, labels, and splits are fixed. Only the instance-descriptor columns change between rungs.

| Rung | Instance descriptors | Notes |
|---|---|---|
| R0 | Exact j, h (TFI); jx, jy, jz (Heisenberg); dt; steps | Reuses the 20-seed fits of Appendix M.4 |
| N1, N2, N3, N4 | Each coupling plus z·s, with s = 0.01, 0.03, 0.10, 0.30 | Coupling range is 1.0 (uniform on [0.2, 1.2]); no clipping, because these are inputs, not physics |
| R3-TFI | h withheld for TFI items only | The Heisenberg items keep R0 descriptors; read only the TFI row |
| R3-Heis | jz withheld for Heisenberg items only | Read only the Heisenberg row |
| R4 | Couplings, dt, steps replaced by the released ML-QEM encoding | `count_gates_by_rotation_angle`, qiskit-community/ml-qem commit b1eccf8, mlp.py SHA-256 6d62128392…; 5 native-gate counts plus 160 angle bins of width 0.025π, scaled by 0.01 as released; applied to each item's transpiled circuit |
| R5 | None | Keeps only the columns below |

For the N rungs, z is one standard-normal draw per circuit and coupling. It comes from `numpy.random.default_rng(SeedSequence([20261002, dataset_seed, circuit_index]))`, where circuit_index is the circuit's position in sorted circuit_id order. The same z is used at every N rung and in the training, validation, and test splits.

**Columns kept at every rung, and by every arm.** These are the noisy estimate (where the arm reads it), log2 shots, qubit count, family indicators, two-qubit gate count, compiled depth, and observable locality. In Part A the last three are constant within a family.

## Part B: QAOA-MaxCut, Random Graphs

**Data.** New datasets from the released generator and validator, built the same way as the near-Clifford control:
- setting S0 with 10 qubits and p in {1, 2};
- graph classes Erdős–Rényi (edge probability 0.3) and 3-regular, which are the only classes whose edge set varies at 10 qubits;
- campaign noise (depolarizing plus readout at L1 and L3) and 2,048 shots;
- 640 training, 320 source-validation, and 160 test circuits;
- dataset seeds 101, 211, 307.

**Observable.** zz_mid only. ⟨Z_i⟩ is identically zero for MaxCut QAOA under the global X-flip symmetry, so z_mid is excluded. The macro error averages the L1 and L3 cells. A cell is dropped before any fit if its training-label SD is below 0.01, and the drop is disclosed.

| Rung | Instance descriptors |
|---|---|
| B-partial | The schema's QAOA columns: p, edge count, two γ, two β, graph-class indicators |
| B-complete | B-partial plus 45 edge-indicator columns, one per qubit pair, from the stored edge set |
| B-none | None (the R5 analogue) |

**Reference.** A gradient-boosted tree model on B-complete descriptors only, tuned on source validation, reported as a reference. It is not an arm.

## Arms

- **R:** the raw noisy estimate.
- **A:** an affine control on the rung's descriptors, fitted once per dataset and rung.
- **C:** `LiaoMitigator` without the noisy estimate.
- **F:** `LiaoMitigator` with the noisy estimate.
- **P:** F refitted on a permuted noisy-estimate column.

C, F, and P each run at learner seeds 1 to 20 through the campaign's own fitting path, extended with a feature-builder hook. Before any new fit, the R0 path must reproduce the archived R0 test predictions to 1e-11 through that hook. If it does not, no new fit runs.

**Measurement-only arm M.** M is F at R5 (Part A) and F at B-none (Part B). It is also fitted at learner seeds 1 to 20 on the three near-Clifford datasets of Appendix N. C at R5 and at B-none is fitted the same way.

Every fit saves each candidate's per-item test predictions (random forest and MLP), the selected candidate, and the selection scores.

## Estimands

Each row (dataset seed by family in Part A, dataset seed in Part B) and rung gets the following:
- **D = C − F**, macro mean absolute error averaged over learner seeds.
- **D/C**, computed as a ratio of means inside each bootstrap draw.
- **E = M − F**, the error the rung's descriptors remove on top of the measurement.
- **S = (A − C)/(A − F)**, reported only when A − F is positive in every draw.
- **Paired contrasts against R0** (Part A) and against B-partial (Part B): ΔD and Δ(D/C), using the same seed and circuit indices in each draw.
- **A learner-fixed secondary D**, with the MLP candidate in both arms, plus selection counts per cell.
- **D per severity cell**, descriptive only.

## Uncertainty

A two-stage percentile bootstrap: resample the 20 learner seeds, then the test circuits as whole units. It uses 10,000 draws and a fresh `numpy.random.default_rng(20261002)` per row and quantity. Each draw calls `integers(0, n_seeds, n_seeds)`, then `integers(0, n_circuits, n_circuits)`. Intervals are 95 percent, 2.5 to 97.5 percentiles, and pointwise.

## Classification (Mutually Exclusive)

Each cell (row by rung) gets one label:
1. **Measurement adds:** the 95 percent interval for mean D lies wholly above zero.
2. **Measurement hurts:** the interval lies wholly below zero.
3. **Not distinguished:** otherwise. The largest remaining-error reduction not excluded is reported as the upper limit of mean D divided by mean C.

A rung-level statement for a family (Part A) or for Part B is made only when all three dataset seeds share one label; otherwise the rung reads "mixed". No equivalence claim is made. Counts of cells are not promoted to claims.

## Expectations Stated in Advance (Not Decision Rules)

- R0 is not "measurement adds", which is already known.
- D/C rises with the noise level s, and from R0 to R3.
- R4 sits near N2 to N3, because the encoding collapses about 640 training circuits to about 35 distinct vectors per family and cannot tell jx, jy, and jz apart.
- B-complete has a lower D than B-partial.

## Pre-Fit, Label-Free Diagnostics (Reported Beside Results)

- For R4: the number of distinct descriptor vectors among the training circuits, and the k-nearest-neighbour R² for recovering each coupling from the R4 vector.
- For Part B: the training-label SD per cell and per graph class, and the attenuation slope of the noisy estimate.

## Deviations

- The convergence rule and the warning sensitivity follow Appendix M.4: only the platform's spurious matrix-multiplication warnings are excluded in the sensitivity analysis.
- If QAOA validation fails at 10 qubits, Part B uses the largest even qubit count up to 10 that validates. That count is chosen before any fit and disclosed.
- Any analysis added after results are seen is labeled exploratory.
- Two independent analysis scripts must agree: point estimates to 1e-9, and intervals exactly (they share the RNG scheme above).

### Amendments

Both amendments were approved by the author on 2026-10-02 and committed with this file before any fit on a new rung or dataset.

- Amendment to the R0 gate, made 2026-10-02 after the gate run and before any fit on a new rung or dataset. The frozen tolerance of 1e-11 was tighter than the reproduction bound that Appendix M.4 already reports (1.4e-11 or better). Through the hook, the original-seed refits reproduce the archived predictions to 1.37e-11 (largest: C at dataset seed 307), while F stays within 1.5e-13 and A within 1e-13. On the same machine, the hooked refits equal the pre-hook M.4 refits exactly (maximum difference 0). The gate therefore uses a tolerance of 1.4e-11, the bound M.4 states.
- Amendment to the agreement criterion for the two analysis scripts, made 2026-10-02 before any fit on a new rung or dataset. Run on the Appendix M.4 fits presented as rung R0, the two independent scripts agree on every quantity to within 1e-15 and give identical labels, but their interval bounds are not bit-identical, because they sum in a different order. The criterion is therefore agreement of point estimates and interval bounds to 1e-12, and identical labels and rung statements.

## Release

The release contains:
- this file;
- the scripts;
- per-fit, per-candidate test predictions and the cached test items for Parts A and B;
- the same inputs for the existing learner-seed (Appendix M.4) and near-Clifford (Appendix N) fits.
