# Frozen Rule: Zero-Shot Transfer Under a Depth Shift (S4) and a Noise-Strength Shift (S2)

Written 2026-10-06 (US Pacific time), before any computation this rule governs. The ninth internal review panel asked for one shift setting at rungs R0, N3, R4, and R5. Every fit and analysis below starts only after this rule and the code it names are pushed to the public repository. Data generation reads no outcome, so the six artifacts may be generated while this rule is reviewed.

## Known When This Rule Was Written

- **S0 ladder results.** `artifacts/descriptor-information/analysis-a.json` is public. At R0, all six rows (three dataset seeds, two families) are "not distinguished", with D/C from −0.306 to +0.166. At N3, R4, and R5, all six rows are "F beats C", with D/C from +0.250 to +0.584. The raw error R is 0.251 to 0.281 for transverse-field Ising and 0.596 to 0.609 for Heisenberg.
- **Prototype generation.** To time the generator and test source identity, the seed-101 artifact of each shift was generated with the specification below. Each took 12 to 13 minutes, two running at a time. In both, all 5,120 training and 2,560 validation rows equal the S0 dataset's rows in every identity field, and the target rows pass the structure check. The dataset hashes were `1c0c027c753c36e04a10b9f3924fa24ddece42618f9cbe220c8fbc5a05442fdc` (S4) and `1e605e7461c1f5f70d06cdf8e6d0ecd0fea75208965ec4b767d0294899874520` (S2). Only row keys, identity fields, and target structure were read. No fit ran on any shifted artifact, and no target label or target noisy estimate was examined.
- **Smoke test without shifted data.** The tool's fit path was run with S0's own test rows in place of target rows. It reproduced 16 archived S0 ladder fits bit for bit, on validation and test predictions. These were seed 101, rungs R0, N3, R4, and R5, arms F and C, and learner seeds 1 and 2.
- **Fit cost.** Archived ladder fits take 8 to 12 seconds each, so the 960 fits below take about 2.5 CPU hours.

## Design

**Shifts.** Both shifts start from the campaign's S0 specification, `campaign_split_spec("shipped", 640)`, and move one axis from the fixed axes to the source and target domains.

- S4 (deeper circuits): `family_native_depth` has source [3] and target [6]. Test circuits run twice as many Trotter steps as any training circuit.
- S2 (stronger noise): `noise_strength` has source [L1, L3] and target [L4]. L4 is the strongest level of the frozen depolarizing grid (p1 = 0.02, p2 = 0.10, readout 0.10, against 0.01, 0.06, and 0.06 at L3).

Every other axis keeps its S0 value: ten qubits, both spin-chain families at the shipped time steps, Z and ZZ observables, 2,048 shots, and 160 test circuits per family. The two target values are fixed here, and no other value is run. The generator keys each circuit pool to its role and depth. S2's test circuits are therefore S0's test circuits, measured at L4 instead of L1 and L3, while every S4 test circuit is new.

**Data.** `tools/shift_transfer.py generate` writes one split-v2 artifact per shift and dataset seed (101, 211, 307) with `generate_split`. The master seed is the dataset seed. Each artifact must pass `validate_split_artifact`, and `prepare` checks that its recorded specification equals the one above. The seed-101 artifacts must reproduce the prototype's dataset hashes. The S0 rows come from the 2,048-shot level datasets of release `descriptor-information-v1` (asset `shot-sweep-v1`). These are byte-equal to the regenerated campaign datasets, and their dataset hashes must equal those analysis A recorded for the S0 ladder.

**Source identity (precondition).** `prepare` compares each artifact's train and validation rows with the S0 dataset of the same seed, field by field. The fields include the circuit, couplings, depth, label, noisy estimate, standard error, and counts hash; `IDENTITY_FIELDS` in the tool lists all of them. It also checks the target rows: 160 circuits per family, depth 6 at L1 and L3 for S4, and depth 3 at L4 for S2. S2's target circuits must be exactly S0's test circuits, and no S4 target circuit may occur in S0. If either check fails for any artifact, the experiment stops and the failure is reported.

**Models.** The cache for each shift and seed holds the S0 dataset's train and validation rows, cached by the ladder's own preparation. Its test set is the artifact's target rows. Arms F and C are fitted at R0, N3, R4, and R5 at learner seeds 1 to 20. Candidates, selection, and convergence flags are those of the ladder (rule 2026-10-02), through the same code. No fit reads a strength indicator: none is defined for L4, and the S0 ladder fits read none. N3 keeps S0's coupling-noise draw for every S0 circuit, including the test circuits S2 reuses. Each new target circuit draws z from `default_rng(SeedSequence([20261002, dataset_seed, index], spawn_key=(1000 + c,)))`, with c = 4 for S4 and c = 2 for S2. Here index is the circuit's position among the sorted target circuit identifiers. R4 encodes every circuit with the released ML-QEM encoder, as the ladder did. Fits run on the machine and package versions that produced the S0 ladder fits (macOS, arm64). The archived R0 gate (`artifacts/descriptor-information/gate/gate_r0.json`) must show a pass.

**Pairing check (precondition).** The training rows, learner seeds, code, and platform all match the S0 ladder's. Each fit should therefore equal the archived S0 fit with the same key, rung, arm, and learner seed. `run` checks this bit for bit on the stored validation predictions of the selected model and of both candidates. If any fit differs, the analysis does not run and the difference is reported. If all match, every shifted value comes from the same fitted models as the S0 value it is compared with.

**Estimands and labels.** `analyze` runs analysis A of rule 2026-10-02 (`tools/descriptor_ladder_analysis.py`, unchanged) on each shift's fits. Per row (dataset seed and family) and rung, it reports mean C, F, and R, D = C − F, and D/C. The label comes from the two-stage bootstrap interval of D (10,000 draws, seed 20261002). It is "F beats C" when the interval lies above zero, "C beats F" when it lies below zero, and "not distinguished" otherwise. A family's rung statement is the label shared by all three seeds, or "mixed". `analyze` adds F − R, C − R, and the change of C, F, and D/C from the S0 value of the same row and rung. These additions are point values, reported as descriptive.

## Expectations Stated in Advance (Not Decision Rules)

1. Both preconditions hold for all six artifacts and all 960 fits.
2. S4: C's error exceeds its S0 value in every row and rung, because no training circuit has depth 6. This is uncertain, because the label distribution also changes.
3. S4: at R0, both families move from "not distinguished" to "F beats C". This expectation is uncertain. At R0 the S0 F fits score close to C, which suggests that they rely little on the noisy estimate.
4. S4: at N3, R4, and R5, both families stay "F beats C".
5. S2: C's error equals its S0 value to 1e-12 in every row and rung. The test circuits, labels, and C's inputs, N3 draws included, are unchanged, and C reads no noisy estimate. This is a structural check.
6. S2: F's error exceeds its S0 value at N3, R4, and R5 in every row.
7. F − R is negative in every row and rung of both shifts.

## Reading

The paper reports every row and rung of both shifts in an appendix table and summarizes R0 and R5 in the main text, beside the S0 values.

- **S4.** The paper reports C, F, R, D, and D's interval in every row and rung, with the frozen labels and family statements, whatever the expectations show. At depth 6 the exact descriptors still determine each label, so S4 tests whether the fitted learners extrapolate from depth 3 to depth 6. An interval above zero shows that these learners gain from the measurement under that extrapolation. An interval spanning zero leaves the comparison unresolved, and one below zero favors C.
- **S2.** S2 changes only the measurement noise and holds test circuits, labels, and descriptor values fixed. The paper reports the same quantities and labels, whatever the expectations show. Changes in F describe the fitted full method's response to L4. Once expectation 5 holds, the change in D equals the negative of F's change.

No result changes a shift, target value, rung, arm, seed, or test set. Any analysis chosen after these results are seen is labeled post hoc. Claims extend to these two shifts of the spin-chain setting only: one target value each, one noise family, and ten qubits.

## Deviations and Release

Any deviation is reported with its reason. The artifacts are regenerable from this rule and the code, and their SHA-256 values are recorded in `prepare.json`. The repository adds `prepare.json`, the two pairing checks, the two analysis-A files, and `shift-report.json` under `artifacts/descriptor-information/shift-transfer/`. The fits and the generation logs go to a release asset.
