> Note: Written 2026-10-01 before the corresponding run (file modification time 20:44 PDT on 2026-10-01; the learner-seed fits started 03:57 UTC on 2026-10-02 = 20:57 PDT on 2026-10-01). Committed later, so the repository commit date postdates the runs. The dataset path under Population names the author's local copy at writing time; the datasets regenerate with tools/regenerate_example.py.

# Frozen Decision Rule: Learner-Seed Replication of the Capacity-Matched Gap

Written 2026-10-01 before any replication run in this round. Do not change it after results are seen; report any deviation explicitly.

## Population

The six primary rows: shipped step sizes, 640 training circuits per family, dataset seeds 101, 211, 307, families transverse-field Ising (TFI) and Heisenberg. Datasets: the regenerated item streams at
/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/data/regen-shipped-s{101,211,307}-n640/items.jsonl
(validation and test items match the archived records exactly; verify before use).

## Fits

For each dataset and each learner seed k in 1..20, fit three arms with the campaign's own code path, exactly as `qemscore/campaign/analysis.py::_fit_arm` does:

- F (full, archive name `liao`): `LiaoMitigator(random_state=k).fit(train, validation)`.
- C (capacity-matched control, `liao-feat-only`): `LiaoMitigator(random_state=k, drop_features=("noisy_expectation",)).fit(train, validation)`.
- P (training-shuffle refit, `liao-training-shuffle`): `LiaoMitigator(random_state=k).fit(shuffle_noisy_items(train, seed=k), validation)`, using the campaign's `shuffle_noisy_items`.

This includes the full RF and MLP candidate search and the one-standard-error selection rule. Also refit the original seed (k = dataset seed) and confirm it reproduces the archived test predictions; report that check, but keep it out of the 20-seed statistics.

## Outcomes

Untouched-test macro MAE, equally weighted over the four strength-by-observable cells, per family. Per row and seed: D_k = C_k − F_k, D_k / C_k, P_k − C_k, P_k − F_k, the selected candidate per arm, and convergence flags.

## Interval That Includes Training Variance

For each row, compute a two-stage percentile bootstrap of the mean over seeds of D (10,000 draws, seed 20261001). Each draw first resamples the 20 learner seeds with replacement. It then resamples the 160 test physical circuits with replacement, once per draw and shared across arms and seeds, keeping all four rows of a circuit together. The interval is the 2.5th to 97.5th percentile. Report the same interval for the mean of P − C and P − F.

## Decision Rule (Frozen)

- A row supports a positive capacity-matched gap only if its two-stage 95 percent interval for mean D lies wholly above zero.
- If at least five rows support it, Finding 2 keeps its form with the seed-averaged numbers.
- If one to four rows support it, Finding 2 is rewritten to report the seed-averaged result and the count of supporting rows.
- If no row supports it, Finding 2 leaves the abstract. It is replaced by the statement that the capacity-matched gap is not distinguishable from learner-initialization variance in this setting.
- The decision uses all fits. A fit is flagged as non-converged if any non-finite value or numerical RuntimeWarning occurs during training or prediction. Report results with flagged fits excluded as a sensitivity analysis only.
