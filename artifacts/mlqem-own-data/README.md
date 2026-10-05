# ML-QEM on Its Own Data: Results

Rule: `docs/frozen-rules/2026-10-04-mlqem-own-data.md` (pushed at commit b04f49b before any fit on these data). Code and commands: `reanalysis/mlqem/`.

- `analysis.json`: the frozen analysis (`reanalysis/mlqem/analyze.py`, 10,000 draws, bootstrap seed 20261002, learner seeds 1 to 20) for the three settings and three models, with `reproduction_check`.
- `run-log.jsonl`: the runner's environment record (ml-qem commit b1eccf8, the loaded `blackwater` path, package versions, thread settings, arguments, rule SHA-256), written before encoding.
- Timing: the record itself carries no timestamp. The file modification times, which the release archive preserves, place it after the push of commit b04f49b to GitHub at 2026-10-05T03:47:32Z (remote-tracking reflog): run log 03:49:40Z, encoded matrices 03:50:43Z to 03:52:18Z, and the 438 fits 03:52:22Z to 03:58:19Z.
- `encoded/*.json`: the encoded feature matrices' records, with the SHA-256 of every data file read (all match `reanalysis/mlqem/upstream_data_sha256.json`).
- `release-asset-sha256.txt`: the SHA-256 of `mlqem-own-data-v1.tar.gz`, which holds the 438 fits (predictions and records) and the encoded matrices.

Results. The reproduction check passes. On the no-readout test circuits the 20-seed mean random-forest error is 0.015533 against the published 0.015510 (ratio 1.0015; per seed 0.015508 to 0.015572). Reading 1 of the rule therefore applies. Every setting and model reads "F beats C".

| Setting | Model | C | F | D/C [95% interval] | F − R |
|---|---|---|---|---|---|
| No readout error | Random forest | 0.0497 | 0.0155 | 0.688 [0.676, 0.699] | −0.045 |
| No readout error | MLP | 0.1170 | 0.0239 | 0.796 [0.788, 0.803] | −0.037 |
| No readout error | OLS | 0.1525 | 0.0460 | 0.698 [0.691, 0.705] | −0.015 |
| Readout error | Random forest | 0.0592 | 0.0160 | 0.729 [0.715, 0.742] | −0.089 |
| Readout error | MLP | 0.1257 | 0.0245 | 0.805 [0.794, 0.815] | −0.081 |
| Readout error | OLS | 0.1981 | 0.0581 | 0.707 [0.698, 0.715] | −0.047 |
| Coherent over-rotation | Random forest | 0.0497 | 0.0242 | 0.513 [0.483, 0.541] | −0.085 |
| Coherent over-rotation | MLP | 0.1182 | 0.0655 | 0.446 [0.421, 0.470] | −0.043 |
| Coherent over-rotation | OLS | 0.1519 | 0.0863 | 0.432 [0.410, 0.454] | −0.023 |

The coherent setting releases no separate test split, so its test circuits are its validation circuits (the MLP's scheduler read them). F − Rcal is below zero for every model. The MLP refit's error is 1.074 times the published MLP's (no criterion).
