# ML-QEM Against Exact Labels and With Exact Circuit Parameters: Results

Rule: `docs/frozen-rules/2026-10-06-round9-follow-ups.md`, Parts A and B, pushed at commit 82aaaf6 before any governed computation. The commands are those of the rule; release asset `round9-run-logs-v1` holds their log. They ran from 2026-10-07T00:13:48Z to 00:36:17Z at HEAD 82aaaf6.

## Files

- `../../mlqem-exact/`: exact statevector labels per setting and split (`exact-<setting>-<split>.npz` and manifests) and `gate.json`.
- `../analysis-exact-targets.json`: the round-8 fits (release `confirmation-v1`, asset `mlqem-own-data-v1`) scored against exact labels, with `C_minus_R`, `F_minus_R`, and `reference_error`.
- `exact-descriptors-analysis-exact.json` and `exact-descriptors-analysis-archived.json`: Part B fits (exact J, basis, and step added to F, C, and P) scored against exact labels (primary) and archived references.
- `exact-training-encoding/` and `exact-training-exact/`: the secondary F and C fits of the primary cell with exact training labels, scored against both references.
- `release-asset-sha256.txt`: SHA-256 of the release assets that hold the Part B fits, the secondary fits, and the run logs (release `exact-labels-oracle-v1`). The fits are not in git.

## Part A

The gate passes on all eight setting and split pairs. Mean z lies within 0.005 of zero (0.010 for no-readout validation), its SD is 0.987 to 1.005, and max |z| is 4.64. One no-readout training circuit above step 0 (`step_8.pk:132`) has no coupling sandwich and gets J = 0. The archived references differ from the exact labels by a mean of 0.0074 to 0.0077 per setting.

| Setting | Model | C | F | D/C [95%] | C − R | F − R |
|---|---|---|---|---|---|---|
| No readout error | RF | 0.0479 | 0.0127 | 0.735 [0.725, 0.745] | −0.012 | −0.047 |
| No readout error | MLP | 0.1166 | 0.0223 | 0.808 [0.801, 0.816] | +0.057 | −0.038 |
| No readout error | OLS | 0.1521 | 0.0451 | 0.704 [0.697, 0.710] | +0.092 | −0.015 |
| Readout error | RF | 0.0576 | 0.0134 | 0.768 [0.755, 0.780] | −0.048 | −0.092 |
| Readout error | MLP | 0.1251 | 0.0231 | 0.816 [0.805, 0.826] | +0.020 | −0.082 |
| Readout error | OLS | 0.1979 | 0.0572 | 0.711 [0.702, 0.719] | +0.093 | −0.048 |
| Coherent over-rotation | RF | 0.0479 | 0.0218 | 0.544 [0.514, 0.573] | −0.061 | −0.087 |
| Coherent over-rotation | MLP (validation only) | 0.1179 | 0.0649 | 0.450 [0.424, 0.474] | +0.009 | −0.044 |
| Coherent over-rotation | OLS | 0.1517 | 0.0857 | 0.435 [0.413, 0.457] | +0.043 | −0.023 |

Expectations, all held:

1. The gate passes.
2. Primary-cell D/C against exact labels (0.735) is at least its archived value (0.688).
3. Every cell keeps "F beats C".
4. F − R lies below zero in every cell.

## Part B

Against exact labels, the random forest reads "C beats F" in every setting, and its C falls from 0.048 to 0.008. Its D/C by setting:

- No readout error: −0.364 [−0.403, −0.327].
- Readout error: −0.380 [−0.446, −0.317].
- Coherent over-rotation: −1.038 [−1.159, −0.922].

Least squares and the MLP read "F beats C" (D/C 0.31 to 0.73). Against archived references the labels are the same. For the random forest, P errs within 0.004 of F.

Secondary fits use the primary cell, exact training labels, and exact scoring labels. With the published encoding, D/C is 0.752 [0.742, 0.762]. With exact parameters, C is 0.0050, F 0.0097, and D/C −0.948 [−1.033, −0.870].

Expectation: primary-cell D/C with exact parameters falls below Part A's 0.735, held.

No deviation from the rule occurred.
