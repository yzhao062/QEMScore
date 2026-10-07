# Zero-Shot Transfer Under a Depth Shift (S4) and a Noise-Strength Shift (S2)

Records of the frozen rule `docs/frozen-rules/2026-10-06-shift-transfer.md`, run with `tools/shift_transfer.py` at commit `24a2cbe` (the rule's commit, pushed before any fit). The six artifacts were generated locally with `generate`; `prepare`, `run`, and `analyze` then ran in one session on the machine that produced the S0 ladder fits (macOS, arm64).

## Files

- `S4/` and `S2/`: `prepare.json` (artifact and S0 hashes, source identity, target structure, encoder checks), `pairing_check.json`, `run_config.json`, `run_<time>.json` (driver summary), and `analysis-a.json` (analysis A of rule 2026-10-02 on the shift's fits).
- `shift-report.json`: every row and rung of both shifts beside the S0 value of the same row and rung, with F − R, C − R, and the changes of C, F, and D/C.
- `release-asset-sha256.txt`: the SHA-256 of the release asset `shift-transfer-fits-v1.tar.xz`. It holds the 960 fits, with selected and per-candidate predictions and their records. It also holds the N3 coupling-noise tables, the R4 encoder caches, and the fit, generation, and run logs.

## Preconditions

- **Data.** All six artifacts passed `validate_split_artifact` and matched the frozen specification. The seed-101 artifacts reproduced the prototype's dataset hashes. Generation took 32 to 36 minutes per artifact, six running at once.
- **Source identity.** In every artifact, all 5,120 training and 2,560 validation rows equal the S0 dataset's rows in every identity field. S2's 320 target circuits are exactly S0's test circuits; none of S4's 320 occurs in S0.
- **Pairing.** All 480 fits of each shift equal the archived S0 ladder fit bit for bit on the stored validation predictions (selected model and both candidates).
- **Gate.** `require_gate` accepted the archived R0 gate, which passed under the amendment disclosed on 2026-10-02 (tolerance 1.4e-11 instead of 1e-11). The descriptor-information ladder ran under the same amendment.

## Results

Means over dataset seeds 101, 211, and 307 of test macro errors, in units of 0.001. Reading: the label all three seeds share, else mixed; n.d.: not distinguished.

| Shift | Family | Rung | R | C | F | Reading | S0 C | S0 F | S0 reading |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| S4 | TFI | R0 | 335.4 | 341.9 | 297.0 | n.d. | 2.9 | 2.8 | n.d. |
| S4 | TFI | N3 | 335.4 | 339.4 | 272.3 | F beats C | 60.9 | 35.8 | F beats C |
| S4 | TFI | R4 | 335.4 | 294.3 | 264.2 | mixed | 38.7 | 26.3 | F beats C |
| S4 | TFI | R5 | 335.4 | 315.2 | 277.3 | mixed | 210.7 | 92.1 | F beats C |
| S4 | Heis. | R0 | 609.9 | 623.6 | 551.6 | n.d. | 3.7 | 4.1 | n.d. |
| S4 | Heis. | N3 | 609.9 | 550.2 | 465.0 | n.d. | 49.8 | 35.8 | F beats C |
| S4 | Heis. | R4 | 609.9 | 300.8 | 623.8 | mixed | 70.9 | 49.0 | F beats C |
| S4 | Heis. | R5 | 609.9 | 436.5 | 476.6 | n.d. | 94.2 | 59.1 | F beats C |
| S2 | TFI | R0 | 530.3 | 2.9 | 4.2 | mixed | 2.9 | 2.8 | n.d. |
| S2 | TFI | N3 | 530.3 | 60.9 | 177.8 | C beats F | 60.9 | 35.8 | F beats C |
| S2 | TFI | R4 | 530.3 | 38.7 | 112.3 | C beats F | 38.7 | 26.3 | F beats C |
| S2 | TFI | R5 | 530.3 | 210.7 | 323.9 | C beats F | 210.7 | 92.1 | F beats C |
| S2 | Heis. | R0 | 887.7 | 3.7 | 4.5 | mixed | 3.7 | 4.1 | n.d. |
| S2 | Heis. | N3 | 887.7 | 49.8 | 50.3 | n.d. | 49.8 | 35.8 | F beats C |
| S2 | Heis. | R4 | 887.7 | 70.9 | 84.3 | C beats F | 70.9 | 49.0 | F beats C |
| S2 | Heis. | R5 | 887.7 | 94.2 | 128.4 | C beats F | 94.2 | 59.1 | F beats C |

**S4.** Neither learned arm extrapolates from depth 3 to depth 6. Both errors rise to about the raw estimate's: at R0 by a factor of 103 to 142 for F. F stays below R in every row and rung except Heisenberg R4, where it errs more than R on two seeds (F − R up to +0.083). D/C points are positive at R0 and N3 (+0.08 to +0.21), but only transverse-field Ising N3 reads "F beats C" on all three seeds. Heisenberg R4 reads "C beats F" on two seeds.

**S2.** C's errors equal the S0 values exactly in every row and rung. F's error rises by a factor of 1.1 to 1.6 at R0 and 1.4 to 5.6 at N3, R4, and R5. Every rung that read "F beats C" under S0 now reads "C beats F", except Heisenberg N3, which reads "not distinguished". At R0, transverse-field Ising reads "C beats F" on two seeds and Heisenberg on one. F still errs far less than R in every row and rung (F − R from −0.89 to −0.20).

## Expectations

1. Held: both preconditions hold for all six artifacts and all 960 fits.
2. Held: under S4, C's error exceeds its S0 value in every row and rung.
3. Failed: under S4 at R0, both families read "not distinguished".
4. Failed except transverse-field Ising N3: under S4, N3, R4, and R5 do not all stay "F beats C".
5. Held: under S2, C's error equals its S0 value (largest difference 0).
6. Held: under S2, F's error exceeds its S0 value at N3, R4, and R5 in every row.
7. Failed in two cells: F − R is negative everywhere except Heisenberg R4 under S4 on seeds 101 and 211.

## Reproduction

```
MLQEM_PATH=<ml-qem at b1eccf8>/docs/tutorials/mlp.py PYTHONPATH=. python tools/shift_transfer.py generate --root DATA
MLQEM_PATH=... PYTHONPATH=. python tools/shift_transfer.py prepare --root DATA --s0-data <descriptor-information-v1 shot-sweep-v1>/levels/shots-2048 --out OUT
PYTHONPATH=. python tools/shift_transfer.py run --out OUT --s0-fits <descriptor-information-v1>/fits
PYTHONPATH=. python tools/shift_transfer.py analyze --out OUT
```
