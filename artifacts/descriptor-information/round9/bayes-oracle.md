# Bayes Oracle Evaluation Summary

- Evaluation split: test (pseudo-test: False)
- Elapsed time: 257.23 seconds
- Dataset seeds: [101, 211, 307, 401, 503, 607]
- Shot levels: [256, 1024, 2048, 8192, 32768, 131072]

## Shot-Noise Model Checks

- Level 256: N=46080, mean(z)=0.0042, sd(z)=1.0063, max(|z|)=4.5498
- Level 1024: N=46080, mean(z)=0.0065, sd(z)=1.0051, max(|z|)=4.0624
- Level 2048: N=46080, mean(z)=-0.0044, sd(z)=0.9990, max(|z|)=4.3408
- Level 8192: N=46080, mean(z)=-0.0021, sd(z)=0.9974, max(|z|)=4.6601
- Level 32768: N=46080, mean(z)=0.0005, sd(z)=0.9995, max(|z|)=4.2554
- Level 131072: N=46080, mean(z)=-0.0016, sd(z)=0.9997, max(|z|)=4.4003

## Surrogate Failure Flags (test MAE >= 1e-3)

- Flagged surrogates: 0 / 72
- Flagged oracle rows: 0 / 432
- Flagged Equation 2 rows: 0 / 696
- Flagged confusion entries: 0 / 696

## Excluded Mapping Tuples

- Total excluded tuples: 504

## Equation 2 Identity Verification

- Verified to 1e-12: True
- Max identity error: 8.33e-17
- Evaluated pipeline rows: 696

