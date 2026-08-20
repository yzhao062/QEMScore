# CDR viability probe summary

Payload SHA-256: `043839b9eec041364a38d4b7f78fc8d7acaa8fe07a3576c871f19cd5a512ba38`.
Seed: `20260819`. Targets: 31. Noise: `depolarizing_readout` `L2`.

The exact design is the two-parameter affine matrix `[1, noisy expectation]`. The sampled design uses the shipped preset shot count. Circuit hashes are taken both before and after the execution compilation; the tables below use executed hashes.

## Compiled targets

| Target | Preset | n | Depth | Graph | G | Total RZ | Source hash |
|---|---|---:|---:|---|---:|---:|---|
| `tfi-n3-steps1` | `t0-micro` | 3 | 1 |  | 5 | 11 | `bef02c203f2acd03ae29aed8e4cd1ea29f012cc76caa0b9339ac1abb51f52a87` |
| `tfi-n3-steps2` | `t0-micro` | 3 | 2 |  | 10 | 22 | `7b3db64f5c0142086e7643cecce746bf6f47cdc7655af4471c23cac9a317614d` |
| `tfi-n4-steps1` | `t0-smoke` | 4 | 1 |  | 7 | 15 | `a7a3df963825b579cf7cb06e6867485fe1882f2bbff6c3a7fd508365d1370661` |
| `tfi-n4-steps2` | `t0-smoke` | 4 | 2 |  | 14 | 30 | `28a80648d819b042fe9eee54f19ce2f9ca6726b0d676b72202b3e08b238bb4ed` |
| `tfi-n4-steps3` | `t0-smoke` | 4 | 3 |  | 21 | 45 | `7b0975c55786954a4c8ab9c9f14f51a6a95ac46cdfeae780b5a3ae3690c3707e` |
| `tfi-n5-steps1` | `t0-smoke` | 5 | 1 |  | 9 | 19 | `62471c1c6cdfb3ef1ed8b1994c8efabdf0adb713fdcc35241a661c17405c6e27` |
| `tfi-n5-steps2` | `t0-smoke` | 5 | 2 |  | 18 | 38 | `3aac8c2a0d0e4345e5dda87541913afa09611a912cec58116223e4eda1969eb6` |
| `tfi-n5-steps3` | `t0-smoke` | 5 | 3 |  | 27 | 57 | `4809b5c910b90fa6cf3596a6a5b515895c80d445e2dfc0e543a2f183e3b942d3` |
| `tfi-n6-steps1` | `t0-smoke` | 6 | 1 |  | 11 | 23 | `76ff191dc7781b978886f2ac766960961516e38bce22c384c95a2ff285d1e24c` |
| `tfi-n6-steps2` | `t0-smoke` | 6 | 2 |  | 22 | 46 | `d9785d5c9276634e586dd74fb0eb5555beedbd289ac237f076328e09fea9851c` |
| `tfi-n6-steps3` | `t0-smoke` | 6 | 3 |  | 33 | 69 | `69391d1e4008804c81dad9e3a320819ec09d6186c54a01bdcf265c81c07f9e13` |
| `qaoa-n4-p1-path` | `t0-qaoa-micro` | 4 | 1 | path | 7 | 23 | `2b459e085bea0b1b5beea767b018ba65de32ce481fb4cf6a39ca0d444e808516` |
| `qaoa-n4-p1-cycle` | `t0-qaoa-micro` | 4 | 1 | cycle | 8 | 24 | `75e0d6910e357147ecf0a7d6c45dffd7113609ba4e6ac19325f7138c77ead53d` |
| `qaoa-n4-p1-erdos_renyi` | `t0-qaoa-micro` | 4 | 1 | erdos_renyi | 5 | 15 | `e23f415d033c1f46b9404966cf9058711fcef2d31b4011dbb98117da089caf44` |
| `qaoa-n4-p1-3_regular` | `t0-qaoa-micro` | 4 | 1 | 3_regular | 10 | 26 | `9ed71fb7850214967a485f12c5126333379fd5f65a83a603a6fea58eead3101a` |
| `qaoa-n4-p2-path` | `t0-qaoa-micro` | 4 | 2 | path | 14 | 38 | `37c147ba892de5c23401e6d98fa3b0146eca0463c08b0821d176c1c11d26d78b` |
| `qaoa-n4-p2-cycle` | `t0-qaoa-micro` | 4 | 2 | cycle | 16 | 40 | `18ab89a8a9b1679ba1dbb9b30e8c17598ab8f0b4a2e6a297c191e60768c59a1c` |
| `qaoa-n4-p2-erdos_renyi` | `t0-qaoa-micro` | 4 | 2 | erdos_renyi | 18 | 42 | `46cb5a5712210a4e0b6c2a7a38a7d1d24a48675b6c836677d03dcb6ac7153e42` |
| `qaoa-n4-p2-3_regular` | `t0-qaoa-micro` | 4 | 2 | 3_regular | 20 | 44 | `e928b15271bc6ccee1f5e681a55e1ba4fb87d2d98546ec39d3825939b0b88fbb` |
| `qaoa-n6-p1-path` | `t0-qaoa-micro` | 6 | 1 | path | 11 | 35 | `bccdbf1e33d9b7c94675a3bb359cec480b49da48612e2cefee79624e5cbd0d8c` |
| `qaoa-n6-p1-cycle` | `t0-qaoa-micro` | 6 | 1 | cycle | 12 | 36 | `8cdd71982013297b41c036e5ebf98f28511bc793dec1e595b87c730afe0eca0d` |
| `qaoa-n6-p1-erdos_renyi` | `t0-qaoa-micro` | 6 | 1 | erdos_renyi | 9 | 30 | `ba437778ed62ba668f690ca450529ee0c7733d9c1b3bec674d8dbe70c947f05a` |
| `qaoa-n6-p1-3_regular` | `t0-qaoa-micro` | 6 | 1 | 3_regular | 15 | 39 | `126a1ac59bdfa8dba6b8f45757f12622b3ac2775ae6e084d9620c8c55ee18be7` |
| `qaoa-n6-p2-path` | `t0-qaoa-micro` | 6 | 2 | path | 22 | 58 | `61f8e9b12bc371c0a518ece6e8e653251a38eae089a3454d3873ed3a8a7b3c53` |
| `qaoa-n6-p2-cycle` | `t0-qaoa-micro` | 6 | 2 | cycle | 24 | 60 | `decd491db543420ee262c54524205560c69714524abeba5a1ea4100dc1725d20` |
| `qaoa-n6-p2-erdos_renyi` | `t0-qaoa-micro` | 6 | 2 | erdos_renyi | 26 | 62 | `3f017cd5c6ff4756db7f663965a2fab54afb2c7466f37d76b28dcea20cd6e93b` |
| `qaoa-n6-p2-3_regular` | `t0-qaoa-micro` | 6 | 2 | 3_regular | 30 | 66 | `cad8309405a8dc677777814d09bd2d8284a5c2c8d3d598c3a63e6a29ec7c9780` |
| `heisenberg-n3-steps1` | `t0-heisenberg-micro` | 3 | 1 |  | 6 | 18 | `95b9340c42d8ee162c495373e7f0f3ed1570df80fb8b26632c8a2f31460164c4` |
| `heisenberg-n3-steps2` | `t0-heisenberg-micro` | 3 | 2 |  | 12 | 36 | `33997d2d973a140a680f26ec1bf6c68a5ec4e5ca090d513db3cad661ec4034d6` |
| `heisenberg-n4-steps1` | `t0-heisenberg-micro` | 4 | 1 |  | 9 | 27 | `4b18f82e17b0c1812502afb3daa3a2382dbb59d18c64291a9fd426499c09d0ca` |
| `heisenberg-n4-steps2` | `t0-heisenberg-micro` | 4 | 2 |  | 18 | 54 | `34eb7dcdd4cf68c256bab4efbe60488f8031b1b9a6691e1668783bd50be7ff76` |

## Training-set metrics at m = 70

Each row is one family and construction configuration. Ranges are across that family's target configurations. Rank counts use two observables per feasible target. A condition number of `inf` means that at least one exact design is rank deficient.

### tfi

| Replacement | Retained rule | r | Exact construction space | Executed distinct | Duplicate rate | Ideal-label range | Exact rank 2/rows | Sampled rank 2/rows | Worst exact condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closest | `mitiq-default-fraction-0.1` | 0 to 3 | 1 to 5456 | 1 to 66 | 5.7% to 98.6% | 0 to 0.426 | 20/22 | 22/22 | inf |
| closest | `fixed-0` | 0 | 1 | 1 | 98.6% | 0 to 0 | 0/22 | 22/22 | inf |
| closest | `fixed-1` | 1 | 5 to 33 | 5 to 11 | 84.3% to 92.9% | 0.013 to 0.113 | 22/22 | 22/22 | 94.694 |
| closest | `fixed-2` | 2 | 10 to 528 | 10 to 45 | 35.7% to 85.7% | 0.013 to 0.426 | 22/22 | 22/22 | 72.592 |
| closest | `fixed-3` | 3 | 10 to 5456 | 10 to 66 | 5.7% to 85.7% | 0.016 to 0.426 | 22/22 | 22/22 | 62.483 |
| closest | `fixed-5` | 5 | 1 to 237336 | 1 to 70 | 0.0% to 98.6% | 0 to 0.491 | 20/22 | 22/22 | inf |
| uniform | `mitiq-default-fraction-0.1` | 0 to 3 | 1024 to 6.290e+21 | 69 to 70 | 0.0% to 1.4% | 1.927 to 2.000 | 22/22 | 22/22 | 5.868 |
| uniform | `fixed-0` | 0 | 1024 to 7.379e+19 | 69 to 70 | 0.0% to 1.4% | 2.000 to 2.000 | 22/22 | 22/22 | 5.590 |
| uniform | `fixed-1` | 1 | 1280 to 6.087e+20 | 69 to 70 | 0.0% to 1.4% | 1.955 to 2.000 | 22/22 | 22/22 | 5.768 |
| uniform | `fixed-2` | 2 | 640 to 2.435e+21 | 66 to 70 | 0.0% to 5.7% | 2.000 to 2.000 | 22/22 | 22/22 | 5.230 |
| uniform | `fixed-3` | 3 | 160 to 6.290e+21 | 51 to 70 | 0.0% to 27.1% | 1.789 to 2.000 | 22/22 | 22/22 | 5.868 |
| uniform | `fixed-5` | 5 | 1 to 1.710e+22 | 1 to 70 | 0.0% to 98.6% | 0 to 2.000 | 20/22 | 22/22 | inf |

### qaoa

| Replacement | Retained rule | r | Exact construction space | Executed distinct | Duplicate rate | Ideal-label range | Exact rank 2/rows | Sampled rank 2/rows | Worst exact condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closest | `mitiq-default-fraction-0.1` | 0 to 3 | 1 to 4060 | 1 to 70 | 0.0% to 98.6% | 0 to 0.417 | 4/32 | 32/32 | inf |
| closest | `fixed-0` | 0 | 1 | 1 | 98.6% | 0 to 0 | 0/32 | 32/32 | inf |
| closest | `fixed-1` | 1 | 5 to 30 | 5 to 28 | 60.0% to 92.9% | 0 to 0.417 | 1/32 | 32/32 | inf |
| closest | `fixed-2` | 2 | 10 to 435 | 10 to 67 | 4.3% to 85.7% | 0 to 0.417 | 10/32 | 32/32 | inf |
| closest | `fixed-3` | 3 | 10 to 4060 | 10 to 70 | 0.0% to 85.7% | 0 to 0.417 | 12/32 | 32/32 | inf |
| closest | `fixed-5` | 5 | 1 to 142506 | 1 to 70 | 0.0% to 98.6% | 0 to 0.570 | 13/32 | 32/32 | inf |
| uniform | `mitiq-default-fraction-0.1` | 0 to 3 | 1024 to 7.314e+19 | 69 to 70 | 0.0% to 1.4% | 0 to 2.000 | 14/32 | 32/32 | inf |
| uniform | `fixed-0` | 0 | 1024 to 1.153e+18 | 69 to 70 | 0.0% to 1.4% | 0 to 2.000 | 13/32 | 32/32 | inf |
| uniform | `fixed-1` | 1 | 1280 to 8.647e+18 | 66 to 70 | 0.0% to 5.7% | 0 to 2.000 | 13/32 | 32/32 | inf |
| uniform | `fixed-2` | 2 | 640 to 3.135e+19 | 65 to 70 | 0.0% to 7.1% | 0 to 2.000 | 14/32 | 32/32 | inf |
| uniform | `fixed-3` | 3 | 160 to 7.314e+19 | 57 to 70 | 0.0% to 18.6% | 0 to 2.000 | 13/32 | 32/32 | inf |
| uniform | `fixed-5` | 5 | 1 to 1.604e+20 | 1 to 70 | 0.0% to 98.6% | 0 to 1.989 | 14/32 | 32/32 | inf |

### heisenberg

| Replacement | Retained rule | r | Exact construction space | Executed distinct | Duplicate rate | Ideal-label range | Exact rank 2/rows | Sampled rank 2/rows | Worst exact condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closest | `mitiq-default-fraction-0.1` | 1 to 2 | 6 to 153 | 6 to 48 | 31.4% to 91.4% | 0.031 to 0.124 | 8/8 | 8/8 | 150.375 |
| closest | `fixed-0` | 0 | 1 | 1 | 98.6% | 0 to 0 | 0/8 | 8/8 | inf |
| closest | `fixed-1` | 1 | 6 to 18 | 6 to 12 | 82.9% to 91.4% | 0.031 to 0.064 | 8/8 | 8/8 | 150.375 |
| closest | `fixed-2` | 2 | 15 to 153 | 15 to 48 | 31.4% to 78.6% | 0.048 to 0.124 | 8/8 | 8/8 | 90.180 |
| closest | `fixed-3` | 3 | 20 to 816 | 20 to 63 | 10.0% to 71.4% | 0.048 to 0.247 | 8/8 | 8/8 | 91.263 |
| closest | `fixed-5` | 5 | 6 to 8568 | 6 to 70 | 0.0% to 91.4% | 0.048 to 0.295 | 8/8 | 8/8 | 111.697 |
| uniform | `mitiq-default-fraction-0.1` | 1 to 2 | 6144 to 6.571e+11 | 70 | 0.0% | 2.000 to 2.000 | 8/8 | 8/8 | 6.020 |
| uniform | `fixed-0` | 0 | 4096 to 6.872e+10 | 68 to 70 | 0.0% to 2.9% | 1.000 to 2.000 | 8/8 | 8/8 | 8.561 |
| uniform | `fixed-1` | 1 | 6144 to 3.092e+11 | 70 | 0.0% | 2.000 to 2.000 | 8/8 | 8/8 | 6.706 |
| uniform | `fixed-2` | 2 | 3840 to 6.571e+11 | 68 to 70 | 0.0% to 2.9% | 1.968 to 2.000 | 8/8 | 8/8 | 6.020 |
| uniform | `fixed-3` | 3 | 1280 to 8.762e+11 | 68 to 70 | 0.0% to 2.9% | 1.973 to 2.000 | 8/8 | 8/8 | 8.052 |
| uniform | `fixed-5` | 5 | 24 to 5.750e+11 | 24 to 70 | 0.0% to 65.7% | 1.933 to 2.000 | 8/8 | 8/8 | 8.174 |

## Interpretation notes

- `construction_space_size_exact` counts retained-position choices and, for uniform replacement, the four Clifford-angle choices at every replaced position.

- Distinct and duplicate metrics use post-substitution, post-execution-compilation hashes. The JSON also retains every ordered source and executed hash.

- Exact rank is the structural check. Sampled rank is reported separately because independent shot noise can give repeated circuits different noisy estimates.

- Full per-target, per-observable metrics for every reported m are in the JSON artifact.
