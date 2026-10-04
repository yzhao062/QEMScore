"""Tests for post hoc learner adequacy and seed ensembling analysis."""

from pathlib import Path
import numpy as np
import pytest

from tools import descriptor_ladder_analysis as dla
from tools import learner_adequacy as la


def test_spread_arithmetic():
    """Verify compute_spread min, max, median, and ddof=1 sample SD."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = la.compute_spread(vals)
    assert res["n"] == 5
    assert res["min"] == 1.0
    assert res["max"] == 5.0
    assert res["median"] == 3.0
    assert res["mean"] == 3.0
    assert np.isclose(res["sd"], np.std(vals, ddof=1))


def test_seed_ensemble_arithmetic_and_sign_flip():
    """Verify that ensemble averaging before scoring reduces error and can flip point sign of D."""
    # Synthetic toy with 2 circuits, 2 cells, 4 items total.
    # item 0: circuit 0, cell 0
    # item 1: circuit 0, cell 1
    # item 2: circuit 1, cell 0
    # item 3: circuit 1, cell 1
    test_rows = [
        {"item_id": "i0", "circuit_id": "c0", "family": "tfi", "noise_family": "depol",
         "severity": "L1", "observable": "z", "ideal_expectation": 0.0, "noisy_expectation": 0.0},
        {"item_id": "i1", "circuit_id": "c0", "family": "tfi", "noise_family": "depol",
         "severity": "L2", "observable": "z", "ideal_expectation": 0.0, "noisy_expectation": 0.0},
        {"item_id": "i2", "circuit_id": "c1", "family": "tfi", "noise_family": "depol",
         "severity": "L1", "observable": "z", "ideal_expectation": 0.0, "noisy_expectation": 0.0},
        {"item_id": "i3", "circuit_id": "c1", "family": "tfi", "noise_family": "depol",
         "severity": "L2", "observable": "z", "ideal_expectation": 0.0, "noisy_expectation": 0.0},
    ]
    row = dla.Row({"test": test_rows}, "tfi", "toy/tfi")

    # 2 learner seeds:
    # Under seed 1: C predicts +2.0, F predicts +1.0 -> err C = 2.0, err F = 1.0 -> D_1 = +1.0
    # Under seed 2: C predicts -2.0, F predicts +1.0 -> err C = 2.0, err F = 1.0 -> D_2 = +1.0
    # Per-seed macro MAEs: C_1 = 2.0, F_1 = 1.0, C_2 = 2.0, F_2 = 1.0
    # Script A mean D = mean(C) - mean(F) = 2.0 - 1.0 = +1.0 (positive!)
    # But ensemble predictions:
    # C mean prediction = (+2.0 + (-2.0)) / 2 = 0.0
    # F mean prediction = (+1.0 + (+1.0)) / 2 = +1.0
    # Ensemble MAE: E_C = |0.0 - 0.0| = 0.0, E_F = |1.0 - 0.0| = 1.0
    # D_ens = E_C - E_F = 0.0 - 1.0 = -1.0 (negative -> SIGN FLIP!)
    c_preds = np.array([
        [2.0, 2.0, 2.0, 2.0],
        [-2.0, -2.0, -2.0, -2.0],
    ])
    f_preds = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ])

    # 3 draws for testing:
    # draw 0: both seeds drawn once: sc = [1, 1]
    # draw 1: seed 1 drawn twice: sc = [2, 0]
    # draw 2: seed 2 drawn twice: sc = [0, 2]
    seed_counts = np.array([
        [1, 1],
        [2, 0],
        [0, 2],
    ])
    # circuit counts: both circuits drawn once in each draw
    circuit_counts = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
    ])

    ens_res = la.compute_cell_ensemble(row, c_preds, f_preds, seed_counts, circuit_counts, seeds=[1, 2])

    assert np.isclose(ens_res["E_C"], 0.0)
    assert np.isclose(ens_res["E_F"], 1.0)
    assert np.isclose(ens_res["D_ens"]["point"], -1.0)

    # In draw 0 (sc = [1, 1]), ensemble prediction C is 0.0, F is 1.0 -> d_ens = -1.0
    assert np.isclose(ens_res["draws_D_ens"][0], -1.0)
    # In draw 1 (sc = [2, 0]), ensemble prediction C is 2.0, F is 1.0 -> d_ens = 2.0 - 1.0 = +1.0
    assert np.isclose(ens_res["draws_D_ens"][1], +1.0)
    # In draw 2 (sc = [0, 2]), ensemble prediction C is -2.0, F is 1.0 -> d_ens = 2.0 - 1.0 = +1.0
    assert np.isclose(ens_res["draws_D_ens"][2], +1.0)


def test_pooled_bootstrap_shares_seed_draw_between_family_rows():
    """Verify that joint bootstrap applies one seed draw to both family rows of a dataset seed,
    resamples circuits separately per row, and draws dataset seeds independently."""
    dataset_seeds = (101, 211, 307)
    row_circuits = {
        "shipped-s101-n640/tfi": 10,
        "shipped-s101-n640/heisenberg": 12,
        "shipped-s211-n640/tfi": 10,
        "shipped-s211-n640/heisenberg": 12,
        "shipped-s307-n640/tfi": 10,
        "shipped-s307-n640/heisenberg": 12,
    }
    n_seeds = 20
    draws = 100
    seed = 20261002

    sc_by_ds, cc_by_row = la.generate_pooled_draws(
        dataset_seeds, row_circuits, n_seeds=n_seeds, draws=draws, seed=seed
    )

    # 1. Each dataset seed has exactly one seed-count matrix of shape (draws, n_seeds)
    for ds in dataset_seeds:
        assert sc_by_ds[ds].shape == (draws, n_seeds)
        # Check that seed counts sum to n_seeds in every draw
        assert np.all(sc_by_ds[ds].sum(axis=1) == n_seeds)

    # 2. Within dataset seed 101, the seed draw is identical for both family rows
    # (they both access sc_by_ds[101]).
    # We verify that circuits for tfi and heisenberg are resampled separately:
    cc_tfi_101 = cc_by_row["shipped-s101-n640/tfi"]
    cc_heis_101 = cc_by_row["shipped-s101-n640/heisenberg"]
    assert cc_tfi_101.shape == (draws, 10)
    assert cc_heis_101.shape == (draws, 12)
    assert np.all(cc_tfi_101.sum(axis=1) == 10)
    assert np.all(cc_heis_101.sum(axis=1) == 12)

    # 3. Dataset seeds are drawn independently (sc for 101 != sc for 211 across draws)
    diff = np.sum(np.abs(sc_by_ds[101] - sc_by_ds[211]))
    assert diff > 0, "different dataset seeds should have independent seed draws"


def test_pooled_r0_synthetic_computation():
    """Verify compute_pooled_r0 arithmetic on small synthetic mock objects."""
    dataset_seeds = (101, 211, 307)
    draws = 50
    n_seeds = 2

    # Create 6 synthetic rows
    r0_rows = []
    row_circuits = {}
    for ds in dataset_seeds:
        for fam in ("tfi", "heisenberg"):
            label = f"shipped-s{ds}-n640/{fam}"
            row_circuits[label] = 4
            test_rows = [
                {"item_id": f"{label}-i{i}", "circuit_id": f"c{i//2}", "family": fam,
                 "noise_family": "depol", "severity": "L1", "observable": "z",
                 "ideal_expectation": 0.0, "noisy_expectation": 0.0}
                for i in range(4)
            ]
            row = dla.Row({"test": test_rows}, fam, label)
            row_circuits[label] = row.n_circuits
            # Both seeds predict 0.01 for F, 0.03 for C
            c_preds = np.full((n_seeds, 4), 0.03)
            f_preds = np.full((n_seeds, 4), 0.01)
            pts_c = np.full(n_seeds, 0.03)
            pts_f = np.full(n_seeds, 0.01)
            r0_rows.append({
                "label": label,
                "dataset_seed": ds,
                "family": fam,
                "row": row,
                "c_preds": c_preds,
                "f_preds": f_preds,
                "c_seed_points": pts_c,
                "f_seed_points": pts_f,
            })

    sc_by_ds, cc_by_row = la.generate_pooled_draws(
        dataset_seeds, row_circuits, n_seeds=n_seeds, draws=draws, seed=20261002
    )
    res = la.compute_pooled_r0(r0_rows, sc_by_ds, cc_by_row, n_seeds=n_seeds)

    # Standard: pooled D = 0.03 - 0.01 = 0.02; pooled D/C = 0.02 / 0.03 = 2/3
    assert np.isclose(res["standard"]["pooled_D"]["point"], 0.02)
    assert np.isclose(res["standard"]["pooled_D_over_C"]["point"], 2.0 / 3.0)

    # Ensemble: pooled D_ens = 0.02; pooled D_ens/C = 2/3
    assert np.isclose(res["ensemble"]["pooled_D_ens"]["point"], 0.02)
    assert np.isclose(res["ensemble"]["pooled_D_ens_over_C"]["point"], 2.0 / 3.0)
    assert res["sign_flip"] is False
