"""Arms module for ML-QEM replication control.

Implements the experimental arms:
- F: Full features (descriptors + noisy expectation values, 58 features)
- C: Descriptors only (noisy expectation value columns 54..57 ablated, 54 features)
- P: Full features with noisy expectation columns permuted across training rows only
     (permutation seeded by learner_seed; validation and test rows preserved unpermuted)
- Rcal: Calibrated noisy-only baseline via OLS LinearRegression fit on training noisy columns (4 features)
- R: Raw unmitigated noisy expectation value baseline (no fitting)

Models supported matching published pipeline (Part 1.b):
- OLS: LinearRegression(fit_intercept=True)
- RF: RandomForestRegressor(n_estimators=300), other settings at their defaults (no depth limit),
      as h15_random_forest.ipynb cell 11 sets it; fit independently per observable with
      random_state=learner_seed + q
- MLP: MLP1 architecture (2-layer, hidden_size=64, output_size=4, Adam lr=0.001,
       ReduceLROnPlateau(patience=15, factor=0.1, min_lr=1e-5), batch=32, 100 epochs,
       deterministic CPU execution, validation loss scheduler step following h10_mlp.ipynb cell 13)
"""

import os
import time
import json
import random
import hashlib
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import qiskit

from blackwater.library.learning.mlp import MLP1


def set_seed_deterministic(seed: int):
    """Enforces full determinism across Python, NumPy, and PyTorch CPU."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def compute_file_sha256(path: str) -> str:
    """Computes the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_string_sha256(s: str) -> str:
    """Computes the SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prepare_arm_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_val: np.ndarray,
    noisy_range: Tuple[int, int],
    arm: str,
    learner_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Constructs train, test, and val feature matrices for the specified arm.

    Args:
        X_train: original training features (N_train x D).
        X_test: original testing features (N_test x D).
        X_val: original validation features (N_val x D).
        noisy_range: (start_col, end_col) slice of noisy expectation values.
        arm: 'F', 'C', 'P', 'R', or 'Rcal'.
        learner_seed: random seed for permutation in arm P.

    Returns:
        (X_tr, X_te, X_va, active_column_indices)
    """
    start_c, end_c = noisy_range
    total_cols = X_train.shape[1]

    if arm == "F":
        active_cols = list(range(total_cols))
        return X_train.copy(), X_test.copy(), X_val.copy(), active_cols

    elif arm == "C":
        active_cols = [c for c in range(total_cols) if c < start_c or c >= end_c]
        return X_train[:, active_cols].copy(), X_test[:, active_cols].copy(), X_val[:, active_cols].copy(), active_cols

    elif arm == "P":
        active_cols = list(range(total_cols))
        X_tr = X_train.copy()
        X_te = X_test.copy()
        X_va = X_val.copy()
        # Permute ONLY training rows for the noisy columns using learner_seed
        rng = np.random.RandomState(learner_seed)
        perm = rng.permutation(X_tr.shape[0])
        X_tr[:, start_c:end_c] = X_tr[perm, start_c:end_c]
        # Test and validation rows remain unpermuted
        return X_tr, X_te, X_va, active_cols

    elif arm in ["R", "Rcal"]:
        active_cols = list(range(start_c, end_c))
        return X_train[:, active_cols].copy(), X_test[:, active_cols].copy(), X_val[:, active_cols].copy(), active_cols

    else:
        raise ValueError(f"Unknown arm: {arm}. Must be one of 'F', 'C', 'P', 'R', 'Rcal'.")


def fit_and_predict_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray
) -> np.ndarray:
    """Fits OLS LinearRegression model on feature matrix and returns predictions."""
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


def fit_and_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    learner_seed: int
) -> np.ndarray:
    """Fits independent RandomForestRegressor per observable matching published specs (Part 1.b).

    h15_random_forest.ipynb cell 11 fits RandomForestRegressor(n_estimators=300) per observable
    and saves the list as model/ising_init_from_qasm_no_readout/rfr_list.pk, the file
    h17_compare_over_steps.ipynb loads. Every other setting keeps its default (no depth limit,
    all features per split). The published random_state is unset (the forests draw from numpy's
    global generator after fix_random_seed(0)); here it is learner_seed + q.
    """
    n_targets = y_train.shape[1] if y_train.ndim > 1 else 1
    preds_list = []

    for q in range(n_targets):
        y_col = y_train[:, q] if y_train.ndim > 1 else y_train
        rfr = RandomForestRegressor(
            n_estimators=300,
            random_state=learner_seed + q
        )
        rfr.fit(X_train, y_col)
        pred_col = rfr.predict(X_test)
        preds_list.append(pred_col)

    if n_targets > 1:
        return np.column_stack(preds_list)
    return preds_list[0]


def fit_and_predict_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learner_seed: int,
    epochs: int = 100,
    batch_size: int = 32
) -> np.ndarray:
    """Fits MLP matching published specs with deterministic CPU training (Part 1.b & Item 3).

    Follows docs/tutorials/h10_mlp.ipynb cell 13:
    ```python
    fix_random_seed(0)

    train_losses = []
    test_losses = []

    N_EPOCHS = 100

    progress = tqdm(range(N_EPOCHS), desc='Model training', leave=True)
    for epoch in progress:
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        test_loss = 0.0
        model.eval()
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

        scheduler.step(test_loss)
    ```
    """
    set_seed_deterministic(learner_seed)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    y_tr_2d = y_train if y_train.ndim > 1 else y_train[:, None]
    y_va_2d = y_val if y_val.ndim > 1 else y_val[:, None]

    model = MLP1(input_size=input_dim, hidden_size=64, output_size=output_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(learner_seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_tr_2d, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_va_2d, dtype=torch.float32)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(batch_size * 1000, 1),
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-5)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
        scheduler.step(val_loss)

    model.eval()
    with torch.no_grad():
        test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(test_t).numpy()

    if y_train.ndim == 1 and preds.shape[1] == 1:
        return preds[:, 0]
    return preds


def run_arm_job(
    setting: str,
    model: str,
    arm: str,
    learner_seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noisy_range: Tuple[int, int],
    output_dir: str,
    train_ids: Optional[List[Any]] = None,
    test_ids: Optional[List[Tuple[str, int]]] = None,
    test_steps: Optional[List[int]] = None,
    test_is_validation: bool = False,
    frozen_rule_path: Optional[str] = None,
    data_files_sha256: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Executes a single (setting, model, arm, learner_seed) run, saving NPZ and JSON.

    Enforces active frozen rule validation and records input and rule SHA-256 values.
    """
    if frozen_rule_path is None or not os.path.isfile(frozen_rule_path):
        raise RuntimeError(
            f"FROZEN RULE BARRIER: Refusing execution without a valid, existing "
            f"--frozen-rule file. Provided path: {frozen_rule_path!r}."
        )

    frozen_rule_sha = compute_file_sha256(frozen_rule_path)

    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{setting}_{model}_{arm}_seed{learner_seed}"
    npz_path = os.path.join(output_dir, f"{base_name}.npz")
    json_path = os.path.join(output_dir, f"{base_name}.json")

    start_time = time.time()

    # Prepare features for arm
    X_tr, X_te, X_va, active_cols = prepare_arm_features(
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        noisy_range=noisy_range,
        arm=arm,
        learner_seed=learner_seed
    )

    start_c, end_c = noisy_range
    raw_noisy = X_test[:, start_c:end_c]

    # Model prediction
    if arm == "R":
        preds = raw_noisy.copy()
    elif arm == "Rcal":
        if model.lower() != "ols":
            raise ValueError(f"Arm 'Rcal' is only defined for model 'ols', got model: {model!r}")
        preds = fit_and_predict_ols(X_tr, y_train, X_te)
    elif model.lower() == "ols":
        preds = fit_and_predict_ols(X_tr, y_train, X_te)
    elif model.lower() in ["rf", "random_forest"]:
        preds = fit_and_predict_rf(X_tr, y_train, X_te, learner_seed)
    elif model.lower() == "mlp":
        preds = fit_and_predict_mlp(X_tr, y_train, X_te, X_va, y_val, learner_seed)
    else:
        raise ValueError(f"Unknown model: {model}. Supported: 'ols', 'rf', 'mlp'.")

    wall_time = time.time() - start_time

    # Record identifiers
    if test_ids is None:
        test_ids = [("unknown", i) for i in range(len(X_test))]
    if test_steps is None:
        test_steps = [0] * len(X_test)

    # Feature names hash
    col_str = ",".join(str(c) for c in active_cols)
    feature_names_hash = compute_string_sha256(col_str)

    # Save NPZ
    np.savez_compressed(
        npz_path,
        predictions=preds,
        targets=y_test,
        raw_noisy=raw_noisy,
        test_circuit_files=np.array([item[0] for item in test_ids]),
        test_circuit_indices=np.array([item[1] for item in test_ids], dtype=int),
        test_circuit_ids=np.array([f"{item[0]}:{item[1]}" for item in test_ids]),
        test_steps=np.array(test_steps, dtype=int)
    )

    # Save JSON metadata
    metadata = {
        "setting": setting,
        "model": model,
        "arm": arm,
        "learner_seed": learner_seed,
        "num_train_samples": int(X_tr.shape[0]),
        "num_val_samples": int(X_va.shape[0]),
        "num_test_samples": int(X_te.shape[0]),
        "feature_dim": int(X_tr.shape[1]),
        "feature_column_count": int(len(active_cols)),
        "feature_columns": [int(c) for c in active_cols],
        "feature_names_hash": feature_names_hash,
        "test_is_validation": bool(test_is_validation),
        "frozen_rule_path": os.path.abspath(frozen_rule_path),
        "frozen_rule_sha256": frozen_rule_sha,
        "data_files_sha256": data_files_sha256 or {},
        "package_versions": {
            "qiskit": getattr(qiskit, "__version__", "unknown"),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__
        },
        "wall_time_seconds": float(wall_time),
        "npz_file": os.path.basename(npz_path)
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
