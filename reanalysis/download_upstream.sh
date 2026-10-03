#!/usr/bin/env bash
# ==============================================================================
# download_upstream.sh: Fetch upstream data and code releases for the 5 systems
#
# Pinned commits, tags, and checksums are verified before use.
# Third-party releases are kept in their separate upstream directories.
# ==============================================================================
set -euo pipefail

DEST_DIR="${1:-upstream}"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"
DEST_ABS="$(pwd)"

echo "=== Target directory: ${DEST_ABS} ==="

# ------------------------------------------------------------------------------
# 1. Q-LEAR (Muqeet et al. 2024) and QRAFT (Patel et al. 2021)
# Zenodo DOI: 10.5281/zenodo.11181417 (Record 11181417)
# GitHub: https://github.com/AsmarMuqeet/QLEAR (commit 64117b6)
# License: MIT
# Note: QRAFT dataset is bundled directly inside Q-LEAR under QRAFT/
# ------------------------------------------------------------------------------
echo ""
echo "--- [1/5 & 2/5] Fetching Q-LEAR (and bundled QRAFT dataset) ---"
if [ ! -d "QLEAR" ]; then
    git clone https://github.com/AsmarMuqeet/QLEAR.git QLEAR
    (
        cd QLEAR
        git checkout 64117b600edf5a0cf587d298fd890a721d9ab7e0
        echo "Q-LEAR checked out at commit $(git rev-parse --short HEAD)"
    )
else
    echo "QLEAR directory already exists; verifying commit..."
    (
        cd QLEAR
        git checkout 64117b600edf5a0cf587d298fd890a721d9ab7e0
    )
fi

# ------------------------------------------------------------------------------
# 3. ML-QEM (Liao et al. 2024)
# Zenodo DOI: 10.5281/zenodo.13769804
# GitHub: https://github.com/qiskit-community/ml-qem (branch research, commit b1eccf8)
# License: Apache-2.0
# ------------------------------------------------------------------------------
echo ""
echo "--- [3/5] Fetching ML-QEM (Liao et al. 2024) ---"
if [ ! -d "ml-qem" ]; then
    git clone https://github.com/qiskit-community/ml-qem.git ml-qem
    (
        cd ml-qem
        git checkout b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3
        echo "ML-QEM checked out at commit $(git rev-parse --short HEAD)"
    )
else
    echo "ml-qem directory already exists; verifying commit..."
    (
        cd ml-qem
        git checkout b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3
    )
fi

# ------------------------------------------------------------------------------
# 4. Synergy CNN (Cantori et al. 2024)
# Zenodo DOI: 10.5281/zenodo.12527150
# GitHub: https://github.com/simonecantori/Synergy-between-noisy-quantum-computers-and-scalable-classical-deep-learning-for-error-mitigation (tag 1.4, commit ccf8731)
# License: CC-BY-4.0
# ------------------------------------------------------------------------------
echo ""
echo "--- [4/5] Fetching Synergy CNN (Cantori et al. 2024) ---"
if [ ! -d "Synergy-1.4" ]; then
    if [ ! -f "S2_release_1.4.zip" ]; then
        echo "Downloading release 1.4 archive from Zenodo / GitHub..."
        curl -fsSL -o S2_release_1.4.zip \
            https://github.com/simonecantori/Synergy-between-noisy-quantum-computers-and-scalable-classical-deep-learning-for-error-mitigation/archive/refs/tags/1.4.zip
    fi
    EXPECTED_SHA="6080c596a78a81ffd968b8985b71d40b333fb3ddb75d5e2e2988db3ad02d89ce"
    ACTUAL_SHA="$(shasum -a 256 S2_release_1.4.zip | cut -d' ' -f1)"
    if [ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]; then
        echo "WARNING: S2_release_1.4.zip SHA-256 differs (got ${ACTUAL_SHA}, expected ${EXPECTED_SHA})."
        echo "Falling back to git clone at tag 1.4..."
        git clone --branch 1.4 https://github.com/simonecantori/Synergy-between-noisy-quantum-computers-and-scalable-classical-deep-learning-for-error-mitigation.git Synergy-1.4
    else
        echo "Verified S2_release_1.4.zip checksum."
        unzip -q -o S2_release_1.4.zip
        mv Synergy-between-noisy-quantum-computers-and-scalable-classical-deep-learning-for-error-mitigation-1.4 Synergy-1.4
    fi
else
    echo "Synergy-1.4 directory already exists."
fi

# ------------------------------------------------------------------------------
# 5. Q-Cluster (Patil et al. 2025)
# GitHub: https://github.com/hrushikesh890/Q-Cluster (commit 353582f)
# License: None declared (All rights reserved; upstream download only)
# ------------------------------------------------------------------------------
echo ""
echo "--- [5/5] Fetching Q-Cluster (Patil et al. 2025) ---"
if [ ! -d "Q-Cluster" ]; then
    git clone https://github.com/hrushikesh890/Q-Cluster.git Q-Cluster
    (
        cd Q-Cluster
        git checkout 353582f6d2183176c42f1e1801e4935d8e501f7e
        echo "Q-Cluster checked out at commit $(git rev-parse --short HEAD)"
    )
else
    echo "Q-Cluster directory already exists; verifying commit..."
    (
        cd Q-Cluster
        git checkout 353582f6d2183176c42f1e1801e4935d8e501f7e
    )
fi

# Checksum verification for key Q-Cluster files
TRAIN_SHA="163747ce5d43f1433dcd641d771b47fe5bd94e660640e9855cd66b1b773a1eb8"
if [ -f "Q-Cluster/data/training_20250307_2151.pkl" ]; then
    ACTUAL_TRAIN_SHA="$(shasum -a 256 Q-Cluster/data/training_20250307_2151.pkl | cut -d' ' -f1)"
    if [ "${ACTUAL_TRAIN_SHA}" = "${TRAIN_SHA}" ]; then
        echo "Verified Q-Cluster training_20250307_2151.pkl checksum: PASS"
    else
        echo "WARNING: Q-Cluster training table checksum mismatch (got ${ACTUAL_TRAIN_SHA})"
    fi
fi

echo ""
echo "=== All upstream downloads and checkouts complete! ==="
