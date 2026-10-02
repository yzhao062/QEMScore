"""
S2 phase 2, step 2: COMPUTE-BUDGET PROBE.

Times the released architecture on SYNTHETIC arrays of the released shapes so
the fold count and the epoch budget can be fixed before any arm touches the
released rows. Nothing here reads mz_exact, theta, qubits or z_noisy.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("logical CPUs:", os.cpu_count())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REL = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "upstream", "s2_synergy"))
REL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REL
sys.path.insert(0, REL)
# The released cnn_2D.py calls K.sum / K.square, removed in Keras 3. That r2 is a
# REPORTING metric only: it is not the loss and not the selection rule, so a
# Keras-3 equivalent is substituted and the trained model is unchanged.
import keras.backend as _K
from keras import ops as _ops
for _n in ("sum", "square", "mean"):
    if not hasattr(_K, _n):
        setattr(_K, _n, getattr(_ops, _n))
from cnn_2D import CNN  # the released file, verbatim

for nch in (2, 3):
    m = CNN(synergy=(nch == 3))
    X = np.random.rand(72, 16, 20, nch).astype("float32")
    y = np.random.rand(72).astype("float32")
    m.fit(X, y, epochs=1, batch_size=8, verbose=0)          # warm up / compile
    t0 = time.time()
    m.fit(X, y, epochs=10, batch_size=8, verbose=0)
    dt = (time.time() - t0) / 10.0
    print(f"  config-B 2D CNN, {nch} channels: {dt*1000:.0f} ms/epoch on 72 rows "
          f"-> {dt*400:.1f} s for a 400-epoch fold")
    print(f"     params = {m.count_params():,}")
    del m
    tf.keras.backend.clear_session()

# 1D analogue for configuration A (panel a); the release ships only the 2D file.
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from cnn_2D import r2 as r2_metric


def CNN1D(synergy):
    model = Sequential()
    ch = 3 if synergy else 2
    model.add(Conv1D(filters=128, activation="relu", kernel_size=3, strides=1,
                     padding="same", input_shape=(None, ch)))
    model.add(Conv1D(filters=128, activation="relu", kernel_size=3, strides=1, padding="same"))
    model.add(Conv1D(filters=128, activation="relu", kernel_size=3, strides=1, padding="same"))
    model.add(GlobalAveragePooling1D())
    for _ in range(4):
        model.add(Dense(512, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae", r2_metric])
    return model


for nch in (2, 3):
    m = CNN1D(synergy=(nch == 3))
    X = np.random.rand(72, 16, nch).astype("float32")
    y = np.random.rand(72).astype("float32")
    m.fit(X, y, epochs=1, batch_size=8, verbose=0)
    t0 = time.time()
    m.fit(X, y, epochs=10, batch_size=8, verbose=0)
    dt = (time.time() - t0) / 10.0
    print(f"  config-A 1D CNN, {nch} channels: {dt*1000:.0f} ms/epoch on 72 rows "
          f"-> {dt*400:.1f} s for a 400-epoch fold")
    print(f"     params = {m.count_params():,}")
    del m
    tf.keras.backend.clear_session()
