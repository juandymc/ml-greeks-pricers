"""
Carga el modelo entrenado + el escalador y genera los plots para
una nueva malla de spots.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ml_greeks_pricers.nn import TwinNetwork, MCEuropeanOption
from ml_greeks_pricers.pricers.european import (
    MarketData,
    EuropeanAsset,
    AnalyticalEuropeanOption,
)

# ------- ficheros generados en la fase de entrenamiento --------------------
MODEL_F  = Path("model.keras")
SCALER_F = Path("scaler.pkl")
# ---------------------------------------------------------------------------

# ------------ spots sobre los que quieres evaluar el modelo ---------------
S_MIN, S_MAX, N_SPOTS = 0.5, 1.5, 200      # por ejemplo de 50 % a 150 %
# ---------------------------------------------------------------------------


def plot(title, x, y_pred, y_target, y_ana, ylabel):
    plt.figure(figsize=(7, 4))
    plt.plot(x * 100, y_pred * 100, "co", ms=3, mfc="w", label="predicción NN")
    plt.plot(x * 100, y_target * 100, "r.", ms=1.5, label="target MC")
    plt.plot(x * 100, y_ana * 100, "g--", lw=1, label="analítico")
    plt.xlabel("spot (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # ---------- carga del modelo y del escalador ----------
    model = tf.keras.models.load_model(
        MODEL_F,
        compile=False,
        custom_objects={"TwinNetwork": TwinNetwork},
    )
    with open(SCALER_F, "rb") as fh:
        scaler = pickle.load(fh)

    # ---------- nueva malla de spots ----------
    spots = np.linspace(S_MIN, S_MAX, N_SPOTS, dtype=np.float32).reshape(-1, 1)

    # ---------- transforma, predice e invierte el escalado ----------
    x_s          = scaler.x_transform(spots)
    y_s, dy_s    = model.predict(x_s, verbose=0)
    value, delta = scaler.inverse(y_s, dy_s)

    # ---------- precios y griegas analíticas (Black-Scholes) ----------
    market = MarketData(0.0, 0.2)
    asset  = EuropeanAsset(S0=1.0, q=0.0, T=1.0, dt=0.5, n_paths=2)

    # Creamos un generador solo para leer el strike coherente con el entrenamiento
    gen = MCEuropeanOption(market, asset)
    K   = gen.K           # strike usado al entrenar
    T   = asset.T

    ana = AnalyticalEuropeanOption(
        tf.constant(spots, dtype=tf.float64),
        K,
        T,
        0.0,
        market.r,
        asset.q,
        market._flat_sigma,
        is_call=True,
    )
    v_ana = ana.price().numpy().reshape(-1, 1)
    d_ana = ana.delta().numpy().reshape(-1, 1)

    # ---------- dibuja ----------
    plot("Precio Black-Scholes", spots, value[:, 0], value[:, 0], v_ana[:, 0], "value")
    plot("Delta Black-Scholes",  spots, delta[:, 0], delta[:, 0], d_ana[:, 0], "delta")


if __name__ == "__main__":
    main()
