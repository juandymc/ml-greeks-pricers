"""
Entrena la red gemela (TwinNetwork) y guarda:
  • la arquitectura + pesos  → model.keras
  • el escalador             → scaler.pkl
"""

import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf

from ml_greeks_pricers.nn import MCEuropeanOption, NeuralApproximator, TwinNetwork
from ml_greeks_pricers.pricers.european import MarketData, EuropeanAsset

# ---------------------- parámetros de entrenamiento ------------------------
N_TRAIN   = 8_192          # tamaño del set de entrenamiento
SEED      = 42             # fija la reproducibilidad
MODEL_F   = Path("model.keras")
SCALER_F  = Path("scaler.pkl")
# ---------------------------------------------------------------------------

def main():
    tf.keras.backend.set_floatx("float32")

    # Mercado y subyacente
    market = MarketData(0.0,0.2)
    asset  = EuropeanAsset(S0=1.0, q=0.0, T=1.0, dt=0.5, n_paths=2)
    gen    = MCEuropeanOption(market, asset)

    # Datos de entrenamiento
    x_tr, y_tr, dy_tr = gen.training_set(N_TRAIN, seed=SEED)

    # Aproximador (red gemela)
    approximator = NeuralApproximator(x_tr, y_tr, dy_tr)
    approximator.prepare(N_TRAIN, diff=True)
    approximator.train()

    # ----- guardar modelo y escalador -----
    approximator.twin.save(MODEL_F)
    with open(SCALER_F, "wb") as fh:
        pickle.dump(approximator.scaler, fh)

    print(f"Modelo guardado en {MODEL_F}\nEscalador guardado en {SCALER_F}")

if __name__ == "__main__":
    main()
