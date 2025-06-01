import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml_greeks_pricers.nn import MCEuropeanOption, NeuralApproximator, TwinNetwork
from ml_greeks_pricers.pricers.european import MarketData, EuropeanAsset


if __name__ == "__main__":
    tf.keras.backend.set_floatx("float32")

    model_path = Path(__file__).with_suffix(".keras")
    scaler_path = Path(__file__).with_suffix(".pkl")

    market = MarketData(0.0, 0.2)
    # ``EuropeanAsset`` defaults to antithetic sampling.  Use an even number of
    # paths and time steps to avoid assertion errors during initialisation.
    asset = EuropeanAsset(1.0, 0.0, T=1.0, dt=0.5, n_paths=2)
    gen = MCEuropeanOption(market, asset)

    x_tr, y_tr, dy_tr = gen.training_set(4096, seed=0)
    approximator = NeuralApproximator(x_tr, y_tr, dy_tr)
    approximator.prepare(4096, diff=True)
    approximator.train()

    approximator.twin.save(model_path)
    with open(scaler_path, "wb") as fh:
        pickle.dump(approximator.scaler, fh)

    loaded = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"TwinNetwork": TwinNetwork},
    )
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    x_te, _, y_te, dy_te, _ = gen.test_set(n=100)
    x_s = scaler.x_transform(x_te)
    y_s, dy_s = loaded.predict(x_s, verbose=0)
    y_pred, dy_pred = scaler.inverse(y_s, dy_s)

    print("target", y_te[:5].flatten())
    print("pred", y_pred[:5, 0])
    print("delta", dy_pred[:5, 0])
