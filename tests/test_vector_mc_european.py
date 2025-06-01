import tensorflow as tf
import numpy as np
from ml_greeks_pricers.pricers.european import (
    AnalyticalEuropeanOption,
    MCEuropeanOption,
    EuropeanAsset,
    MarketData,
)

def test_vector_monte_carlo_prices():
    tf.keras.backend.set_floatx("float64")
    S = tf.constant([110.0, 120.0], dtype=tf.float64)
    K = tf.constant([90.0, 100.0], dtype=tf.float64)
    T = tf.constant(0.5, dtype=tf.float64)
    r = tf.constant(0.06, dtype=tf.float64)
    q = tf.constant(0.0, dtype=tf.float64)
    sigma = tf.constant(0.212, dtype=tf.float64)

    anal = AnalyticalEuropeanOption(S, K, T, 0.0, r, q, sigma, is_call=False)
    analytical = anal.price().numpy()

    n_paths = 20_000
    n_steps = 50
    dt = float(T.numpy()) / n_steps

    asset = EuropeanAsset(
        S,
        q,
        T=float(T.numpy()),
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=0,
    )
    market = MarketData(r, float(sigma.numpy()))
    mc = MCEuropeanOption(asset, market, K, T, is_call=False)
    price = mc().numpy()

    diff = np.abs(price - analytical) / analytical
    assert np.all(diff < 0.1)
