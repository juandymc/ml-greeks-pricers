import numpy as np
import tensorflow as tf
from ml_greeks_pricers.pricers.tf import american as tf_amer
from ml_greeks_pricers.pricers.tf import european as tf_eur
from ml_greeks_pricers.pricers.np import american as np_amer
from ml_greeks_pricers.pricers.np import european as np_eur

tf.keras.backend.set_floatx('float64')

def compare_european():
    S0, K, r, q, T = 110.0, 90.0, 0.06, 0.0, 0.5
    sigma = 0.212
    n_paths, n_steps = 50_000, 100
    dt = T / n_steps

    # TensorFlow pricer
    market_tf = tf_eur.MarketData(r, sigma)
    asset_tf = tf_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, use_scan=True, seed=0)
    opt_tf = tf_eur.MCEuropeanOption(asset_tf, market_tf, K, T, is_call=False)
    price_tf = opt_tf().numpy()

    # NumPy pricer
    market_np = np_eur.MarketData(r, sigma)
    asset_np = np_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, seed=0)
    opt_np = np_eur.MCEuropeanOption(asset_np, market_np, K, T, is_call=False)
    price_np = opt_np()

    print("European TF:", price_tf)
    print("European NP:", price_np)
    print("Close:", np.isclose(price_tf, price_np, rtol=5e-2))


def compare_american():
    S0, K, r, q, T = 110.0, 90.0, 0.06, 0.0, 0.5
    sigma = 0.212
    n_paths, n_steps = 20_000, 50
    dt = T / n_steps

    market_tf = tf_eur.MarketData(r, sigma)
    asset_tf = tf_amer.AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=0)
    opt_tf = tf_amer.MCAmericanOption(asset_tf, market_tf, K, T, is_call=False)
    price_tf = opt_tf().numpy()

    market_np = np_eur.MarketData(r, sigma)
    asset_np = np_amer.AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, seed=0)
    opt_np = np_amer.MCAmericanOption(asset_np, market_np, K, T, is_call=False)
    price_np = opt_np()

    print("American TF:", price_tf)
    print("American NP:", price_np)
    print("Close:", np.isclose(price_tf, price_np, rtol=5e-2))


if __name__ == "__main__":
    compare_european()
    compare_american()
