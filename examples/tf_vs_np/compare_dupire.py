import numpy as np
import tensorflow as tf

from ml_greeks_pricers.pricers.tf import european as tf_eur
from ml_greeks_pricers.pricers.np import european as np_eur
from ml_greeks_pricers.volatility.discrete import DupireLocalVol as TfDupire
from ml_greeks_pricers.volatility.np_discrete import DupireLocalVol as NpDupire


tf.keras.backend.set_floatx("float64")

def compare_dupire():
    S0, K, r, q, T = 110.0, 90.0, 0.06, 0.0, 0.5
    n_paths, n_steps = 50_000, 100
    dt = T / n_steps

    strikes = [60, 70, 80, 90, 100, 110, 120, 130, 140]
    mats = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    iv = [
        [0.248, 0.233, 0.220, 0.209, 0.200, 0.193, 0.188, 0.185, 0.184],
        [0.251, 0.236, 0.223, 0.212, 0.203, 0.196, 0.191, 0.188, 0.187],
        [0.254, 0.239, 0.226, 0.215, 0.206, 0.199, 0.194, 0.191, 0.190],
        [0.257, 0.242, 0.229, 0.218, 0.209, 0.202, 0.197, 0.194, 0.193],
        [0.260, 0.245, 0.232, 0.221, 0.212, 0.205, 0.200, 0.197, 0.196],
        [0.263, 0.248, 0.235, 0.224, 0.215, 0.208, 0.203, 0.200, 0.199],
        [0.266, 0.251, 0.238, 0.227, 0.218, 0.211, 0.206, 0.203, 0.202],
        [0.269, 0.254, 0.241, 0.230, 0.221, 0.214, 0.209, 0.206, 0.205],
    ]

    dup_tf = TfDupire(strikes, mats, iv, S0, r, q)
    market_tf = tf_eur.MarketData(r, dup_tf)
    asset_tf = tf_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, use_scan=True, seed=0)
    opt_tf = tf_eur.MCEuropeanOption(asset_tf, market_tf, K, T, is_call=False)
    price_tf = opt_tf().numpy()

    dup_np = NpDupire(strikes, mats, iv, S0, r, q)
    market_np = np_eur.MarketData(r, dup_np)
    asset_np = np_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, seed=0)
    opt_np = np_eur.MCEuropeanOption(asset_np, market_np, K, T, is_call=False)
    price_np = opt_np()

    print("Dupire TF:", price_tf)
    print("Dupire NP:", price_np)
    print("Close:", np.isclose(price_tf, price_np, rtol=5e-2))


if __name__ == "__main__":
    compare_dupire()
