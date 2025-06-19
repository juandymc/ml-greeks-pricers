import pandas as pd
import tensorflow as tf
from ml_greeks_pricers.pricers.tf.european import (
    AnalyticalEuropeanOption,
    EuropeanAsset,
    MCEuropeanOption,
    MarketData,
)
from ml_greeks_pricers.volatility.discrete import DupireLocalVol

S0, r, q = 110., 0.06, 0.
iv_vol = 0.212
strikes = [60, 70, 80, 90, 100, 110, 120, 130, 140]
mats = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
T_max = 2.0
n_paths = 1_000
n_steps = 20
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

def price_surface(market):
    asset = EuropeanAsset(
        S0,
        q,
        T=T_max,
        dt=T_max / n_steps,
        n_paths=n_paths,
        use_scan=True,
        seed=0,
    )
    temp = {}
    for T in sorted(mats, reverse=True):
        row = []
        for K in strikes:
            opt = MCEuropeanOption(asset, market, K, T, is_call=False, use_cache=True)
            row.append(opt().numpy())
        temp[T] = row
    return pd.DataFrame([temp[T] for T in mats], index=mats, columns=strikes)

def analytical_surface():
    rows = []
    for i, T in enumerate(mats):
        row = []
        for j, K in enumerate(strikes):
            sigma = iv[i][j]
            ana = AnalyticalEuropeanOption(S0, K, T, 0.0, r, q, sigma, is_call=False)
            row.append(ana().numpy())
        rows.append(row)
    return pd.DataFrame(rows, index=mats, columns=strikes)

def test_surface_diff_with_cache():
    tf.keras.backend.set_floatx("float64")
    prices_ana = analytical_surface()
    market_flat = MarketData(r, iv_vol)
    dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
    market_dup = MarketData(r, dup)

    prices_flat = price_surface(market_flat)
    prices_dup = price_surface(market_dup)

    diff_flat = 100.0 * (prices_flat - prices_ana).abs() / prices_ana
    diff_dup = 100.0 * (prices_dup - prices_ana).abs() / prices_ana

    assert diff_flat.values.max() < 150.0
    assert diff_dup.values.max() < 150.0
