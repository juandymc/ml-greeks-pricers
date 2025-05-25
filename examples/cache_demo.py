import os
import time
import tensorflow as tf

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.pricers.european import (
    MarketData,
    MCEuropeanOption,
    EuropeanAsset,
    AnalyticalEuropeanOption,
)
import pandas as pd
from ml_greeks_pricers.volatility.discrete import DupireLocalVol

# parameters copied from examples/european.py
S0, r, q = 110., 0.06, 0.
iv_vol = 0.212
n_paths = 50_000
n_steps = 100
T_max = 2.0
# reuse same dt as examples/european.py (0.5/100)
dt = 0.5 / n_steps

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

market_flat = MarketData(r, iv_vol)
dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
market_dup = MarketData(r, dup)

# helper to price the full surface

def price_surface(market, use_cache):
    """Price all strike/maturity pairs for ``market``.

    Returns a ``pandas.DataFrame`` with maturities as index and strikes as
    columns.
    """
    asset = EuropeanAsset(
        S0,
        q,
        T=T_max,
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=0,
    )

    rows = []
    for T in mats:
        row = []
        for K in strikes:
            opt = MCEuropeanOption(asset, market, K, T, is_call=False, use_cache=use_cache)
            row.append(opt().numpy())
        rows.append(row)

    return pd.DataFrame(rows, index=mats, columns=strikes)


def measure(market, name, use_cache):
    """Time the pricing of the full surface using ``market``."""
    warm_asset = EuropeanAsset(S0, q, T=T_max, dt=dt, n_paths=n_paths, use_scan=True, seed=0)
    warm_opt = MCEuropeanOption(warm_asset, market, strikes[0], mats[-1], is_call=False, use_cache=use_cache)
    warm_opt()

    start = time.perf_counter()
    df = price_surface(market, use_cache)
    elapsed = time.perf_counter() - start
    print(f"{name} ({'cache' if use_cache else 'no cache'}): {elapsed:.2f}s")
    return df


def analytical_surface():
    """Return analytical prices for every strike/maturity pair."""
    rows = []
    for i, T in enumerate(mats):
        row = []
        for j, K in enumerate(strikes):
            sigma = iv[i][j]
            ana = AnalyticalEuropeanOption(S0, K, T, 0.0, r, q, sigma, is_call=False)
            row.append(ana().numpy())
        rows.append(row)
    return pd.DataFrame(rows, index=mats, columns=strikes)


if __name__ == '__main__':
    prices_ana = analytical_surface()
    prices_flat = measure(market_flat, 'flat', True)
    prices_dup = measure(market_dup, 'dupire', True)

    diff_flat = 100.0 * (prices_flat - prices_ana) / prices_ana
    diff_dup = 100.0 * (prices_dup - prices_ana) / prices_ana

    print('\nAnalytical Prices:')
    print(prices_ana)
    print('\nFlat MC Prices:')
    print(prices_flat)
    print('\nDupire MC Prices:')
    print(prices_dup)
    print('\nFlat % diff vs analytical:')
    print(diff_flat)
    print('\nDupire % diff vs analytical:')
    print(diff_dup)
