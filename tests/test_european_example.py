import warnings
import tensorflow as tf
from ml_greeks_pricers.pricers.european import (
    AnalyticalEuropeanOption,
    MCEuropeanOption,
)
from ml_greeks_pricers.volatility.discrete import DupireLocalVol

# parameters copied from examples/european.py
S0, K, T, r, q = 110., 90., 0.5, 0.06, 0.
iv_vol = 0.212
n_paths = 50_000
n_steps = 100

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

# tolerance thresholds for Monte Carlo vs analytical result
WARN_TOL = 0.025  # 2.5%
FAIL_TOL = 0.05   # 5%

def test_monte_carlo_prices_close_to_analytical():
    tf.keras.backend.set_floatx("float64")

    anal = AnalyticalEuropeanOption(S0, K, T, 0, r, q, iv_vol, is_call=False)
    analytical_price = anal().numpy()

    mc_flat = MCEuropeanOption(
        S0, K, T, r, q, iv_vol,
        n_paths=n_paths, n_steps=n_steps,
        use_scan=True, seed=0,
    )
    flat_price = mc_flat().numpy()

    dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
    mc_loc = MCEuropeanOption(
        S0, K, T, r, q, dup,
        n_paths=n_paths, n_steps=n_steps,
        use_scan=True, seed=0,
    )
    dupire_price = mc_loc().numpy()

    def check(label, price):
        diff = abs(price - analytical_price) / analytical_price
        if diff > WARN_TOL:
            warnings.warn(
                f"{label} price differs from analytical by {diff:.2%}",
                RuntimeWarning,
            )
        return diff

    flat_diff = check("flat", flat_price)
    dupire_diff = check("dupire", dupire_price)

    assert flat_diff < FAIL_TOL
    assert dupire_diff < FAIL_TOL
