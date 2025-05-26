import os
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.pricers.european import MarketData
from ml_greeks_pricers.pricers.american import AmericanAsset, MCAmericanOption
from ml_greeks_pricers.volatility.discrete import DupireLocalVol
import pandas as pd

S0, r, q = 110.0, 0.06, 0.0
iv_vol = 0.212
n_paths = 50_000
n_steps = 100
T_max = 2.0
# reuse same dt as examples/european.py (0.5/100)
dt = 0.5 / n_steps

log_path = Path(__file__).with_name("execution.log")

# store CSV files in a dedicated folder for American option results
csv_dir = Path(__file__).with_name('american_surfaces')
csv_dir.mkdir(exist_ok=True)

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

def surface(market, use_cache=False):
    asset = AmericanAsset(
        S0,
        q,
        T=T_max,
        dt=dt,
        n_paths=n_paths,
        antithetic=True,
        seed=0,
    )

    price_rows = []
    delta_rows = []
    vega_rows = []
    for T in mats:
        price_row = []
        delta_row = []
        vega_row = []
        for K in strikes:
            opt = MCAmericanOption(
                asset, market, K, T, is_call=False, use_cache=use_cache
            )
            price_row.append(opt().numpy())
            delta_row.append(opt.delta().numpy())
            vega_row.append(opt.vega().numpy())
        price_rows.append(price_row)
        delta_rows.append(delta_row)
        vega_rows.append(vega_row)

    return (
        pd.DataFrame(price_rows, index=mats, columns=strikes),
        pd.DataFrame(delta_rows, index=mats, columns=strikes),
        pd.DataFrame(vega_rows, index=mats, columns=strikes),
    )

def measure(market, name, use_cache=True):
    warm_asset = AmericanAsset(
        S0, q, T=T_max, dt=dt, n_paths=n_paths, antithetic=True, seed=0
    )
    warm_opt = MCAmericanOption(
        warm_asset, market, strikes[0], mats[-1], is_call=False, use_cache=use_cache
    )
    warm_opt()

    start = time.perf_counter()
    result = surface(market, use_cache)
    elapsed = time.perf_counter() - start
    print(f"{name} ({'cache' if use_cache else 'no cache'}): {elapsed:.2f}s")
    return result, elapsed

def save_surfaces(prefix, prices, deltas, vegas):
    prices.to_csv(csv_dir / f"{prefix}_prices.csv")
    deltas.to_csv(csv_dir / f"{prefix}_deltas.csv")
    vegas.to_csv(csv_dir / f"{prefix}_vegas.csv")

if __name__ == "__main__":
    (prices_flat, deltas_flat, vegas_flat), t_flat = measure(market_flat, "flat", True)
    (prices_dup, deltas_dup, vegas_dup), t_dup = measure(market_dup, "dupire", True)

    print("\nFlat MC Prices:")
    print(prices_flat)
    print("\nDupire MC Prices:")
    print(prices_dup)

    print("\nFlat MC Delta:")
    print(deltas_flat)
    print("\nDupire MC Delta:")
    print(deltas_dup)

    print("\nFlat MC Vega:")
    print(vegas_flat)
    print("\nDupire MC Vega:")
    print(vegas_dup)

    save_surfaces("flat", prices_flat, deltas_flat, vegas_flat)
    save_surfaces("dupire", prices_dup, deltas_dup, vegas_dup)

    with log_path.open("a") as f:
        f.write(
            f"{datetime.now().isoformat()} american_surfaces n_steps={n_steps} n_paths={n_paths} "
            f"flat_time={t_flat:.4f} dupire_time={t_dup:.4f}\n"
        )
