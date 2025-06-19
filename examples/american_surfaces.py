import os
import time
from pathlib import Path
import tensorflow as tf

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.pricers.tf.european import MarketData
from ml_greeks_pricers.pricers.tf.american import AmericanAsset, MCAmericanOption
from ml_greeks_pricers.volatility.discrete import DupireLocalVol
import pandas as pd

S0, r, q = 110.0, 0.06, 0.0
iv_vol = 0.212
n_paths = 50_000
n_steps = 100
T_max = 2.0
# reuse same dt as examples/european.py (0.5/100)
dt = 0.5 / n_steps

# store CSV files in a dedicated folder for American option results
csv_dir = Path(__file__).with_name('outputs') / 'american_surfaces'
csv_dir.mkdir(parents=True, exist_ok=True)

inputs_dir = Path(__file__).with_name('inputs')
iv_df = pd.read_csv(inputs_dir / 'implied_vol_surface.csv', index_col=0)

strikes = [float(c) for c in iv_df.columns]
mats = [float(i) for i in iv_df.index]
iv = iv_df.values.tolist()


market_flat = MarketData(r, iv_vol)
dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
dup_df = pd.DataFrame(dup().numpy(), index=mats, columns=strikes)
dup_df.to_csv(csv_dir / 'dupire_local_vol.csv')
market_dup = MarketData(r, dup)

def surface(market, use_cache=False):
    """Return price, delta and vega surfaces for ``market``."""
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
    """Time the computation of the full surface using ``market``."""
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
    return result

def save_surfaces(prefix, prices, deltas, vegas):
    prices.to_csv(csv_dir / f"{prefix}_prices.csv")
    deltas.to_csv(csv_dir / f"{prefix}_deltas.csv")
    vegas.to_csv(csv_dir / f"{prefix}_vegas.csv")

if __name__ == "__main__":
    prices_flat, deltas_flat, vegas_flat = measure(market_flat, "flat", True)
    prices_dup, deltas_dup, vegas_dup = measure(market_dup, "dupire", True)

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
