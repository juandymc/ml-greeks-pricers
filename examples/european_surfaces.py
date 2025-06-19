import os
import time
from pathlib import Path
import tensorflow as tf

# suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.pricers.tf.european import (
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
n_paths = 50_000#10#
n_steps = 100#10#
T_max = 2.0
# reuse same dt as examples/european.py (0.5/100)
dt = 0.5 / n_steps

# folder where CSV files will be stored
csv_dir = Path(__file__).with_name('outputs') / 'european_surfaces'
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

# helper to price and compute greeks for the full surface

def surface(market, use_cache):
    """Return price, delta and vega surfaces for ``market``.

    Each surface is a ``pandas.DataFrame`` with maturities as index and
    strikes as columns.
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

    price_rows = []
    delta_rows = []
    vega_rows = []
    for T in mats:
        price_row = []
        delta_row = []
        vega_row = []
        for K in strikes:
            opt = MCEuropeanOption(
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


def measure(market, name, use_cache):
    """Time the computation of the full surface using ``market``."""
    warm_asset = EuropeanAsset(
        S0, q, T=T_max, dt=dt, n_paths=n_paths, use_scan=True, seed=0
    )
    warm_opt = MCEuropeanOption(
        warm_asset, market, strikes[0], mats[-1], is_call=False, use_cache=use_cache
    )
    warm_opt()

    start = time.perf_counter()
    result = surface(market, use_cache)
    elapsed = time.perf_counter() - start
    print(f"{name} ({'cache' if use_cache else 'no cache'}): {elapsed:.2f}s")
    return result


def analytical_surface():
    """Return analytical price, delta and vega surfaces."""
    price_rows = []
    delta_rows = []
    vega_rows = []
    for i, T in enumerate(mats):
        price_row = []
        delta_row = []
        vega_row = []
        for j, K in enumerate(strikes):
            sigma = iv[i][j]
            ana = AnalyticalEuropeanOption(
                S0, K, T, 0.0, r, q, sigma, is_call=False
            )
            price_row.append(ana().numpy())
            delta_row.append(ana.delta().numpy())
            vega_row.append(ana.vega().numpy())
        price_rows.append(price_row)
        delta_rows.append(delta_row)
        vega_rows.append(vega_row)
    return (
        pd.DataFrame(price_rows, index=mats, columns=strikes),
        pd.DataFrame(delta_rows, index=mats, columns=strikes),
        pd.DataFrame(vega_rows, index=mats, columns=strikes),
    )


def save_surfaces(prefix, prices, deltas, vegas):
    prices.to_csv(csv_dir / f"{prefix}_prices.csv")
    deltas.to_csv(csv_dir / f"{prefix}_deltas.csv")
    vegas.to_csv(csv_dir / f"{prefix}_vegas.csv")


if __name__ == '__main__':
    prices_ana, deltas_ana, vegas_ana = analytical_surface()
    prices_flat, deltas_flat, vegas_flat = measure(market_flat, 'flat', True)
    prices_dup, deltas_dup, vegas_dup = measure(market_dup, 'dupire', False)

    diff_flat_p = 100.0 * (prices_flat - prices_ana) / prices_ana
    diff_dup_p = 100.0 * (prices_dup - prices_ana) / prices_ana
    diff_flat_d = 100.0 * (deltas_flat - deltas_ana) / deltas_ana
    diff_dup_d = 100.0 * (deltas_dup - deltas_ana) / deltas_ana
    diff_flat_v = 100.0 * (vegas_flat - vegas_ana) / vegas_ana
    diff_dup_v = 100.0 * (vegas_dup - vegas_ana) / vegas_ana

    print('\nAnalytical Prices:')
    print(prices_ana)
    print('\nFlat MC Prices:')
    print(prices_flat)
    print('\nDupire MC Prices:')
    print(prices_dup)
    print('\nFlat % diff vs analytical (price):')
    print(diff_flat_p)
    print('\nDupire % diff vs analytical (price):')
    print(diff_dup_p)

    print('\nAnalytical Delta:')
    print(deltas_ana)
    print('\nFlat MC Delta:')
    print(deltas_flat)
    print('\nDupire MC Delta:')
    print(deltas_dup)
    print('\nFlat % diff vs analytical (delta):')
    print(diff_flat_d)
    print('\nDupire % diff vs analytical (delta):')
    print(diff_dup_d)

    print('\nAnalytical Vega:')
    print(vegas_ana)
    print('\nFlat MC Vega:')
    print(vegas_flat)
    print('\nDupire MC Vega:')
    print(vegas_dup)
    print('\nFlat % diff vs analytical (vega):')
    print(diff_flat_v)
    print('\nDupire % diff vs analytical (vega):')
    print(diff_dup_v)

    # save surfaces to CSV files
    save_surfaces('analytical', prices_ana, deltas_ana, vegas_ana)
    save_surfaces('flat', prices_flat, deltas_flat, vegas_flat)
    save_surfaces('dupire', prices_dup, deltas_dup, vegas_dup)
    diff_flat_p.to_csv(csv_dir / 'diff_flat_price.csv')
    diff_dup_p.to_csv(csv_dir / 'diff_dupire_price.csv')
    diff_flat_d.to_csv(csv_dir / 'diff_flat_delta.csv')
    diff_dup_d.to_csv(csv_dir / 'diff_dupire_delta.csv')
    diff_flat_v.to_csv(csv_dir / 'diff_flat_vega.csv')
    diff_dup_v.to_csv(csv_dir / 'diff_dupire_vega.csv')
