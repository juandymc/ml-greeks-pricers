import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf

from ml_greeks_pricers.volatility.discrete import DupireLocalVol
from ml_greeks_pricers.pricers.european import MarketData
from ml_greeks_pricers.pricers.american import AmericanAsset, MCAmericanOption


tf.keras.backend.set_floatx('float64')

dtype = tf.float64

if __name__ == "__main__":
    S0 = 110.0
    K = 90.0
    r = 0.06
    q = 0.0
    T = 0.5
    n_paths = 200_000
    n_steps = 50
    seed = 42

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

    dt = T / n_steps
    log_path = Path(__file__).with_name("execution.log")

    # ---- constant volatility ----
    market_flat = MarketData(r, 0.212)
    asset_flat = AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_flat = MCAmericanOption(asset_flat, market_flat, K, T, is_call=False)
    mc_flat()  # warm-up
    start = time.perf_counter()
    price_flat = mc_flat()
    delta_flat = mc_flat.delta()
    vega_flat = mc_flat.vega()
    elapsed_flat = time.perf_counter() - start
    tf.print("Flat", price_flat, delta_flat, vega_flat)

    # ---- Dupire local-vol ----
    dupire_lv = DupireLocalVol(strikes, mats, iv, S0, r, q)
    market_dup = MarketData(r, dupire_lv)
    asset_dup = AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_dup = MCAmericanOption(asset_dup, market_dup, K, T, is_call=False)
    mc_dup()  # warm-up
    start = time.perf_counter()
    price_dup = mc_dup()
    delta_dup = mc_dup.delta()
    vega_dup = mc_dup.vega()
    elapsed_dup = time.perf_counter() - start
    tf.print("Dupire", price_dup, delta_dup, vega_dup)

    with log_path.open("a") as f:
        f.write(
            f"{datetime.now().isoformat()} american n_steps={n_steps} n_paths={n_paths} "
            f"flat_time={elapsed_flat:.4f} dupire_time={elapsed_dup:.4f} "
            f"flat_price={float(price_flat):.6f} flat_delta={float(delta_flat):.6f} flat_vega={float(vega_flat):.6f} "
            f"dupire_price={float(price_dup):.6f} dupire_delta={float(delta_dup):.6f} dupire_vega={float(vega_dup):.6f}\n"
        )
