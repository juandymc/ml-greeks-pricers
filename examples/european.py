import os
import warnings
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.pricers.european import (
    AnalyticalEuropeanOption,
    MCEuropeanOption,
    MarketData,
    EuropeanAsset,
)
from ml_greeks_pricers.volatility.discrete import DupireLocalVol

if __name__ == '__main__':
    n_paths = 50_000
    n_steps = 100
    S0, K, T, r, q = 110., 90., 0.5, 0.06, 0.
    dt = T / n_steps
    log_path = Path(__file__).with_name('execution.log')

    iv_vol = 0.212
    analEur = AnalyticalEuropeanOption(S0, K, T, 0, r, q, iv_vol, is_call=False)
    analytical_price = analEur().numpy()
    tf.print('analytical', analytical_price, analEur.delta(), analEur.vega())

    # ---- constant volatility ----
    market_flat = MarketData(r, iv_vol)
    asset_flat = EuropeanAsset(
        S0,
        q,
        T=T,
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=0,
    )
    mc_flat = MCEuropeanOption(asset_flat, market_flat, K, T, is_call=False)
    mc_flat()  # warm-up
    start = time.perf_counter()
    flat_price = mc_flat().numpy()
    flat_delta = mc_flat.delta().numpy()
    flat_vega = mc_flat.vega().numpy()
    elapsed_flat = time.perf_counter() - start
    tf.print('flat', flat_price, flat_delta, flat_vega)

    # ---- Dupire local-vol --------------------------------------------
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
    dup = DupireLocalVol(strikes, mats, iv, S0, r, q)

    market_dup = MarketData(r, dup)
    asset_dup = EuropeanAsset(
        S0,
        q,
        T=T,
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=0,
    )
    mc_loc = MCEuropeanOption(asset_dup, market_dup, K, T, is_call=False)
    mc_loc()  # warm-up
    start = time.perf_counter()
    dupire_price = mc_loc().numpy()
    dupire_delta = mc_loc.delta().numpy()
    dupire_vega = mc_loc.vega().numpy()
    elapsed_dup = time.perf_counter() - start
    tf.print('dupire', dupire_price, dupire_delta, dupire_vega)

    def warn_if_far(name, price):
        diff = abs(price - analytical_price) / analytical_price
        if diff > 0.025:
            warnings.warn(
                f"{name} price differs from analytical by {diff:.2%}",
                RuntimeWarning,
            )

    warn_if_far('flat', flat_price)
    warn_if_far('dupire', dupire_price)

    with log_path.open('a') as f:
        f.write(
            f"{datetime.now().isoformat()} european n_steps={n_steps} n_paths={n_paths} "
            f"flat_time={elapsed_flat:.4f} dupire_time={elapsed_dup:.4f} "
            f"flat_price={flat_price:.6f} flat_delta={flat_delta:.6f} flat_vega={flat_vega:.6f} "
            f"dupire_price={dupire_price:.6f} dupire_delta={dupire_delta:.6f} dupire_vega={dupire_vega:.6f}\n"
        )
