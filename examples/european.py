import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.keras.backend.clear_session()  


import os, warnings, re, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # suprime avisos de backend C++
tf.get_logger().setLevel('ERROR')          # suprime avisos Python de TF


from ml_greeks_pricers.pricers.european import (
    AnalyticalEuropeanOption,
    MCEuropeanOption,
    EuropeanAsset,
    MarketData,
)
from ml_greeks_pricers.volatility.discrete import DupireLocalVol


if __name__ == '__main__':
    n_paths = 50_000
    n_steps = 100
    S0,K,T,r,q = 110.,90.,0.5,0.06,0.
    iv_vol = 0.212
    analEur = AnalyticalEuropeanOption(S0, K, T, 0, r, q, iv_vol, is_call=False)
    analytical_price = analEur().numpy()
    tf.print('analytical', analytical_price, analEur.delta(), analEur.vega())
    # ---- volatilidad constante ---------------------------------------
    market_flat = MarketData(r, iv_vol)
    asset_flat = EuropeanAsset(
        S0,
        q,
        T,
        n_paths=n_paths,
        n_steps=n_steps,
        use_scan=True,
        seed=0,
    )
    mc_flat = MCEuropeanOption(asset_flat, market_flat, K, T, is_call=False)
    flat_price = mc_flat().numpy()
    tf.print('flat', flat_price, mc_flat.delta(), mc_flat.vega())

    # ---- Dupire local-vol --------------------------------------------
    strikes = [60, 70, 80, 90, 100, 110, 120, 130, 140]
    mats = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

    iv = [
        #  sigma_IV(T , K )
        [0.248, 0.233, 0.220, 0.209, 0.200, 0.193, 0.188, 0.185, 0.184],  # T=0.25
        [0.251, 0.236, 0.223, 0.212, 0.203, 0.196, 0.191, 0.188, 0.187],  # 0.50
        [0.254, 0.239, 0.226, 0.215, 0.206, 0.199, 0.194, 0.191, 0.190],  # 0.75
        [0.257, 0.242, 0.229, 0.218, 0.209, 0.202, 0.197, 0.194, 0.193],  # 1.00
        [0.260, 0.245, 0.232, 0.221, 0.212, 0.205, 0.200, 0.197, 0.196],  # 1.25
        [0.263, 0.248, 0.235, 0.224, 0.215, 0.208, 0.203, 0.200, 0.199],  # 1.50
        [0.266, 0.251, 0.238, 0.227, 0.218, 0.211, 0.206, 0.203, 0.202],  # 1.75
        [0.269, 0.254, 0.241, 0.230, 0.221, 0.214, 0.209, 0.206, 0.205],  # 2.00
    ]
    dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
    market_dup = MarketData(r, dup)
    asset_dup = EuropeanAsset(
        S0,
        q,
        T,
        n_paths=n_paths,
        n_steps=n_steps,
        use_scan=True,
        seed=0,
    )
    mc_loc = MCEuropeanOption(asset_dup, market_dup, K, T, is_call=False)
    dupire_price = mc_loc().numpy()
    tf.print('dupire', dupire_price, mc_loc.delta(), mc_loc.vega())

    def warn_if_far(name, price):
        diff = abs(price - analytical_price) / analytical_price
        if diff > 0.025:
            warnings.warn(
                f"{name} price differs from analytical by {diff:.2%}",
                RuntimeWarning,
            )

    warn_if_far('flat', flat_price)
    warn_if_far('dupire', dupire_price)
