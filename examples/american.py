import tensorflow as tf

from ml_greeks_pricers.volatility.discrete import DupireLocalVol
from ml_greeks_pricers.pricers.european import MarketData, EuropeanAsset, MCEuropeanOption
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
    dupire_lv = DupireLocalVol(strikes, mats, iv, S0, r, q)
    dupire_lv = 0.212  # constant volatility as reference

    market = MarketData(r, dupire_lv)

    dt = T / n_steps
    asset_eur = EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_eur = MCEuropeanOption(asset_eur, market, K, T, is_call=False)

    price_eur = mc_eur()
    delta_eur = mc_eur.delta()
    vega_eur = mc_eur.vega()
    tf.print("European:", price_eur, "Delta:", delta_eur, "Vega:", vega_eur)

    asset_amer = AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_amer = MCAmericanOption(asset_amer, market, K, T, is_call=False)

    price_amer = mc_amer()
    delta_amer = mc_amer.delta()
    vega_amer = mc_amer.vega()
    tf.print("American:", price_amer, "Delta:", delta_amer, "Vega:", vega_amer)
