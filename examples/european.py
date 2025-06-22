import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.keras.backend.clear_session()  


import os, warnings, re, tensorflow as tf
from pathlib import Path
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # suprime avisos de backend C++
tf.get_logger().setLevel('ERROR')          # suprime avisos Python de TF


from ml_greeks_pricers.pricers.tf.european import (
    AnalyticalEuropeanOption,
    MCEuropeanOption,
    MarketData,
    EuropeanAsset,
)
from ml_greeks_pricers.volatility.discrete import ImpliedVolSurface, DupireLocalVol


if __name__ == '__main__':
    n_paths = 150_000
    n_steps = 200
    S0,K,T,r,q = 100.,90.,1.5,0.,0.
    dt = T/n_steps
    iv_vol = 	0.208
    seed = 40
    analEur = AnalyticalEuropeanOption(S0, K, T, 0, r, q, iv_vol, is_call=True)
    analytical_price = analEur().numpy()
    tf.print('analytical', analytical_price, analEur.delta(), analEur.vega())
    # ---- volatilidad constante ---------------------------------------
    market_flat = MarketData(r, iv_vol)
    asset_flat = EuropeanAsset(
        S0,
        q,
        T=T,
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=seed,
    )
    mc_flat = MCEuropeanOption(asset_flat, market_flat, K, T, is_call=False)
    flat_price = mc_flat().numpy()
    tf.print('flat', flat_price, mc_flat.delta())#, mc_flat.vega())

    # ---- Dupire local-vol --------------------------------------------
    inputs_dir = Path(__file__).with_name("inputs")
    csv_dir = Path(__file__).with_name("outputs")
    csv_dir.mkdir(exist_ok=True)
    iv_df = pd.read_csv(inputs_dir / "implied_vol_surface.csv", index_col=0)
    iv_df[:]=iv_vol
    strikes = [float(c) for c in iv_df.columns]
    mats = [float(i) for i in iv_df.index]
    iv = iv_df.values.tolist()
    dup = DupireLocalVol(strikes, mats, iv, S0, r, q)
    pd.DataFrame(dup().numpy(), index=mats, columns=strikes).to_csv(
        csv_dir / "dupire_local_vol.csv"
    )

    market_dup = MarketData(r, dup)
    asset_dup = EuropeanAsset(
        S0,
        q,
        T=T,
        dt=dt,
        n_paths=n_paths,
        use_scan=True,
        seed=seed,
    )
    mc_loc = MCEuropeanOption(asset_dup, market_dup, K, T, is_call=False)
    dupire_price = mc_loc().numpy()
    tf.print('dupire', dupire_price, mc_loc.delta())#, mc_loc.vega())

    def warn_if_far(name, price):
        diff = 100*abs(price - analytical_price) / S0
        if diff > 0.05:
            warnings.warn(
                f"{name} price differs from analytical: price = {price}, analytical_price = {analytical_price}, diff = {diff}",
                RuntimeWarning,
            )

    warn_if_far('flat', flat_price)
    warn_if_far('dupire', dupire_price)
