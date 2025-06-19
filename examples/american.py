import tensorflow as tf
from pathlib import Path
import pandas as pd

from ml_greeks_pricers.volatility.discrete import DupireLocalVol
from ml_greeks_pricers.pricers.tf.european import MarketData, EuropeanAsset, MCEuropeanOption
from ml_greeks_pricers.pricers.tf.american import AmericanAsset, MCAmericanOption


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
    
    dt = T / n_steps

    csv_dir = Path(__file__).with_name("outputs")
    csv_dir.mkdir(exist_ok=True)
    inputs_dir = Path(__file__).with_name("inputs")
    iv_df = pd.read_csv(inputs_dir / "implied_vol_surface.csv", index_col=0)

    strikes = [float(c) for c in iv_df.columns]
    mats = [float(i) for i in iv_df.index]
    iv = iv_df.values.tolist()

    dupire_lv = DupireLocalVol(strikes, mats, iv, S0, r, q)
    pd.DataFrame(dupire_lv().numpy(), index=mats, columns=strikes).to_csv(
        csv_dir / "dupire_local_vol.csv"
    )
    flat_lv = 0.212

    market_flat = MarketData(r, flat_lv)
    market_dup  = MarketData(r, dupire_lv)


    asset_amer = AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_amer = MCAmericanOption(asset_amer, market_flat, K, T, is_call=False, use_cache = True)

    price_amer = mc_amer()
    delta_amer = mc_amer.delta()
    vega_amer = mc_amer.vega()
    tf.print("American flat:", price_amer, "Delta:", delta_amer, "Vega:", vega_amer)
    
    asset_amer = AmericanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, antithetic=True, seed=seed)
    mc_amer = MCAmericanOption(asset_amer, market_dup, K, T, is_call=False, use_cache = True)
    price_amer = mc_amer()
    delta_amer = mc_amer.delta()
    vega_amer = mc_amer.vega()
    tf.print("American Dupire:", price_amer, "Delta:", delta_amer, "Vega:", vega_amer)
