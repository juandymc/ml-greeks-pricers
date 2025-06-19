import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

from ml_greeks_pricers.pricers.tf import european as tf_eur
from ml_greeks_pricers.pricers.np import european as np_eur
from ml_greeks_pricers.volatility.discrete import DupireLocalVol as TfDupire
from ml_greeks_pricers.volatility.np_discrete import DupireLocalVol as NpDupire


tf.keras.backend.set_floatx("float64")

def compare_dupire():
    S0, K, r, q, T = 110.0, 90.0, 0.06, 0.0, 0.5
    n_paths, n_steps = 50_000, 100
    dt = T / n_steps

    inputs_dir = Path(__file__).resolve().parent.parent.joinpath('inputs')
    # when running from this subfolder adjust relative path
    if not inputs_dir.exists():
        inputs_dir = Path(__file__).with_name('inputs')
    iv_df = pd.read_csv(inputs_dir / 'implied_vol_surface.csv', index_col=0)
    strikes = [float(c) for c in iv_df.columns]
    mats = [float(i) for i in iv_df.index]
    iv = iv_df.values.tolist()

    outputs_dir = Path(__file__).resolve().parent.parent.joinpath('outputs')
    outputs_dir.mkdir(exist_ok=True)

    dup_tf = TfDupire(strikes, mats, iv, S0, r, q)
    pd.DataFrame(dup_tf().numpy(), index=mats, columns=strikes).to_csv(
        outputs_dir / 'dupire_local_vol.csv'
    )
    market_tf = tf_eur.MarketData(r, dup_tf)
    asset_tf = tf_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, use_scan=True, seed=0)
    opt_tf = tf_eur.MCEuropeanOption(asset_tf, market_tf, K, T, is_call=False)
    price_tf = opt_tf().numpy()

    dup_np = NpDupire(strikes, mats, iv, S0, r, q)
    market_np = np_eur.MarketData(r, dup_np)
    asset_np = np_eur.EuropeanAsset(S0, q, T=T, dt=dt, n_paths=n_paths, seed=0)
    opt_np = np_eur.MCEuropeanOption(asset_np, market_np, K, T, is_call=False)
    price_np = opt_np()

    print("Dupire TF:", price_tf)
    print("Dupire NP:", price_np)
    print("Close:", np.isclose(price_tf, price_np, rtol=5e-2))


if __name__ == "__main__":
    compare_dupire()
