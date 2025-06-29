"""Example of training a neural network for European options."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tensorflow as tf

from ml_greeks_pricers.nn import MCEuropeanOption, run_test
from ml_greeks_pricers.pricers.tf.european import (
    MarketData,
    EuropeanAsset,
    AnalyticalEuropeanOption,
)
from ml_greeks_pricers.volatility.discrete import DupireLocalVol


def plot(title, pred, ana, x, y, sizes, ylabel):
    rows = len(sizes)
    fig, ax = plt.subplots(rows, 2, figsize=(9, 4 * rows))
    for i, sz in enumerate(sizes):
        for j, kind in enumerate(("std", "diff")):
            ax[i, j].plot(x * 100, pred[(kind, sz)] * 100, "co", ms=2, mfc="w", label="pred")
            ax[i, j].plot(x * 100, y * 100, "r.", ms=0.5, label="target")
            ax[i, j].plot(x * 100, ana * 100, "g--", lw=1, label="analytical")
            ax[i, j].set_xlabel("spot (%)")
            ax[i, j].set_ylabel(ylabel)
            if i == 0:
                ax[i, j].set_title("standard" if kind == "std" else "differential")
            ax[i, j].legend(prop={"size": 8})
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sizes = [1024, 80192]
    n_test = 100
    seed = 6233  # np.random.randint(1e4)
    print(f"seed {seed}")

    inputs_dir = Path(__file__).resolve().parents[1] / "inputs"
    iv_df = pd.read_csv(inputs_dir / "nn_implied_vol_surface.csv", index_col=0)

    strikes = [float(c) for c in iv_df.columns]
    mats = [float(i) for i in iv_df.index]
    iv = iv_df.values.tolist()

    flat_sigma = float(iv_df.loc[1.0, "1.10"])

    market_flat = MarketData(0.0, flat_sigma)
    dupire = DupireLocalVol(strikes, mats, iv, 1.0, 0.0, 0.0)
    market_dup = MarketData(0.0, dupire)

    gen = MCEuropeanOption(market_flat, S0=1.0, q=0.0, factor=260, T1=1.0)

    x, y, dy, vp_flat, dp_flat = run_test(gen, sizes, n_test, seed)
    _, _, _, vp_dup, dp_dup = run_test(gen, sizes, n_test, seed, market=market_dup)

    T = gen.T2 - gen.T1
    # ``run_test`` returns float32 arrays, while the analytical pricer uses
    # float64.  Cast the input to avoid dtype mismatches.
    ana = AnalyticalEuropeanOption(
        tf.constant(x, dtype=tf.float64),
        gen.K,
        T,
        0.0,
        market_flat.r,
        0.0,
        market_flat._flat_sigma,
        is_call=True,
    )
    v_ana = ana.price().numpy().reshape(-1, 1)
    d_ana = ana.delta().numpy().reshape(-1, 1)

    plot("Flat volatility values", vp_flat, v_ana[:, 0], x, y, sizes, "value")
    plot("Dupire volatility values", vp_dup, v_ana[:, 0], x, y, sizes, "value")
    plot("Flat volatility deltas", dp_flat, d_ana[:, 0], x, dy, sizes, "delta")
    plot("Dupire volatility deltas", dp_dup, d_ana[:, 0], x, dy, sizes, "delta")
