"""Example of training a neural network for European options."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ml_greeks_pricers.nn import MCEuropeanOption, run_test
from ml_greeks_pricers.pricers.tf.european import (
    MarketData,
    EuropeanAsset,
    AnalyticalEuropeanOption,
)


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
    seed = 6233#np.random.randint(1e4)
    print(f"seed {seed}")
    market = MarketData(0.0, 0.2)
    # Use an even number of paths and steps when antithetic sampling is enabled
    # (default behaviour).  ``dt=0.5`` ensures that ``n_steps`` is even for
    # ``T=1.0``.
    #asset = EuropeanAsset(1.0, 0.0, T=1.0, dt=0.5, n_paths=2)
    gen = MCEuropeanOption(market, S0 = 1.0, q = 0.0, factor = 260, T1 = 1.0 )
    x, y, dy, vp, dp = run_test(gen, sizes, n_test, seed)

    T = gen.T2 - gen.T1
    # ``run_test`` returns float32 arrays, while the analytical pricer uses
    # float64.  Cast the input to avoid dtype mismatches.
    ana = AnalyticalEuropeanOption(
        tf.constant(x, dtype=tf.float64),
        gen.K,
        T,
        0.0,
        market.r,
        0.0,
        market._flat_sigma,
        is_call=True,
    )
    v_ana = ana.price().numpy().reshape(-1, 1)
    d_ana = ana.delta().numpy().reshape(-1, 1)

    plot("Black-Scholes values", vp, v_ana[:, 0], x, y, sizes, "value")
    plot("Black-Scholes deltas", dp, d_ana[:, 0], x, dy, sizes, "delta")
