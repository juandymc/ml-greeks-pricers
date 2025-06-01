"""Example of training a neural network for European options."""

import numpy as np
import matplotlib.pyplot as plt

from ml_greeks_pricers.nn import BlackScholes, run_test


def plot(title, pred, x, y, sizes, ylabel):
    rows = len(sizes)
    fig, ax = plt.subplots(rows, 2, figsize=(9, 4 * rows))
    for i, sz in enumerate(sizes):
        for j, kind in enumerate(("std", "diff")):
            ax[i, j].plot(x * 100, pred[(kind, sz)] * 100, "co", ms=2, mfc="w", label="pred")
            ax[i, j].plot(x * 100, y * 100, "r.", ms=0.5, label="target")
            ax[i, j].set_xlabel("spot (%)")
            ax[i, j].set_ylabel(ylabel)
            if i == 0:
                ax[i, j].set_title("standard" if kind == "std" else "differential")
            ax[i, j].legend(prop={"size": 8})
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sizes = [1024, 8192]
    n_test = 100
    seed = np.random.randint(1e4)
    print(f"seed {seed}")
    gen = BlackScholes()
    x, y, dy, vp, dp = run_test(gen, sizes, n_test, seed)
    plot("Black-Scholes values", vp, x, y, sizes, "value")
    plot("Black-Scholes deltas", dp, x, dy, sizes, "delta")
