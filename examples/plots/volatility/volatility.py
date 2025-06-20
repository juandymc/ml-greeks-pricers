"""Plot the implied volatility surface example.

Run from the project root as ``python examples/plots/volatility/volatility.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_slices(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    zlabel: str,
    title: str,
    out_file: Path,
    strike_idx: int | None = None,
    maturity_idx: int | None = None,
):
    """Generate and save 2D slice plots for fixed strike and maturity."""
    strikes = X[0]
    maturities = Y[:, 0]

    if strike_idx is None:
        strike_idx = len(strikes) // 2
    if maturity_idx is None:
        maturity_idx = len(maturities) // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(maturities, Z[:, strike_idx])
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel(zlabel)
    axes[0].set_title(f"Strike = {strikes[strike_idx]:.2f}")

    axes[1].plot(strikes, Z[maturity_idx, :])
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel(zlabel)
    axes[1].set_title(f"Maturity = {maturities[maturity_idx]:.2f}")

    fig.suptitle(title)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def load_surface(csv_path: Path):
    """Return strike, maturity meshgrids and surface values from a CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    strikes = df.columns.astype(float)
    maturities = df.index.astype(float)
    X, Y = np.meshgrid(strikes, maturities)
    Z = df.values.astype(float)
    return X, Y, Z


def plot_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    out_file: Path,
):
    """Generate and save a 3D surface plot."""
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def main() -> None:
    examples_dir = Path(__file__).resolve().parents[2]
    csv_path = examples_dir / "inputs" / "implied_vol_surface.csv"
    X, Y, Z = load_surface(csv_path)
    out_file = Path(__file__).with_name("implied_vol_surface.png")
    plot_surface(
        X,
        Y,
        Z,
        xlabel="Strike",
        ylabel="Maturity",
        zlabel="Implied Volatility",
        title="Implied Volatility Surface",
        out_file=out_file,
    )
    slice_file = out_file.with_name(out_file.stem + "_slice.png")
    plot_slices(
        X,
        Y,
        Z,
        zlabel="Implied Volatility",
        title="Implied Volatility Surface",
        out_file=slice_file,
    )


if __name__ == "__main__":
    main()
