"""Plot the implied volatility surface example.

Run from the project root as ``python examples/plots/volatility/volatility.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    main()
