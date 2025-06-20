from pathlib import Path
from ..utils import load_surface, plot_surface


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
