from pathlib import Path
from ..utils import load_surface, plot_surface


def main() -> None:
    examples_dir = Path(__file__).resolve().parents[2]
    outputs_dir = examples_dir / "outputs"
    csv_files = outputs_dir.rglob("*_prices.csv")
    for csv_path in csv_files:
        X, Y, Z = load_surface(csv_path)
        name = f"{csv_path.parent.name}_{csv_path.stem}.png"
        out_file = Path(__file__).with_name(name)
        title = f"{csv_path.parent.name} {csv_path.stem.replace('_', ' ').title()}"
        plot_surface(
            X,
            Y,
            Z,
            xlabel="Strike",
            ylabel="Maturity",
            zlabel="Price",
            title=title,
            out_file=out_file,
        )


if __name__ == "__main__":
    main()
