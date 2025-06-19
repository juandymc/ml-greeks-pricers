import pytest
import tensorflow as tf

from ml_greeks_pricers.pricers.tf.american import AmericanAsset, MCAmericanOption
from ml_greeks_pricers.pricers.tf.european import MarketData


tf.keras.backend.set_floatx("float64")


def test_monte_carlo_american():
    S0, K, r, q, T = 110.0, 90.0, 0.06, 0.0, 0.5
    n_paths, n_steps, seed = 200_000, 100, 42
    dt = T / n_steps
    sigma = 0.212

    market = MarketData(r, sigma)
    asset = AmericanAsset(
        S0,
        q,
        T=T,
        dt=dt,
        n_paths=n_paths,
        antithetic=True,
        seed=seed,
    )
    option = MCAmericanOption(asset, market, K, T, is_call=False)

    price = float(option())
    delta = float(option.delta())
    vega = float(option.vega())

    assert price == pytest.approx(0.39972271913087026, rel=1e-2, abs=1e-3)
    assert delta == pytest.approx(-0.054422702022917754, rel=1e-2, abs=1e-3)
    assert vega == pytest.approx(8.5237423677645676, rel=1e-2, abs=1e-2)
