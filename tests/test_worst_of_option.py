import pytest
import tensorflow as tf

from ml_greeks_pricers.pricers.tf import MarketData, BasketAsset, MCWorstOfOption

tf.keras.backend.set_floatx("float64")


def test_monte_carlo_worst_of_option():
    S0 = [100.0, 95.0]
    q = [0.0, 0.0]
    sigma = [0.25, 0.20]
    corr = [[1.0, 0.3], [0.3, 1.0]]
    K = 90.0
    T = 1.0
    r = 0.03
    n_paths = 50_000

    market = MarketData(r, sigma)
    asset = BasketAsset(S0, q, corr, T=T, dt=1/252, n_paths=n_paths, seed=42)
    option = MCWorstOfOption(asset, market, K, T, is_call=False)

    price = float(option())
    delta = option.delta().numpy()
    vega = option.vega().numpy()

    assert price == pytest.approx(6.998, rel=1e-2)
    assert delta[0] == pytest.approx(-0.193, rel=1e-1)
    assert delta[1] == pytest.approx(-0.221, rel=1e-1)
    assert vega[0] == pytest.approx(25.8, rel=1e-1)
    assert vega[1] == pytest.approx(25.5, rel=1e-1)
