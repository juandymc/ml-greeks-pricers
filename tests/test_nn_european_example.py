import numpy as np
import tensorflow as tf
from ml_greeks_pricers.nn import (
    bs_price,
    bs_delta,
    BlackScholes,
    NeuralApproximator,
)


def test_bs_functions():
    price = bs_price(1.0, 1.1, 0.2, 1.0)
    delta = bs_delta(1.0, 1.1, 0.2, 1.0)
    assert np.isclose(price, 0.04292010941409885)
    assert np.isclose(delta, 0.35325369152806296)


def test_training_set_reproducible():
    gen = BlackScholes()
    x, y, dy = gen.training_set(3, seed=0)
    assert x.shape == (3, 1)
    assert y.shape == (3, 1)
    assert dy.shape == (3, 1)
    assert np.isclose(x[0, 0], 1.6229017)
    assert np.isclose(y[0, 0], 0.49586323)
    assert np.isclose(dy[0, 0], 0.9833394)


def test_neural_approximator_shapes():
    tf.keras.backend.set_floatx("float32")
    gen = BlackScholes()
    x, y, dy = gen.training_set(20, seed=1)
    na = NeuralApproximator(x, y, dy)
    na.prepare(10, diff=False, hu=2, hl=1)
    na.train(epochs=2, steps=1, bs=5)
    x_te, _, _, _, _ = gen.test_set(n=5)
    v, d = na.predict(x_te)
    assert v.shape == (5, 1)
    assert d.shape == (5, 1)
