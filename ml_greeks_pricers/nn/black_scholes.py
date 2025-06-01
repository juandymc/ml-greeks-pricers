from __future__ import annotations

import numpy as np
from scipy.stats import norm
import tensorflow as tf

from .scaler import TwinScaler
from .models import vanilla_net, TwinNetwork, WeightedMeanSquaredError
from .utils import lambda_j, alpha_beta, dataset, lr_callback
from ..pricers.european import MarketData, EuropeanAsset


# ---- analytical formulas ----------------------------------------------------

def bs_price(s, k, v, T):
    d1 = (np.log(s / k) + 0.5 * v * v * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return s * norm.cdf(d1) - k * norm.cdf(d2)


def bs_delta(s, k, v, T):
    d1 = (np.log(s / k) + 0.5 * v * v * T) / (v * np.sqrt(T))
    return norm.cdf(d1)


def bs_vega(s, k, v, T):
    d1 = (np.log(s / k) + 0.5 * v * v * T) / (v * np.sqrt(T))
    return s * np.sqrt(T) * norm.pdf(d1)


# ---- data generator ---------------------------------------------------------

class MCEuropeanOption:
    """Simple Monte Carlo generator for European options."""

    def __init__(
        self,
        market: MarketData,
        asset: EuropeanAsset,
        *,
        T1: float = 1.0,
        T2: float = 2.0,
        K: float = 1.1,
        vm: float = 1.5,
    ) -> None:
        if market._flat_sigma is None:
            raise ValueError("Only flat volatility supported")

        self.market = market
        self.asset = asset

        self.s0 = float(asset.S0.numpy())
        self.q = float(asset.q.numpy())
        self.dt = float(asset.dt.numpy())

        self.v = float(market._flat_sigma.numpy())
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.vm = vm

    def training_set(self, m: int, anti: bool = True, seed: int | None = None):
        """Generate a training set using :class:`EuropeanAsset`."""

        np.random.seed(seed)
        r = np.random.normal(size=(m, 2))

        asset = EuropeanAsset(
            self.s0,
            self.q,
            T=self.T2,
            dt=self.T1,
            n_paths=m,
            antithetic=False,
            seed=0 if seed is None else seed,
            use_scan=True,
        )

        dW = np.stack(
            [np.sqrt(self.T1) * r[:, 0], np.sqrt(self.T2 - self.T1) * r[:, 1]],
            axis=0,
        )
        asset._cached_dW[:2].assign(tf.constant(dW, dtype=asset.dtype))
        asset._cache_valid = True
        asset._cached_steps = 2

        S1 = asset.simulate(self.T1, self.market, use_cache=True).numpy().ravel()
        S2 = asset.simulate(self.T2, self.market, use_cache=True).numpy().ravel()

        dt = self.T2 - self.T1
        pay = np.maximum(0.0, S2 - self.K)
        R2 = S2 / S1

        if anti:
            Z = (np.log(R2) + 0.5 * self.v * self.v * dt) / (self.v * np.sqrt(dt))
            R2a = np.exp(-0.5 * self.v * self.v * dt - self.v * np.sqrt(dt) * Z)
            S2a = S1 * R2a
            paya = np.maximum(0.0, S2a - self.K)
            Y = 0.5 * (pay + paya)
            delta = 0.5 * (
                np.where(S2 > self.K, R2, 0.0) + np.where(S2a > self.K, R2a, 0.0)
            )
        else:
            Y = pay
            delta = np.where(S2 > self.K, R2, 0.0)

        return (
            S1.reshape(-1, 1).astype(np.float32),
            Y.reshape(-1, 1).astype(np.float32),
            delta.reshape(-1, 1).astype(np.float32),
        )

    def test_set(self, lo: float = 0.35, hi: float = 1.65, n: int = 100):
        s = np.linspace(lo, hi, n, dtype=np.float32).reshape(-1, 1)
        T = self.T2 - self.T1
        return (
            s,
            s,
            bs_price(s, self.K, self.v, T).astype(np.float32).reshape(-1, 1),
            bs_delta(s, self.K, self.v, T).astype(np.float32).reshape(-1, 1),
            bs_vega(s, self.K, self.v, T).astype(np.float32).reshape(-1, 1),
        )


# ---- approximator -----------------------------------------------------------

class NeuralApproximator:
    def __init__(self, x, y, dy=None):
        self.x_raw, self.y_raw = x, y
        if dy is None:
            self.dy_raw = tf.zeros((y.shape[0], y.shape[1], x.shape[1]), tf.float32)
        else:
            self.dy_raw = dy
        self.scaler = TwinScaler()

    def prepare(self, m: int, diff: bool, lam: float = 1.0, hu: int = 20, hl: int = 4):
        self.scaler.fit(self.x_raw[:m], self.y_raw[:m])
        self.x_s, self.y_s, self.dy_s = self.scaler.transform(
            self.x_raw[:m], self.y_raw[:m], self.dy_raw[:m]
        )
        n = self.x_s.shape[1]
        self.twin = TwinNetwork(vanilla_net(n, hu, hl))
        self.lam_j = lambda_j(self.dy_s)
        self.loss_w = alpha_beta(n, lam) if diff else [1, 0]

    def train(
        self,
        epochs: int = 100,
        lr_sched=[(0, 1e-8), (0.2, 1e-1), (0.6, 1e-2), (0.9, 1e-6), (1, 1e-8)],
        steps: int = 16,
        bs: int = 256,
    ):
        ds = dataset(self.x_s, self.y_s, self.dy_s, bs)
        self.twin.compile(
            optimizer="adam",
            loss=["mse", WeightedMeanSquaredError(self.lam_j)],
            loss_weights=self.loss_w,
        )
        self.twin.fit(ds, epochs=epochs, steps_per_epoch=steps, callbacks=[lr_callback(lr_sched, epochs)], verbose=0)

    def predict(self, x):
        x_s = self.scaler.x_transform(x)
        y_s, dy_s = self.twin.predict(x_s, verbose=0)
        return self.scaler.inverse(y_s, dy_s)


# ---- experiment -------------------------------------------------------------

def run_test(gen: MCEuropeanOption, sizes, n_test: int, seed: int):
    x_tr, y_tr, dy_tr = gen.training_set(max(sizes), seed=seed)
    x_te, x_ax, y_te, dy_te, _ = gen.test_set(n=n_test)
    reg = NeuralApproximator(x_tr, y_tr, dy_tr)
    v_pred, d_pred = {}, {}
    for sz in sizes:
        reg.prepare(sz, diff=False)
        reg.train()
        v, d = reg.predict(x_te)
        v_pred[("std", sz)], d_pred[("std", sz)] = v, d[:, 0]
        reg.prepare(sz, diff=True)
        reg.train()
        v, d = reg.predict(x_te)
        v_pred[("diff", sz)], d_pred[("diff", sz)] = v, d[:, 0]
    return x_ax, y_te, dy_te[:, 0], v_pred, d_pred
