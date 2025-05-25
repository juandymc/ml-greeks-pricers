"""Longstaff-Schwartz Monte Carlo pricer for American options."""

from __future__ import annotations

import tensorflow as tf

from ml_greeks_pricers.pricers.european import (
    MarketData,
    EuropeanAsset,
)
from ml_greeks_pricers.common.constants import USE_XLA


class AmericanAsset(EuropeanAsset):
    """Simulates full price paths for the underlying asset."""

    @tf.function(jit_compile=True, reduce_retracing=True)
    def simulate(self, T, market: MarketData, *, use_cache=True) -> tf.Tensor:
        """Return the full simulated path up to maturity ``T``."""

        # ``self.n_steps`` already stores the integer number of steps derived
        # from ``T`` and ``dt`` at construction time.  Using it directly avoids
        # calling ``tf.get_static_value`` on symbolic tensors, which fails when
        # ``simulate`` is invoked from a ``tf.function``.
        steps = self.n_steps

        if use_cache and self._cache_valid and steps <= self._cached_steps:
            if steps == 0:
                S = tf.fill([self.n_paths], self.S0)
                return tf.expand_dims(S, 0)
            dW = self._cached_dW[:steps]
        else:
            dW = self._brownian(steps)
            if use_cache:
                self._cached_dW[:steps].assign(dW)
                self._cache_valid = True
                self._cached_steps = steps

        times = tf.range(steps, dtype=self.dtype) * self.dt
        S = tf.fill([self.n_paths], self.S0)
        path = tf.TensorArray(self.dtype, size=steps + 1)
        path = path.write(0, S)

        def step(prev, elems):
            dWi, tc = elems
            if market._flat_sigma is not None:
                sig = tf.fill([self.n_paths], market._flat_sigma)
            else:
                sig = market._sigma_fn(
                    tf.stack([tf.fill([self.n_paths], tc), prev], axis=1)
                )
            return prev * tf.exp((market.r - self.q - 0.5 * sig ** 2) * self.dt + sig * dWi)

        for i in tf.range(steps):
            S = step(S, (dW[i], times[i]))
            path = path.write(i + 1, S)

        return path.stack()


class MCAmericanOption:
    """Monte Carlo pricer for American options using the LSM algorithm."""

    def __init__(
        self,
        asset: AmericanAsset,
        market: MarketData,
        K,
        T,
        *,
        is_call: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.asset = asset
        self.market = market
        self.K = tf.constant(K, dtype=asset.dtype)
        self.T = tf.constant(T, dtype=asset.dtype)
        self.is_call = is_call
        self.use_cache = use_cache

        self._eye3 = tf.eye(3, dtype=asset.dtype)
        self._last_price = None
        self._last_delta = None
        self._last_vega = None

    @tf.function(jit_compile=USE_XLA, reduce_retracing=True)
    def _compute_price_and_grads(self):
        dt = self.asset.dt
        df = tf.exp(-self.market.r * dt)
        n_steps = self.asset.n_steps

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.asset.S0)
            if self.market._flat_sigma is not None:
                tape.watch(self.market._flat_sigma)
            elif self.market._dupire_grid is not None:
                tape.watch(self.market._dupire_grid)

            paths = self.asset.simulate(self.T, self.market, use_cache=self.use_cache)

            payoff = tf.where(
                self.is_call,
                tf.nn.relu(paths - self.K),
                tf.nn.relu(self.K - paths),
            )

            CF = payoff[-1]
            eps = tf.cast(1e-6, self.asset.dtype)
            for t in tf.range(n_steps - 1, 0, -1):
                discounted = CF * df
                St = paths[t]
                itm = payoff[t] > 0
                idx = tf.where(itm)[:, 0]
                St_i = tf.gather(St, idx)
                CF_i = tf.gather(discounted, idx)
                X = tf.stack([tf.ones_like(St_i), St_i, St_i ** 2], axis=1)
                XTX = tf.matmul(X, X, transpose_a=True)
                XTC = tf.matmul(X, CF_i[:, None], transpose_a=True)
                beta = tf.linalg.solve(XTX + eps * self._eye3, XTC)
                beta = tf.reshape(beta, [3])
                cont = beta[0] + beta[1] * St + beta[2] * St ** 2
                exercise = itm & (payoff[t] > cont)
                CF = tf.where(exercise, payoff[t], discounted)

            price = tf.reduce_mean(CF * df)

        delta = tape.gradient(price, self.asset.S0)
        if delta is None:
            delta = tf.zeros_like(self.asset.S0)

        if self.market._flat_sigma is not None:
            vega = tape.gradient(price, self.market._flat_sigma)
            if vega is None:
                vega = tf.zeros_like(self.market._flat_sigma)
        else:
            grid_grad = tape.gradient(price, self.market._dupire_grid)
            if grid_grad is None:
                grid_grad = tf.zeros_like(self.market._dupire_grid)
            vega = tf.reduce_sum(grid_grad)

        del tape
        return price, delta, vega

    def __call__(self):
        price, delta, vega = self._compute_price_and_grads()
        self._last_price = price
        self._last_delta = delta
        self._last_vega = vega
        return price

    def delta(self):
        if self._last_delta is None:
            _ = self.__call__()
        return self._last_delta

    def vega(self):
        if self._last_vega is None:
            _ = self.__call__()
        return self._last_vega


__all__ = ["MarketData", "AmericanAsset", "MCAmericanOption"]

