import tensorflow as tf

from ml_greeks_pricers.common.constants import USE_XLA


class AmericanAsset:
    """Simulate paths for an American option underlying asset."""

    def __init__(
        self,
        S0,
        q,
        *,
        T,
        dt,
        n_paths=100_000,
        antithetic=True,
        seed=0,
        dtype=tf.float64,
    ):
        if antithetic:
            tf.debugging.assert_equal(
                n_paths % 2,
                0,
                message="n_paths must be even when antithetic sampling is active",
            )

        self.n_steps = int(round(float(T) / float(dt)))
        if antithetic:
            tf.debugging.assert_equal(
                self.n_steps % 2,
                0,
                message="n_steps must be even when antithetic sampling is active",
            )

        self.S0 = tf.constant(S0, dtype=dtype)
        self.q = tf.constant(q, dtype=dtype)
        self.T = tf.constant(T, dtype=dtype)
        self.dt = tf.constant(float(T) / self.n_steps, dtype=dtype)
        self.n_paths = n_paths
        self.antithetic = antithetic
        self.seed = seed
        self.dtype = dtype
        self._times = tf.range(self.n_steps, dtype=dtype) * self.dt

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _brownian(self):
        M = tf.cast(self.n_steps, tf.int32)
        sd = tf.sqrt(self.dt)
        Z = tf.random.stateless_normal(
            [M, self.n_paths // (2 if self.antithetic else 1)],
            [self.seed, 0],
            dtype=self.dtype,
        )
        if self.antithetic:
            Z = tf.concat([Z, -Z], axis=1)
        return Z * sd

    @tf.function(jit_compile=True, reduce_retracing=True)
    def simulate(self, market):
        dW = self._brownian()
        S = tf.fill([self.n_paths], self.S0)
        paths = tf.TensorArray(self.dtype, size=self.n_steps + 1)
        paths = paths.write(0, S)
        for i in tf.range(self.n_steps):
            t = self._times[i]
            if market._flat_sigma is not None:
                sigma_t = tf.fill([self.n_paths], market._flat_sigma)
            else:
                sigma_t = market._sigma_fn(
                    tf.stack([tf.fill([self.n_paths], t), S], axis=1)
                )
            S = S * tf.exp(
                (market.r - self.q - 0.5 * sigma_t ** 2) * self.dt + sigma_t * dW[i]
            )
            paths = paths.write(i + 1, S)
        return paths.stack()


class MCAmericanOption:
    """Least Squares Monte Carlo pricer for American options."""

    def __init__(self, asset: AmericanAsset, market, K, T, *, is_call=False):
        self.asset = asset
        self.market = market
        self.K = tf.constant(K, dtype=asset.dtype)
        self.T = tf.constant(T, dtype=asset.dtype)
        self.is_call = is_call
        self.dtype = asset.dtype
        self._I3 = tf.eye(3, dtype=self.dtype)

        self._last_price = None
        self._last_delta = None
        self._last_vega = None

    @tf.function(jit_compile=USE_XLA, reduce_retracing=True)
    def _compute_price_and_grads(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.asset.S0)
            if self.market._flat_sigma is not None:
                tape.watch(self.market._flat_sigma)
            elif self.market._dupire_grid is not None:
                tape.watch(self.market._dupire_grid)

            paths = self.asset.simulate(self.market)

            df = tf.exp(-self.market.r * self.asset.dt)
            pay = tf.where(
                self.is_call, tf.nn.relu(paths - self.K), tf.nn.relu(self.K - paths)
            )
            CF = pay[-1]
            eps = tf.constant(1e-6, dtype=self.dtype)
            for t in tf.range(self.asset.n_steps - 1, 0, -1):
                discounted = CF * df
                St = paths[t]
                itm = pay[t] > 0
                idx = tf.where(itm)[:, 0]
                St_i = tf.gather(St, idx)
                CF_i = tf.gather(discounted, idx)
                Xmat = tf.stack([tf.ones_like(St_i), St_i, St_i ** 2], axis=1)
                XTX = tf.matmul(Xmat, Xmat, transpose_a=True)
                XTC = tf.matmul(Xmat, CF_i[:, None], transpose_a=True)
                beta = tf.linalg.solve(XTX + eps * self._I3, XTC)
                beta = tf.reshape(beta, [3])
                cont = beta[0] + beta[1] * St + beta[2] * St ** 2
                exercise = itm & (pay[t] > cont)
                CF = tf.where(exercise, pay[t], discounted)
            price = tf.reduce_mean(CF * df)

        delta = tape.gradient(price, self.asset.S0)
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
