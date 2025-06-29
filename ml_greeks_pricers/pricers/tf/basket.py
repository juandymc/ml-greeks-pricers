import tensorflow as tf
from .european import EuropeanAsset, MarketData


tf.keras.backend.set_floatx('float64')


class BasketAsset(EuropeanAsset):
    """Simulate paths for a basket of correlated assets."""

    def __init__(
        self,
        S0,
        q,
        corr,
        *,
        T,
        dt,
        n_paths=100_000,
        antithetic=True,
        seed=0,
        dtype=tf.float64,
        use_scan=False,
    ):
        self.corr = tf.convert_to_tensor(corr, dtype=dtype)
        self.n_assets = len(S0)
        super().__init__(
            S0,
            q,
            T=T,
            dt=dt,
            n_paths=n_paths,
            antithetic=antithetic,
            seed=seed,
            dtype=dtype,
            use_scan=use_scan,
        )
        self._cached_dW = tf.Variable(
            tf.zeros([self.n_steps, n_paths, self.n_assets], dtype=dtype),
            trainable=False,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _brownian(self, n_steps):
        M = tf.cast(n_steps, tf.int32)
        sd = tf.sqrt(self.dt)
        shape = [M, self.n_paths // (2 if self.antithetic else 1), self.n_assets]
        Z = tf.random.stateless_normal(shape, [self.seed, 0], dtype=self.dtype)
        if self.antithetic:
            Z = tf.concat([Z, -Z], axis=1)
        chol = tf.linalg.cholesky(self.corr)
        corr_Z = tf.einsum('ij,tkj->tki', chol, Z)
        return corr_Z * sd

    def simulate(self, T, market: MarketData, *, use_cache=True, save_path=False):
        T_val = float(tf.get_static_value(T))
        steps = int(round(T_val / float(tf.get_static_value(self.dt))))

        if use_cache and self._cache_valid and steps <= self._cached_steps:
            if steps == 0:
                shape = tf.concat([[self.n_paths], tf.shape(self.S0)], 0)
                return tf.broadcast_to(self.S0, shape)
            dW = self._cached_dW[:steps]
        else:
            dW = self._brownian(steps)
            if use_cache:
                self._cached_dW[:steps].assign(dW)
                self._cache_valid = True
                self._cached_steps = steps

        times = tf.range(steps, dtype=self.dtype) * self.dt
        n_assets = tf.size(self.S0)
        S = tf.broadcast_to(tf.reshape(self.S0, [1, -1]), [self.n_paths, n_assets])

        def step(prev, elems):
            dWi, tc = elems
            if market._flat_sigma is not None:
                sig = tf.fill(tf.shape(prev), market._flat_sigma)
            elif market._vector_sigma is not None:
                sig = tf.broadcast_to(market._vector_sigma, tf.shape(prev))
            else:
                flat_prev = tf.reshape(prev, [-1])
                sig_flat = market._sigma_fn(
                    tf.stack([tf.fill([tf.size(flat_prev)], tc), flat_prev], axis=1)
                )
                sig = tf.reshape(sig_flat, tf.shape(prev))
            return prev * tf.exp(
                (market.r - self.q - 0.5 * sig ** 2) * self.dt + sig * dWi
            )

        if self.use_scan:
            path = tf.scan(step, (dW, times), initializer=S)
            result = path[-1]
            if save_path:
                self.path = path
        else:
            result = tf.foldl(step, (dW, times), initializer=S)
            if save_path:
                self.path = result

        return result


class MCWorstOfOption:
    """Monte Carlo pricer for worst-of basket options."""

    def __init__(self, asset: BasketAsset, market: MarketData, K, T, *, is_call=False, use_cache=True):
        self.asset = asset
        self.market = market
        self.K = tf.constant(K, dtype=asset.dtype)
        self.T = tf.constant(T, dtype=asset.dtype)
        self.is_call = is_call
        self.use_cache = use_cache
        self._last_price = None
        self._last_delta = None
        self._last_vega = None

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _compute_price_and_grads(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.asset.S0)
            if self.market._flat_sigma is not None:
                tape.watch(self.market._flat_sigma)
            elif self.market._vector_sigma is not None:
                tape.watch(self.market._vector_sigma)
            elif self.market._dupire_grid is not None:
                tape.watch(self.market._dupire_grid)
            ST = self.asset.simulate(self.T, self.market, use_cache=self.use_cache)
            worst = tf.reduce_min(ST, axis=1)
            payoff = tf.where(
                self.is_call,
                tf.nn.relu(worst - self.K),
                tf.nn.relu(self.K - worst),
            )
            price = tf.exp(-self.market.r * self.T) * tf.reduce_mean(payoff)

        delta = tape.gradient(price, self.asset.S0)
        if delta is None:
            delta = tf.zeros_like(self.asset.S0)

        if self.market._flat_sigma is not None:
            vega = tape.gradient(price, self.market._flat_sigma)
            if vega is None:
                vega = tf.zeros_like(price)
        elif self.market._vector_sigma is not None:
            vega = tape.gradient(price, self.market._vector_sigma)
            if vega is None:
                vega = tf.zeros_like(self.market._vector_sigma)
        else:
            grid_grad = tape.gradient(price, self.market._dupire_grid)
            if grid_grad is None:
                vega = tf.zeros_like(price)
            else:
                vega = tf.reduce_sum(grid_grad, axis=list(range(1, grid_grad.shape.rank)))
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
