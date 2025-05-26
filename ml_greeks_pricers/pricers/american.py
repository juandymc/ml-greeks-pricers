import tensorflow as tf
from ml_greeks_pricers.volatility.discrete import ImpliedVolSurface, DupireLocalVol
from ml_greeks_pricers.common.constants import USE_XLA

class MCAmericanOption:
    """Optimized LSM Monte Carlo pricer for American options with pathwise Greeks using GradientTape."""
    def __init__(
        self, S0, K, T, r, q, vol_model,
        *, is_call=False,
        n_paths=100_000,
        n_steps=60,
        antithetic=True,
        dtype=tf.float64,
        seed=0
    ):
        if antithetic:
            tf.debugging.assert_equal(
                n_paths % 2,
                0,
                message="n_paths must be even when antithetic sampling is active",
            )

        # parameters
        self.S0 = tf.Variable(S0, dtype=dtype, trainable=True)
        self.K  = tf.constant(K,  dtype=dtype)
        self.T  = tf.constant(T,  dtype=dtype)
        self.r  = tf.constant(r,  dtype=dtype)
        self.q  = tf.constant(q,  dtype=dtype)
        self.is_call = is_call
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.dtype = dtype
        self._dt = self.T / tf.cast(n_steps, dtype)
        self._times = tf.cast(tf.range(n_steps), dtype) * self._dt
        self._rng = tf.random.Generator.from_seed(seed)
        self._I3 = tf.eye(3, dtype=dtype)

        # volatility
        self._flat_sigma = None
        self._dupire_grid = None
        self._sigma_fn = self._init_sigma_fn(vol_model)

    def _init_sigma_fn(self, vm):
        if isinstance(vm, (float, int)):
            sigma = tf.Variable(float(vm), dtype=self.dtype, trainable=True)
            self._flat_sigma = sigma
            def sig_fn(inputs):
                return tf.fill([tf.shape(inputs)[0]], sigma)
            return sig_fn

        if isinstance(vm, DupireLocalVol):
            grid = tf.Variable(vm.surface, dtype=self.dtype, trainable=True)
            self._dupire_grid = grid
            strikes = tf.convert_to_tensor(vm.strikes, dtype=self.dtype)
            mats    = tf.convert_to_tensor(vm.maturities, dtype=self.dtype)
            def sig_fn(inputs):
                t = inputs[:,0]; s = inputs[:,1]
                return ImpliedVolSurface.bilinear(t, s, grid, strikes, mats)
            return sig_fn

        raise TypeError("vol_model must be float/int or DupireLocalVol")

    @tf.function(jit_compile=USE_XLA)
    def price_and_grads(self):
        # generate paths and price under one tape for gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.S0)
            if self._flat_sigma is not None:
                tape.watch(self._flat_sigma)
            if self._dupire_grid is not None:
                tape.watch(self._dupire_grid)

            # simulate Brownian increments
            half = self.n_paths // 2
            Z = self._rng.normal([self.n_steps, half], dtype=self.dtype)
            Z = tf.concat([Z, -Z], axis=1) if self.antithetic else Z
            dW = Z * tf.sqrt(self._dt)

            # simulate asset paths
            S = tf.fill([self.n_paths], self.S0)
            paths = tf.TensorArray(self.dtype, size=self.n_steps+1)
            paths = paths.write(0, S)
            for i in tf.range(self.n_steps):
                t = self._times[i]
                X = tf.stack([tf.fill([self.n_paths], t), S], axis=1)
                sigma_t = self._sigma_fn(X)
                S = S * tf.exp((self.r - 0.5*sigma_t**2)*self._dt + sigma_t*dW[i])
                paths = paths.write(i+1, S)
            paths = paths.stack()

            # LSM backward induction
            df = tf.exp(-self.r*self._dt)
            pay = tf.where(self.is_call, tf.nn.relu(paths-self.K), tf.nn.relu(self.K-paths))
            CF = pay[-1]
            eps = tf.cast(1e-6, self.dtype)
            for t in tf.range(self.n_steps-1, 0, -1):
                discounted = CF * df
                St = paths[t]
                itm = pay[t] > 0
                idx = tf.where(itm)[:,0]
                St_i = tf.gather(St, idx)
                CF_i = tf.gather(discounted, idx)
                Xmat = tf.stack([tf.ones_like(St_i), St_i, St_i**2], axis=1)
                XTX = tf.matmul(Xmat, Xmat, transpose_a=True)
                XTC = tf.matmul(Xmat, CF_i[:,None], transpose_a=True)
                beta = tf.linalg.solve(XTX + eps*self._I3, XTC)
                beta = tf.reshape(beta, [3])
                cont = beta[0] + beta[1]*St + beta[2]*St**2
                exercise = itm & (pay[t] > cont)
                CF = tf.where(exercise, pay[t], discounted)
            price = tf.reduce_mean(CF*df)

        # gradients
        delta = tape.gradient(price, self.S0)
        if self._flat_sigma is not None:
            vega = tape.gradient(price, self._flat_sigma)
        elif self._dupire_grid is not None:
            ggrid = tape.gradient(price, self._dupire_grid)
            vega = tf.reduce_sum(ggrid)
        else:
            vega = None
        del tape
        return price, delta, vega

    def __call__(self):
        p, d, v = self.price_and_grads()
        self._last_price, self._last_delta, self._last_vega = p, d, v
        return p

    def delta(self):
        if not hasattr(self, '_last_delta'):
            _ = self()
        return self._last_delta

    def vega(self):
        if not hasattr(self, '_last_vega'):
            _ = self()
        return self._last_vega
