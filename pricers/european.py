from volatility.discrete import ImpliedVolSurface, DupireLocalVol

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

USE_XLA = bool(tf.config.list_physical_devices('GPU'))

class AnalyticalEuropeanOption:
    def __init__(
        self,
        S:     tf.Tensor,
        K:     tf.Tensor,
        T:     tf.Tensor,
        t:     tf.Tensor,
        r:     tf.Tensor,
        q:     tf.Tensor,
        sigma: tf.Tensor,
        is_call: bool,
        dtype: tf.DType = tf.float64,
    ):
        self.dtype   = dtype
        self.S       = tf.convert_to_tensor(S,     dtype=dtype)
        self.K       = tf.convert_to_tensor(K,     dtype=dtype)
        self.T       = tf.convert_to_tensor(T,     dtype=dtype)
        self.t       = tf.convert_to_tensor(t,     dtype=dtype)
        self.r       = tf.convert_to_tensor(r,     dtype=dtype)
        self.q       = tf.convert_to_tensor(q,     dtype=dtype)
        self.sigma   = tf.convert_to_tensor(sigma, dtype=dtype)
        self.is_call = is_call

    @staticmethod
    @tf.function
    def bs_price(
        S:     tf.Tensor,
        K:     tf.Tensor,
        T:     tf.Tensor,
        t:     tf.Tensor,
        r:     tf.Tensor,
        q:     tf.Tensor,
        sigma: tf.Tensor,
        is_call: bool
    ) -> tf.Tensor:
        dt    = S.dtype
        half  = tf.constant(0.5, dtype=dt)
        one   = tf.constant(1.0, dtype=dt)
        two   = tf.constant(2.0, dtype=dt)

        tau       = tf.maximum(T - t, tf.constant(0.0, dtype=dt))
        sqrt_tau  = tf.sqrt(tau)
        d1 = (
            tf.math.log(S / K)
            + (r - q + half * tf.square(sigma)) * tau
        ) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau

        cdf = lambda x: half * (one + tf.math.erf(x / tf.sqrt(two)))
        df_r = tf.exp(-r * tau)
        df_q = tf.exp(-q * tau)

        call = S * df_q * cdf(d1) - K * df_r * cdf(d2)
        put  = K * df_r * cdf(-d2) - S * df_q * cdf(-d1)

        return tf.where(is_call, call, put)

    @tf.function
    def price(self) -> tf.Tensor:
        return self.bs_price(
            self.S, self.K, self.T, self.t, self.r, self.q, self.sigma, self.is_call
        )

    @staticmethod
    def _first_derivative(fn, var: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(var)
            y = fn()
        return tape.gradient(y, var)

    @staticmethod
    def _second_derivative(fn, var1: tf.Tensor, var2: tf.Tensor=None) -> tf.Tensor:
        """
        Compute ∂²(fn())/(∂var2 ∂var1).
        If var2 is None, computes ∂²/∂var1².
        """
        if var2 is None:
            var2 = var1
        with tf.GradientTape() as outer:
            outer.watch(var2)
            with tf.GradientTape() as inner:
                inner.watch(var1)
                y = fn()
            first = inner.gradient(y, var1)
        return outer.gradient(first, var2)

    def delta(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.S)

    def vega(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.sigma)

    def rho(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.r)

    def theta_t(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.t)

    def theta_T(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.T)

    def delta_strike(self) -> tf.Tensor:
        return self._first_derivative(lambda: self.price(), self.K)

    def gamma(self) -> tf.Tensor:
        # ∂²P/∂S²
        return self._second_derivative(lambda: self.price(), self.S)

    def gamma_strike(self) -> tf.Tensor:
        # ∂²P/∂K²
        return self._second_derivative(lambda: self.price(), self.K)


    __call__ = price
# Helper to enforce even numbers for antithetic sampling
def _assert_even(n, name):
    tf.debugging.assert_equal(n % 2, 0,
        message=f"{name} must be even when antithetic sampling is active")

# Helper to enforce even numbers for antithetic sampling
def _assert_even(n, name):
    tf.debugging.assert_equal(n % 2, 0,
        message=f"{name} must be even when antithetic sampling is active")

class MCEuropeanOption:
    """
    Monte Carlo para opción europea con cálculo simultáneo de precio, Delta y Vega.
    • vol_model float/int → volatilidad plana (cerrado en un paso)
    • vol_model DupireLocalVol → volatilidad local (Euler iterativo solo S_T)
    """
    def __init__(self, S0, K, T, r, q, vol_model, *,
                 is_call=False, n_paths=100_000, n_steps=60,
                 antithetic=True, whitening=True, seed=0, dtype=tf.float64):
        if antithetic:
            _assert_even(n_paths, 'n_paths')
            _assert_even(n_steps, 'n_steps')

        self.S0 = tf.constant(S0, dtype=dtype)
        self.K  = tf.constant(K,  dtype=dtype)
        self.T  = tf.constant(T,  dtype=dtype)
        self.r  = tf.constant(r,  dtype=dtype)
        self.q  = tf.constant(q,  dtype=dtype)

        self.is_call    = is_call
        self.n_paths    = n_paths
        self.n_steps    = n_steps
        self.antithetic = antithetic
        self.whitening  = whitening
        self.seed       = seed
        self.dtype      = dtype

        self._flat_sigma  = None
        self._dupire_grid = None
        self._sigma_fn    = self._build_sigma_fn(vol_model)

        self._last_price = None
        self._last_delta = None
        self._last_vega  = None

    def _build_sigma_fn(self, vm):
        if isinstance(vm, (float, int)):
            self._flat_sigma = tf.Variable(float(vm), dtype=self.dtype, trainable=True)
            @tf.function
            def const_sigma(x):
                return tf.fill([tf.shape(x)[0]], self._flat_sigma)
            return const_sigma

        if hasattr(vm, 'surface') and hasattr(vm, 'strikes'):
            self._dupire_grid = tf.Variable(vm.surface, dtype=self.dtype, trainable=True)
            strikes, mats = vm.strikes, vm.maturities
            @tf.function
            def local_sigma(x):
                t, spot = x[:,0], x[:,1]
                grid = self._dupire_grid.read_value()
                return ImpliedVolSurface.bilinear(t, spot, grid, strikes, mats)
            return local_sigma

        raise TypeError("vol_model debe ser float/int o DupireLocalVol")

    @tf.function(jit_compile=True)
    def _brownian(self):
        M, N = self.n_steps, self.n_paths
        dt  = self.T / tf.cast(M, self.dtype)
        sd  = tf.sqrt(dt)
        Z = tf.random.stateless_normal([M, N//(2 if self.antithetic else 1)],
                                       [self.seed, 0], dtype=self.dtype)
        if self.whitening and self.antithetic:
            C = tf.matmul(Z, Z, transpose_b=True) / tf.cast(tf.shape(Z)[1], self.dtype)
            e, v = tf.linalg.eigh(C)
            Z = tf.matmul(v, tf.matmul(tf.linalg.diag(tf.math.rsqrt(e)), tf.matmul(v, Z)))
        if self.antithetic:
            Z = tf.concat([Z, -Z], axis=1)
        return Z * sd

    @tf.function(jit_compile=True)
    def _compute_price_and_grads(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.S0)
            if self._flat_sigma is not None:
                tape.watch(self._flat_sigma)
            elif self._dupire_grid is not None:
                tape.watch(self._dupire_grid)

            # simulación y payoff
            if self._flat_sigma is not None:
                sigma = self._flat_sigma
                Z = tf.random.stateless_normal([self.n_paths], [self.seed, 0], dtype=self.dtype)
                ST = self.S0 * tf.exp((self.r - 0.5 * sigma**2) * self.T + sigma * tf.sqrt(self.T) * Z)
            else:
                dt    = self.T / tf.cast(self.n_steps, self.dtype)
                dW    = self._brownian()
                times = tf.range(self.n_steps, dtype=self.dtype) * dt
                S     = tf.fill([self.n_paths], self.S0)
                # solo S_T con foldl para no guardar full path
                def step(prev, elems):
                    dWi, tc = elems
                    sig = self._sigma_fn(tf.stack([tf.fill([self.n_paths], tc), prev], axis=1))
                    return prev * tf.exp((self.r - 0.5 * sig**2) * dt + sig * dWi)
                ST = tf.foldl(step, (dW, times), initializer=S)

            payoff = tf.where(self.is_call,
                              tf.nn.relu(ST - self.K),
                              tf.nn.relu(self.K - ST))
            price = tf.exp(-self.r * self.T) * tf.reduce_mean(payoff)

        delta = tape.gradient(price, self.S0)
        if self._flat_sigma is not None:
            vega = tape.gradient(price, self._flat_sigma)
        else:
            grid_grad = tape.gradient(price, self._dupire_grid)
            vega = tf.reduce_sum(grid_grad)
        del tape
        return price, delta, vega

    def __call__(self):
        price, delta, vega = self._compute_price_and_grads()
        self._last_price = price
        self._last_delta = delta
        self._last_vega  = vega
        return price

    def delta(self):
        if self._last_delta is None:
            _ = self.__call__()
        return self._last_delta

    @tf.function(jit_compile=True)
    def vega_bucket(self):
        if self._dupire_grid is None:
            raise ValueError("bucket-vega sólo disponible con DupireLocalVol")
        with tf.GradientTape() as tape:
            price = self.__call__()
        return tape.gradient(price, self._dupire_grid)

    def vega(self):
        if self._last_vega is None:
            _ = self.__call__()
        return self._last_vega
