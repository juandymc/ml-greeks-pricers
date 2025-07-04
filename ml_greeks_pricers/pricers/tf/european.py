import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from ml_greeks_pricers.volatility.discrete import ImpliedVolSurface, DupireLocalVol
from collections.abc import Sequence
from ml_greeks_pricers.common.constants import USE_XLA

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
    @tf.function(reduce_retracing=True)
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

    @tf.function(reduce_retracing=True)
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
    tf.debugging.assert_equal(
        n % 2,
        0,
        message=f"{name} must be even when antithetic sampling is active",
    )

class MarketData:
    """Container for market information used in simulations."""

    def __init__(self, r, vol_model, *, dtype=tf.float64):
        self.r = tf.constant(r, dtype=dtype)
        self.dtype = dtype

        self._flat_sigma = None
        self._vector_sigma = None
        self._dupire_grid = None
        self._sigma_fn = self._build_sigma_fn(vol_model)

    def _build_sigma_fn(self, vm):
        if isinstance(vm, (float, int)):
            self._flat_sigma = tf.Variable(float(vm), dtype=self.dtype, trainable=True)

            @tf.function(reduce_retracing=True)
            def const_sigma(x):
                return tf.fill([tf.shape(x)[0]], self._flat_sigma)

            return const_sigma

        if isinstance(vm, Sequence) and all(isinstance(v, (float, int)) for v in vm):
            self._vector_sigma = tf.Variable(list(vm), dtype=self.dtype, trainable=True)

            @tf.function(reduce_retracing=True)
            def vector_sigma(x):
                n_assets = tf.size(self._vector_sigma)
                n_rows = tf.shape(x)[0]
                n_paths = n_rows // n_assets
                tiled = tf.broadcast_to(self._vector_sigma, [n_paths, n_assets])
                return tf.reshape(tiled, [-1])

            return vector_sigma

        if hasattr(vm, "surface") and hasattr(vm, "strikes"):
            self._dupire_grid = tf.Variable(vm.surface, dtype=self.dtype, trainable=True)
            strikes, mats = vm.strikes, vm.maturities

            @tf.function(reduce_retracing=True)
            def local_sigma(x):
                t, spot = x[:, 0], x[:, 1]
                grid = self._dupire_grid.read_value()
                return ImpliedVolSurface.bilinear(t, spot, grid, strikes, mats)

            return local_sigma

        raise TypeError("vol_model must be float/int, sequence of floats or DupireLocalVol")

    def discount_factor(self, T):
        T = tf.cast(T, self.dtype)
        return tf.exp(-self.r * T)

    def sigma(self, t, spot):
        return self._sigma_fn(tf.stack([t, spot], axis=1))


class EuropeanAsset:
    """Simulates the underlying asset paths."""

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
        use_scan=False,
    ):
        """Create a new asset simulator.

        Parameters
        ----------
        S0, q : float
            Spot and dividend yield.
        T : float
            Maximum horizon for cached simulations.
        dt : float
            Time step used for the discretisation. ``n_steps`` is derived from
            ``T`` and ``dt`` as ``int(round(T / dt))``.
        n_paths : int, optional
            Number of Monte Carlo paths.
        antithetic : bool, optional
            Use antithetic sampling.
        seed : int, optional
            PRNG seed.
        dtype : tf.DType, optional
            Floating point precision.
        use_scan : bool, optional
            Whether to use ``tf.scan`` instead of ``tf.foldl`` for path
            generation.
        """

        if antithetic:
            _assert_even(n_paths, "n_paths")

        # derive number of steps from dt and ensure dt divides T exactly
        self.n_steps = int(round(float(T) / float(dt)))
        if antithetic:
            _assert_even(self.n_steps, "n_steps")

        # allow vector of initial spots
        self.S0 = tf.convert_to_tensor(S0, dtype=dtype)
        self.q = tf.constant(q, dtype=dtype)
        self.n_paths = n_paths
        self.dt = tf.constant(float(T) / self.n_steps, dtype=dtype)
        self.T = tf.constant(T, dtype=dtype)
        self.antithetic = antithetic
        self.seed = seed
        self.dtype = dtype
        self.use_scan = use_scan

        # cache for Brownian increments.  ``_cached_dW`` is a ``tf.Variable`` so
        # that tensors can be reused safely across ``tf.function`` invocations.
        # Using plain tensors would raise "out of scope" errors when the
        # compiled graphs try to access a tensor produced in a previous call.
        self._cached_dW = tf.Variable(
            tf.zeros([self.n_steps, n_paths], dtype=dtype), trainable=False
        )
        self._cache_valid = False
        self._cached_steps = 0

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _brownian(self, n_steps):
        """Generate Brownian increments for ``n_steps`` using cached ``dt``."""
        M = tf.cast(n_steps, tf.int32)
        sd = tf.sqrt(self.dt)
        Z = tf.random.stateless_normal(
            [M, self.n_paths // (2 if self.antithetic else 1)],
            [self.seed, 0],
            dtype=self.dtype,
        )
        if self.antithetic:
            Z = tf.concat([Z, -Z], axis=1)
        return Z * sd

    def simulate(self, T, market: MarketData, *, use_cache=True, save_path = False):
        """Return the asset price at maturity ``T``.

        When ``use_cache`` is ``True`` and a longer path for the same market has
        already been simulated, the relevant slice of the cached path is
        returned instead of generating a new simulation.
        """

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
        dW = dW[:, :, None]

        def step(prev, elems):
            dWi, tc = elems
            if market._flat_sigma is not None:
                sig = tf.fill(tf.shape(prev), market._flat_sigma)
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

class MCEuropeanOption:
    """Monte Carlo pricer for European options."""

    def __init__(self, asset: EuropeanAsset, market: MarketData, K, T, *, is_call=False, use_cache=True):
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
        n_paths_per_opt = tf.cast(self.asset.n_paths, tf.int32)
        n_opts = tf.cast(tf.size(self.K), tf.int32)
        id_vector = tf.repeat(tf.range(n_opts, dtype=tf.int32), n_paths_per_opt)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.asset.S0)
            if self.market._flat_sigma is not None:
                tape.watch(self.market._flat_sigma)
            elif self.market._vector_sigma is not None:
                tape.watch(self.market._vector_sigma)
            elif self.market._dupire_grid is not None:
                tape.watch(self.market._dupire_grid)

            ST = self.asset.simulate(self.T, self.market, use_cache=self.use_cache)
            ST = tf.reshape(tf.transpose(ST), [-1])
            K_paths = tf.repeat(self.K, n_paths_per_opt)
            payoff = tf.where(self.is_call, tf.nn.relu(ST - K_paths), tf.nn.relu(K_paths - ST))
            discount = self.market.discount_factor(self.T)
            sums = tf.math.unsorted_segment_sum(payoff, id_vector, n_opts)
            counts = tf.math.unsorted_segment_sum(tf.ones_like(payoff), id_vector, n_opts)
            price = discount * (sums / counts)

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

    @tf.function(jit_compile=True, reduce_retracing=True)
    def vega_bucket(self):
        if self.market._dupire_grid is None:
            raise ValueError("bucket-vega only available with DupireLocalVol")
        with tf.GradientTape() as tape:
            price = self.__call__()
        return tape.gradient(price, self.market._dupire_grid)

    def vega(self):
        if self._last_vega is None:
            _ = self.__call__()
        return self._last_vega
