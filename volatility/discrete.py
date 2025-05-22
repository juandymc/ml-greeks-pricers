from pricers.black_utils import bs_price

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from common import USE_XLA
class ImpliedVolSurface:
    def __init__(self, strikes, maturities, implied_vol_surface, t0=None, dtype=tf.float64):
        self.strikes    = tf.convert_to_tensor(strikes,             dtype=dtype)
        self.maturities = tf.convert_to_tensor(maturities,          dtype=dtype)
        self.grid       = tf.convert_to_tensor(implied_vol_surface, dtype=dtype)
        self.t0         = tf.zeros_like(self.maturities) if t0 is None \
                            else tf.convert_to_tensor(t0, dtype=dtype)

    @staticmethod
    @tf.function(jit_compile=USE_XLA)
    def bilinear(t, k, grid, strikes, maturities):
        dt = strikes.dtype
        t = tf.cast(t, dt); k = tf.cast(k, dt)
        Nm = tf.shape(maturities)[0]; Nk = tf.shape(strikes)[0]
        it = tf.clip_by_value(tf.searchsorted(maturities, t, 'right') - 1, 0, Nm - 2)
        ik = tf.clip_by_value(tf.searchsorted(strikes,    k, 'right') - 1, 0, Nk - 2)
        t0 = tf.gather(maturities, it);   t1 = tf.gather(maturities, it + 1)
        k0 = tf.gather(strikes,    ik);   k1 = tf.gather(strikes,    ik + 1)
        wt = (t - t0) / (t1 - t0);         wk = (k - k0) / (k1 - k0)
        idx00 = tf.stack([it,     ik    ], axis=-1)
        idx10 = tf.stack([it + 1, ik    ], axis=-1)
        idx01 = tf.stack([it,     ik + 1], axis=-1)
        idx11 = tf.stack([it + 1, ik + 1], axis=-1)
        g00 = tf.gather_nd(grid, idx00)
        g10 = tf.gather_nd(grid, idx10)
        g01 = tf.gather_nd(grid, idx01)
        g11 = tf.gather_nd(grid, idx11)
        return ((1 - wt) * (1 - wk) * g00 +
                wt       * (1 - wk) * g10 +
                (1 - wt) * wk       * g01 +
                wt       * wk       * g11)
class DupireLocalVol(ImpliedVolSurface):
    def __init__(self, strikes, maturities, implied_vol_surface,
                 S0, r, q, t0=None, dtype=tf.float64):
        super().__init__(strikes, maturities, implied_vol_surface, t0, dtype)
        self.S0 = tf.convert_to_tensor(S0, dtype=dtype)
        self.r  = tf.convert_to_tensor(r,  dtype=dtype)
        self.q  = tf.convert_to_tensor(q,  dtype=dtype)
        # compute the local volatility and store it in ``self.surface``,
        # but also override ``self.grid`` for future interpolation
        self.surface = self._compute_surface()
        self.grid    = self.surface

    @tf.function(jit_compile=USE_XLA)
    def _compute_surface(self):
        strikes, mats, iv = self.strikes, self.maturities, self.grid
        S0, r, q         = self.S0, self.r, self.q
        Tg, Kg           = tf.meshgrid(mats, strikes, indexing='ij')
        Tf = tf.reshape(Tg, [-1]);   Kf = tf.reshape(Kg, [-1])

        # typed constants for the CDF
        dt_scalar = strikes.dtype
        half  = tf.constant(0.5, dt_scalar)
        one   = tf.constant(1.0, dt_scalar)
        sqrt2 = tf.sqrt(tf.constant(2.0, dt_scalar))

        def price_fn():
            sigma_flat = ImpliedVolSurface.bilinear(Tf, Kf, iv, strikes, mats)
            # build broadcast-ready tensors:
            return bs_price(
                S     = S0 * tf.ones_like(Kf),
                K     = Kf,
                T     = Tf,
                t     = tf.zeros_like(Kf),
                r     = r * tf.ones_like(Kf),
                q     = q * tf.ones_like(Kf),
                sigma = sigma_flat,
                is_call=True
            )

        C   = price_fn()
        # first- and second-order derivatives via ``GradientTape``:
        C_T  = tf.gradients(C, Tf)[0]
        C_K  = tf.gradients(C, Kf)[0]
        C_KK = tf.gradients(C_K, Kf)[0]

        num = C_T + (r - q) * Kf * C_K + q * C
        den = 0.5 * Kf**2 * C_KK
        lv  = tf.where((num > 0) & (den > 0),
                       tf.sqrt(num / den),
                       tf.fill(tf.shape(num), tf.constant(float('nan'), dt_scalar)))
        return tf.reshape(lv, tf.shape(Tg))

    def __call__(self):
        return self.surface
