from ml_greeks_pricers.pricers.tf.black_utils import bs_price
from ml_greeks_pricers.common.constants import USE_XLA

import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')

DTYPE = tf.float64 

class ImpliedVolSurface:
    def __init__(self, strikes, maturities, implied_vol_surface, t0=None, dtype=tf.float64):
        self.strikes    = tf.convert_to_tensor(strikes,             dtype=dtype)
        self.maturities = tf.convert_to_tensor(maturities,          dtype=dtype)
        self.grid       = tf.convert_to_tensor(implied_vol_surface, dtype=dtype)
        self.t0         = tf.zeros_like(self.maturities) if t0 is None \
                            else tf.convert_to_tensor(t0, dtype=dtype)

    @staticmethod
    @tf.function(jit_compile=False, reduce_retracing=True)
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
    @staticmethod
    def bilinear_var(Tq, Kq, iv_grid, strikes, maturities):
        """Bilinear sobre la varianza total w = σ²·T."""
        T = tf.constant(maturities, dtype=iv_grid.dtype)
        K = tf.constant(strikes,   dtype=iv_grid.dtype)
        var_grid = iv_grid**2 * T[:, None]          # (M,N)
        #  -> reutilizamos el bilinear escalar que ya tienes:
        return ImpliedVolSurface.bilinear(Tq, Kq, var_grid, strikes, maturities)

def _cubic(P, t):
    # Hermite base: p1 + t*(p2-p0)/2 + t²*(p0-2.5p1+2p2-0.5p3)
    a =  P[...,0] - 2.5*P[...,1] + 2.0*P[...,2] - 0.5*P[...,3]
    b = (P[...,2] - P[...,0]) * 0.5
    d =  P[...,1]
    return ((a*t + b)*t + d)          # (sin término t³ porque 2ª deriv = 0)

def bicubic_catmull(grid, yq, xq):
    M = tf.shape(grid)[0]; N = tf.shape(grid)[1]
    yi = tf.clip_by_value(tf.cast(tf.floor(yq), tf.int32), 0, M-1)
    xi = tf.clip_by_value(tf.cast(tf.floor(xq), tf.int32), 0, N-1)
    dy = tf.cast(yq - tf.floor(yq), DTYPE)
    dx = tf.cast(xq - tf.floor(xq), DTYPE)

    def nb(i, nmax):       # vecinos 4
        return tf.clip_by_value(tf.stack([i-1,i,i+1,i+2], -1), 0, nmax-1)
    ix = nb(xi, N);  iy = nb(yi, M)

    rows  = tf.gather(grid, iy)                         # (Q,4,N)
    patch = tf.gather(rows, ix, axis=2, batch_dims=1)   # (Q,4,4)

    col  = _cubic(patch, tf.tile(dx[:, None], [1,4]))   # interp. x
    val  = _cubic(col, dy)                              # interp. y
    return val
    
def w_interp_hybrid(Tq, Kq, iv_grid, strikes, maturities):
    """
    Devuelve w(T,K) con:
        • bilineal en celdas que tocan el borde
        • bicúbica Catmull‑Rom en el interior
    Devolución 100 % TensorFlow–autodiff‑safe.
    """
    M = len(maturities)
    N = len(strikes)

    # escala física -> índice
    y = (Tq - maturities[0]) / (maturities[-1] - maturities[0]) * (M - 1)
    x = (Kq - strikes[0])    / (strikes[-1]   - strikes[0])    * (N - 1)

    # máscara de “borde”
    mask_edge = (y < 1.0) | (y > (M - 2)) | (x < 1.0) | (x > (N - 2))

    # -- bilineal varianza (ya implementada en paso 1)
    w_lin = ImpliedVolSurface.bilinear_var(Tq, Kq, iv_grid, strikes, maturities)

    # -- bicúbica interior (la Catmull‑Rom que pasé antes)
    w_cubic = bicubic_catmull(
        tf.constant(iv_grid**2 * maturities[:, None], dtype=Tq.dtype),
        y, x)

    # combínalos
    return tf.where(mask_edge, w_lin, w_cubic)

class DupireLocalVol(ImpliedVolSurface):
    def __init__(self, strikes, maturities, implied_vol_surface,
                 S0, r, q, t0=None, dtype=tf.float64, backend="tf"):
        super().__init__(strikes, maturities, implied_vol_surface, t0, dtype)
        self.S0 = tf.convert_to_tensor(S0, dtype=dtype)
        self.r  = tf.convert_to_tensor(r,  dtype=dtype)
        self.q  = tf.convert_to_tensor(q,  dtype=dtype)
        self.backend = backend.lower()          #  "tf"  ó  "ql"
        # compute the local volatility and store it in ``self.surface``,
        # but also override ``self.grid`` for future interpolation
        self.surface = self._compute_surface()
        self.grid    = self.surface
        
    def _compute_surface_ql(self):
        # ----------  importar QuantLib de forma segura -------------------
        try:
            import QuantLib as ql
        except ImportError:                    # fallback claro
            raise RuntimeError("backend='ql' requiere 'QuantLib-Python' instalado.")
    
        strikes_np = self.strikes.numpy()
        mats_np    = self.maturities.numpy()
        iv_np      = self.grid.numpy()
    
        # ----------  Ajustes de mercado ----------------------------------
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        rf = ql.FlatForward(today, float(self.r), dc)
        div= ql.FlatForward(today, float(self.q), dc)
        calendar = ql.NullCalendar()
    
        # ----------  BlackVarianceSurface --------------------------------
        dates = [today + int(t*365+0.5) for t in mats_np]
        rows, cols = len(strikes_np), len(dates)
        vol_mat = ql.Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                vol_mat[i][j] = iv_np[j, i]      # Transposición
    
        bvs = ql.BlackVarianceSurface(today, calendar, dates,
                                      strikes_np.tolist(), vol_mat,
                                      dc, True, True)
        bvs.setInterpolation("bicubic")
    
        locvol = ql.LocalVolSurface(
            ql.BlackVolTermStructureHandle(bvs),
            ql.YieldTermStructureHandle(rf),
            ql.YieldTermStructureHandle(div),
            float(self.S0)
        )
    
        # ----------  Evaluar en la malla original ------------------------
        sigma_local = np.empty_like(iv_np)
        for j, T in enumerate(mats_np):
            for i, K in enumerate(strikes_np):
                sigma_local[j, i] = locvol.localVol(float(T), float(K), True)
    
        return tf.constant(sigma_local, dtype=self.strikes.dtype)


    #@tf.function(jit_compile=False, reduce_retracing=True)
    def _compute_surface(self):
        if self.backend == "ql":
            return self._compute_surface_ql()   # ← usa QuantLib y termina
        else:
            strikes, mats, iv = self.strikes, self.maturities, self.grid
            S0, r, q          = self.S0, self.r, self.q
        
            Tg, Kg = tf.meshgrid(mats, strikes, indexing='ij')
            Tf = tf.reshape(Tg, [-1]);        Kf = tf.reshape(Kg, [-1])
            dt_scalar = strikes.dtype
        
            # ----------  precio Black Scholes con σ = √(w/T)  -----------------
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([Tf, Kf])
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch([Tf, Kf])
        
                    w_flat     = w_interp_hybrid(Tf, Kf, iv, strikes, mats)
                    sigma_flat = tf.sqrt(w_flat / Tf)
        
                    C = bs_price(    # tu función analítica
                        S = S0 * tf.ones_like(Kf),
                        K = Kf,
                        T = Tf,
                        t = tf.zeros_like(Kf),
                        r = r * tf.ones_like(Kf),
                        q = q * tf.ones_like(Kf),
                        sigma = sigma_flat,
                        is_call = True
                    )
        
                C_T = tape1.gradient(C, Tf)
                C_K = tape1.gradient(C, Kf)
            C_KK = tape2.gradient(C_K, Kf)
        
            num = C_T + (r - q) * Kf * C_K + q * C
            den = 0.5 * Kf**2 * C_KK
        
            valid = (num > 0) & (den > 0)
            sigma_sq = tf.where(valid, num / den, tf.constant(0.0, dt_scalar))
            lv = tf.sqrt(sigma_sq)               # √0 = 0 en los nodos “inválidos”
            return tf.reshape(lv, tf.shape(Tg))


    def __call__(self):
        return self.surface
