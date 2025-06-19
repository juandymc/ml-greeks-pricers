import numpy as np
from dataclasses import dataclass

from ml_greeks_pricers.pricers.np.black_utils import bs_price


def _ensure_np(a, dtype):
    return np.asarray(a, dtype=dtype)


@dataclass
class ImpliedVolSurface:
    strikes: np.ndarray
    maturities: np.ndarray
    grid: np.ndarray
    t0: np.ndarray | None = None
    dtype: type = np.float64

    def __init__(self, strikes, maturities, implied_vol_surface, t0=None, dtype=np.float64):
        self.dtype = dtype
        self.strikes = _ensure_np(strikes, dtype)
        self.maturities = _ensure_np(maturities, dtype)
        self.grid = _ensure_np(implied_vol_surface, dtype)
        if t0 is None:
            self.t0 = np.zeros_like(self.maturities)
        else:
            self.t0 = _ensure_np(t0, dtype)

    @staticmethod
    def bilinear(t, k, grid, strikes, maturities):
        t = np.asarray(t, dtype=grid.dtype)
        k = np.asarray(k, dtype=grid.dtype)
        strikes = np.asarray(strikes, dtype=grid.dtype)
        maturities = np.asarray(maturities, dtype=grid.dtype)
        t_flat = t.ravel()
        k_flat = k.ravel()
        it = np.clip(np.searchsorted(maturities, t_flat, side="right") - 1, 0, len(maturities) - 2)
        ik = np.clip(np.searchsorted(strikes, k_flat, side="right") - 1, 0, len(strikes) - 2)
        t0 = maturities[it]; t1 = maturities[it + 1]
        k0 = strikes[ik]; k1 = strikes[ik + 1]
        wt = (t_flat - t0) / (t1 - t0)
        wk = (k_flat - k0) / (k1 - k0)
        g00 = grid[it, ik]
        g10 = grid[it + 1, ik]
        g01 = grid[it, ik + 1]
        g11 = grid[it + 1, ik + 1]
        result = (
            (1 - wt) * (1 - wk) * g00
            + wt * (1 - wk) * g10
            + (1 - wt) * wk * g01
            + wt * wk * g11
        )
        return result.reshape(t.shape)


class DupireLocalVol(ImpliedVolSurface):
    def __init__(self, strikes, maturities, implied_vol_surface, S0, r, q, t0=None, dtype=np.float64):
        super().__init__(strikes, maturities, implied_vol_surface, t0, dtype)
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.surface = self._compute_surface()
        self.grid = self.surface

    def _compute_surface(self):
        strikes = self.strikes
        mats = self.maturities
        iv = self.grid
        S0 = self.S0
        r = self.r
        q = self.q
        lv = np.empty((len(mats), len(strikes)), dtype=self.dtype)
        for i, T in enumerate(mats):
            for j, K in enumerate(strikes):
                def price_fn(Tv, Kv):
                    sigma = ImpliedVolSurface.bilinear(Tv, Kv, iv, strikes, mats)
                    return bs_price(S0, Kv, Tv, 0.0, r, q, sigma, True)

                eps_T = 1e-4 * max(float(T), 1.0)
                eps_K = 1e-4 * max(float(K), 1.0)
                C = price_fn(T, K)
                C_T = (price_fn(T + eps_T, K) - price_fn(max(T - eps_T, 1e-8), K)) / (2 * eps_T)
                C_K_plus = price_fn(T, K + eps_K)
                C_K_minus = price_fn(T, max(K - eps_K, 1e-8))
                C_K = (C_K_plus - C_K_minus) / (2 * eps_K)
                C_KK = (C_K_plus - 2 * C + C_K_minus) / (eps_K ** 2)
                num = C_T + (r - q) * K * C_K + q * C
                den = 0.5 * K ** 2 * C_KK
                if num > 0 and den > 0:
                    lv[i, j] = np.sqrt(num / den)
                else:
                    lv[i, j] = np.nan
        return lv

    def __call__(self):
        return self.surface
