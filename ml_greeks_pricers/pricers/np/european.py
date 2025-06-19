import numpy as np
from dataclasses import dataclass
from math import erf
from typing import Union

from .black_utils import bs_price
from ml_greeks_pricers.volatility import np_discrete

@dataclass
class MarketData:
    r: float
    vol_model: Union[float, "np_discrete.DupireLocalVol"]
    dtype: type = np.float64

    def __post_init__(self):
        if isinstance(self.vol_model, (float, int)):
            self.sigma = float(self.vol_model)
            self._dupire = None
        elif isinstance(self.vol_model, np_discrete.DupireLocalVol):
            self.sigma = None
            self._dupire = self.vol_model
        else:
            raise TypeError("vol_model must be float/int or DupireLocalVol")

    def discount_factor(self, T):
        T = np.asarray(T, dtype=self.dtype)
        return np.exp(-self.r * T)

    def sigma_val(self, t, spot):
        if self._dupire is None:
            return np.full_like(np.asarray(spot, dtype=self.dtype), self.sigma)
        else:
            return np_discrete.ImpliedVolSurface.bilinear(
                t,
                spot,
                self._dupire.surface,
                self._dupire.strikes,
                self._dupire.maturities,
            )


@dataclass
class EuropeanAsset:
    S0: float
    q: float
    T: float
    dt: float
    n_paths: int = 100_000
    antithetic: bool = True
    seed: int = 0
    dtype: type = np.float64

    def __post_init__(self):
        self.n_steps = int(round(self.T / self.dt))
        self.dt = float(self.T) / self.n_steps
        self.rng = np.random.default_rng(self.seed)
        self._cached_dW = None
        self._cached_steps = 0
        self._cache_valid = False

    def _brownian(self, steps):
        sd = np.sqrt(self.dt)
        Z = self.rng.standard_normal(
            (steps, self.n_paths // (2 if self.antithetic else 1))
        )
        if self.antithetic:
            Z = np.concatenate([Z, -Z], axis=1)
        return Z * sd

    def simulate(self, T, market: MarketData, use_cache=True, save_path=False):
        steps = int(round(T / self.dt))
        if use_cache and self._cache_valid and steps <= self._cached_steps:
            dW = self._cached_dW[:steps]
        else:
            dW = self._brownian(steps)
            if use_cache:
                self._cached_dW = dW
                self._cached_steps = steps
                self._cache_valid = True
        S = np.full((self.n_paths,), self.S0, dtype=self.dtype)
        for i in range(steps):
            t_cur = i * self.dt
            sig = market.sigma_val(t_cur, S)
            S *= np.exp((market.r - self.q - 0.5 * sig ** 2) * self.dt + sig * dW[i])
        if save_path:
            self.path = S
        return S


class MCEuropeanOption:
    def __init__(self, asset: EuropeanAsset, market: MarketData, K, T, *, is_call=False, use_cache=True):
        self.asset = asset
        self.market = market
        self.K = np.array(K, dtype=asset.dtype)
        self.T = T
        self.is_call = is_call
        self.use_cache = use_cache
        self._last_price = None
        self._last_delta = None
        self._last_vega = None

    def _price(self, S0=None, sigma=None):
        if S0 is not None:
            original = self.asset.S0
            self.asset.S0 = S0
        if sigma is not None and self.market.sigma is not None:
            original_sigma = self.market.sigma
            self.market.sigma = sigma
        ST = self.asset.simulate(self.T, self.market, use_cache=self.use_cache)
        payoff = np.where(self.is_call, np.maximum(ST - self.K, 0), np.maximum(self.K - ST, 0))
        price = self.market.discount_factor(self.T) * payoff.mean()
        if S0 is not None:
            self.asset.S0 = original
        if sigma is not None and self.market.sigma is not None:
            self.market.sigma = original_sigma
        return price

    def __call__(self):
        price = self._price()
        self._last_price = price
        if self.market.sigma is not None:
            self._last_delta = bs_delta(
                self.asset.S0,
                self.K,
                self.market.sigma,
                self.T,
                self.asset.q,
                self.market.r,
                self.is_call,
            )
            self._last_vega = bs_vega(
                self.asset.S0,
                self.K,
                self.market.sigma,
                self.T,
                self.asset.q,
                self.market.r,
                self.is_call,
            )
        else:
            self._last_delta = None
            self._last_vega = None
        return price

    def delta(self):
        if self._last_delta is None:
            _ = self.__call__()
        return self._last_delta

    def vega(self):
        if self._last_vega is None:
            _ = self.__call__()
        return self._last_vega


def bs_delta(S, K, sigma, T, q, r, is_call):
    tau = T
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    cdf = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0)))
    if is_call:
        return cdf
    else:
        return cdf - 1


def bs_vega(S, K, sigma, T, q, r, is_call):
    tau = T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * d1 ** 2)
    return S * np.exp(-q * tau) * np.sqrt(tau) * pdf


class AnalyticalEuropeanOption:
    def __init__(self, S, K, T, t, r, q, sigma, is_call, dtype=np.float64):
        self.S = np.asarray(S, dtype=dtype)
        self.K = np.asarray(K, dtype=dtype)
        self.T = np.asarray(T, dtype=dtype)
        self.t = np.asarray(t, dtype=dtype)
        self.r = np.asarray(r, dtype=dtype)
        self.q = np.asarray(q, dtype=dtype)
        self.sigma = np.asarray(sigma, dtype=dtype)
        self.is_call = is_call
        self.dtype = dtype

    def price(self):
        return bs_price(self.S, self.K, self.T, self.t, self.r, self.q, self.sigma, self.is_call)

    __call__ = price

    def delta(self):
        return bs_delta(self.S, self.K, self.sigma, self.T - self.t, self.q, self.r, self.is_call)

    def vega(self):
        return bs_vega(self.S, self.K, self.sigma, self.T - self.t, self.q, self.r, self.is_call)
