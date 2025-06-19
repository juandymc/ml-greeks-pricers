import numpy as np
from dataclasses import dataclass
from .european import EuropeanAsset, MarketData


class AmericanAsset(EuropeanAsset):
    def simulate(self, T, market: MarketData, use_cache=True):
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
        path = [S.copy()]
        for t in range(steps):
            sig = market.sigma
            S = S * np.exp((market.r - self.q - 0.5 * sig ** 2) * self.dt + sig * dW[t])
            path.append(S.copy())
        return np.array(path)


class MCAmericanOption:
    def __init__(self, asset: AmericanAsset, market: MarketData, K, T, *, is_call=False, use_cache=True):
        self.asset = asset
        self.market = market
        self.K = K
        self.T = T
        self.is_call = is_call
        self.use_cache = use_cache
        self._last_price = None
        self._last_delta = None
        self._last_vega = None

    def _lsm_price(self, S0=None, sigma=None):
        if S0 is not None:
            orig_S0 = self.asset.S0
            self.asset.S0 = S0
        if sigma is not None:
            orig_sigma = self.market.sigma
            self.market.sigma = sigma
        paths = self.asset.simulate(self.T, self.market, use_cache=self.use_cache)
        payoff = np.where(self.is_call, np.maximum(paths - self.K, 0.0), np.maximum(self.K - paths, 0.0))
        CF = payoff[-1]
        df = np.exp(-self.market.r * self.asset.dt)
        n_steps = paths.shape[0] - 1
        eye = np.eye(3)
        for t in range(n_steps - 1, 0, -1):
            discounted = CF * df
            St = paths[t]
            itm = payoff[t] > 0
            idx = np.where(itm)[0]
            if len(idx) == 0:
                CF = discounted
                continue
            St_i = St[idx]
            CF_i = discounted[idx]
            X = np.vstack([np.ones_like(St_i), St_i, St_i ** 2]).T
            beta = np.linalg.lstsq(X, CF_i, rcond=None)[0]
            cont = beta[0] + beta[1] * St + beta[2] * St ** 2
            exercise = itm & (payoff[t] > cont)
            CF = np.where(exercise, payoff[t], discounted)
        price = np.mean(CF * df)
        if S0 is not None:
            self.asset.S0 = orig_S0
        if sigma is not None:
            self.market.sigma = orig_sigma
        return price

    def __call__(self):
        price = self._lsm_price()
        self._last_price = price
        return price

    def delta(self, eps=1e-4):
        if self._last_delta is None:
            price_up = self._lsm_price(S0=self.asset.S0 + eps)
            price_down = self._lsm_price(S0=self.asset.S0 - eps)
            self._last_delta = (price_up - price_down) / (2 * eps)
        return self._last_delta

    def vega(self, eps=1e-4):
        if self._last_vega is None:
            price_up = self._lsm_price(sigma=self.market.sigma + eps)
            price_down = self._lsm_price(sigma=self.market.sigma - eps)
            self._last_vega = (price_up - price_down) / (2 * eps)
        return self._last_vega
