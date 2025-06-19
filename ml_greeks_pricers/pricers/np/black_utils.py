import numpy as np
from math import erf, sqrt, log, exp


def bs_price(S, K, T, t, r, q, sigma, is_call):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tau = np.maximum(T - t, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    def cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)

    call = S * df_q * cdf(d1) - K * df_r * cdf(d2)
    put = K * df_r * cdf(-d2) - S * df_q * cdf(-d1)

    return np.where(is_call, call, put)
