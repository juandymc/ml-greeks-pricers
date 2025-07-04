import QuantLib as ql
import numpy as np

# ---------- parámetros (mismos que tu TF) ----------
S0, r, q, sigma = 100.0, 0.03, 0.0, 0.22
obs_times = [0.5, 1.0, 1.5]
barriers  = [110., 110., 110.]
coupons   = [3.0,  6.0,  9.0 ]
K, T      = 90.0,  2.0

steps   = 48
samples = 200_000
seed    = 42
dt_mc   = T / steps

# ---------- curvas y proceso ----------
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
dayc  = ql.Actual365Fixed()
spot  = ql.SimpleQuote(S0)
rf    = ql.FlatForward(today, r, dayc)
div   = ql.FlatForward(today, q, dayc)
vol   = ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dayc)
process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot),
            ql.YieldTermStructureHandle(div),
            ql.YieldTermStructureHandle(rf),
            ql.BlackVolTermStructureHandle(vol))

# ---------- generador de caminos ----------
rng    = ql.GaussianRandomSequenceGenerator(
           ql.UniformRandomSequenceGenerator(steps, ql.UniformRandomGenerator(seed)))
seqgen = ql.GaussianPathGenerator(process, T, steps, rng, False)

# ---------- índices de observación ----------
obs_idx = [int(round(t / dt_mc)) for t in obs_times]

# ---------- función de payoff ----------
def autocall_payoff(path):
    for k, idx in enumerate(obs_idx):
        if path[idx] >= barriers[k]:
            return coupons[k] * np.exp(-r * obs_times[k])
    ST = path.back()
    return max(K - ST, 0.0) * np.exp(-r * T)

# ---------- Monte Carlo ----------
pv_samples = np.fromiter(
    (autocall_payoff(seqgen.next().value()) for _ in range(samples)),
    dtype=float, count=samples)

price_q  = pv_samples.mean()
stderr_q = pv_samples.std(ddof=1)/np.sqrt(samples)
print(f"QuantLib MC price = {price_q:.4f}  ± {2*stderr_q:.4f}")
