# dupire_mc_numpy.py  ── Monte Carlo explícito bajo Dupire (con extrapolación)
import QuantLib as ql
import numpy as np

# ----------------------- parámetros de entrada -------------------------
S0, K, T = 110.0, 90.0, 0.50          # spot, strike, maturity (años)
r, q     = 0.06, 0.00                 # tipos continuo y dividendo
n_paths, n_steps = 500_000, 50
seed = 42
rng  = np.random.default_rng()

strikes    = [60,70,80,90,100,110,120,130,140]          # 9 strikes
maturities = [0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]  # 8

iv_matrix = [                                           # 8×9  (T × K)
    [0.248,0.233,0.220,0.209,0.200,0.193,0.188,0.185,0.184],
    [0.251,0.236,0.223,0.212,0.203,0.196,0.191,0.188,0.187],
    [0.254,0.239,0.226,0.215,0.206,0.199,0.194,0.191,0.190],
    [0.257,0.242,0.229,0.218,0.209,0.202,0.197,0.194,0.193],
    [0.260,0.245,0.232,0.221,0.212,0.205,0.200,0.197,0.196],
    [0.263,0.248,0.235,0.224,0.215,0.208,0.203,0.200,0.199],
    [0.266,0.251,0.238,0.227,0.218,0.211,0.206,0.203,0.202],
    [0.269,0.254,0.241,0.230,0.221,0.214,0.209,0.206,0.205],
]

# ----------------------- entorno QuantLib --------------------------------
cal   = ql.UnitedStates(ql.UnitedStates.NYSE)
today = ql.Date.todaysDate();  ql.Settings.instance().evaluationDate = today
dc    = ql.Actual365Fixed()

mat_dates = [cal.advance(today, ql.Period(int(m*12), ql.Months))
             for m in maturities]

# VolMatrix: filas=strikes, cols=fechas
rows, cols = len(strikes), len(mat_dates)
vol_matrix = ql.Matrix(rows, cols)
for i_k in range(rows):
    for j_t in range(cols):
        vol_matrix[i_k][j_t] = iv_matrix[j_t][i_k]

black_surf = ql.BlackVarianceSurface(today, cal,
                                     mat_dates, strikes,
                                     vol_matrix, dc)
black_surf.setInterpolation("bilinear")        # interp-intra, extrap-flat
black_surf.enableExtrapolation()               # extrapolar fuera del grid

r_ts = ql.FlatForward(today, r, dc)
q_ts = ql.FlatForward(today, q, dc)

local_surf = ql.LocalVolSurface(
    ql.BlackVolTermStructureHandle(black_surf),
    ql.YieldTermStructureHandle(r_ts),
    ql.YieldTermStructureHandle(q_ts),
    S0
)
local_surf.enableExtrapolation()               # extrapola también local vol

# --------------- Monte Carlo explícito (Euler) --------------------------
dt       = T / n_steps
disc     = np.exp(-r * T)
sqrt_dt  = np.sqrt(dt)

S = np.full(n_paths, S0)

for step in range(1, n_steps + 1):
    t = step * dt
    Z = rng.standard_normal(n_paths)
    # σ_local(t, S) con extrapolación fuera de la malla
    sig = np.array([local_surf.localVol(t, float(s), True) for s in S])
    S *= np.exp((r - 0.5 * sig**2) * dt + sig * sqrt_dt * Z)

payoffs = np.maximum(K - S, 0.0)
price_mc = disc * payoffs.mean()

print(f"Precio Monte Carlo (Dupire, {n_paths} paths, {n_steps} pasos) = "
      f"{price_mc:.6f}")
