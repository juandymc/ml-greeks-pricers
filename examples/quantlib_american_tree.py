import QuantLib as ql

# Parámetros de mercado
spot, K, r, q, sigma, T = 110., 90.0, 0.06, 0.0, 0.212, 0.5
calendar    = ql.NullCalendar()
daycount    = ql.Actual365Fixed()
settlement  = calendar.adjust(ql.Date.todaysDate())
maturity    = settlement + int(T*365)

# Market data handles
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
flat_ts     = ql.YieldTermStructureHandle(
                  ql.FlatForward(settlement, r, daycount))
div_ts      = ql.YieldTermStructureHandle(
                  ql.FlatForward(settlement, q, daycount))
vol_ts      = ql.BlackVolTermStructureHandle(
                  ql.BlackConstantVol(settlement, calendar, sigma, daycount))

# Payoff & American exercise
payoff   = ql.PlainVanillaPayoff(ql.Option.Put, K)
exercise = ql.AmericanExercise(settlement, maturity)

process = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, vol_ts)
engine  = ql.BinomialVanillaEngine(process, "crr", 800)

# Construye la opción
american = ql.VanillaOption(payoff, exercise)
american.setPricingEngine(engine)

# Precio y griegas
price = american.NPV()
delta = american.delta()
gamma = american.gamma()
theta = american.theta()


print(f"American Put CRR:")
print(f"  Price = {price:.6f}")
print(f"  Delta = {delta:.6f}")
print(f"  Gamma = {gamma:.6f}")
print(f"  Theta = {theta:.6f}")

