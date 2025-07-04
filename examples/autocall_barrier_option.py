#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from pathlib import Path
import pandas as pd           # mantenido por coherencia con otros ejemplos

from ml_greeks_pricers.volatility.discrete import DupireLocalVol
from ml_greeks_pricers.pricers.tf.european import MarketData
from ml_greeks_pricers.pricers.tf.american import AmericanAsset

tf.keras.backend.set_floatx("float64")
dtype = tf.float64
# --------------------------------------------------------------------------- #
class MCAutocallBarrierOption:
    """
    Autocall (1 subyacente):

      – Obs. times t_k, barrera_k, cupón_k.
      – Si S(t_k) ≥ barrera_k  ⇒  paga cupón_k descontado y se cancela.
      – Si nunca se dispara  ⇒  Put europeo al vencimiento.

    Vectorizada: sin bucles sobre paths ni sobre fechas.
    """

    def __init__(self,
                 asset:  AmericanAsset,
                 market: MarketData,
                 K: float,
                 T: float,
                 obs_times,          # list/array floats (años)
                 barrier_levels,     # list/array
                 coupons,            # list/array
                 *,
                 is_put=True,
                 use_cache=True):

        self.asset      = asset
        self.market     = market
        self.K          = tf.constant(K, dtype=asset.dtype)
        self.T          = tf.constant(T, dtype=asset.dtype)
        self.is_put     = bool(is_put)
        self.use_cache  = use_cache

        # vectores de longitud m
        self.obs_t   = tf.constant(obs_times,      dtype=asset.dtype)
        self.barrier = tf.constant(barrier_levels, dtype=asset.dtype)
        self.coupon  = tf.constant(coupons,        dtype=asset.dtype)
        self.m       = tf.size(self.obs_t)

        # índices enteros en la malla temporal
        self.obs_idx = tf.cast(tf.round(self.obs_t / asset.dt), tf.int32)

        # caches último cálculo
        self._last_price = self._last_delta = self._last_vega = None

    # --------------------------------------------------------------------- #
    @tf.function(reduce_retracing=True)
    def _price_core(self):
        """
        Devuelve E[PV] – totalmente vectorizado y compatible con Autograph.
        """
        # 1) Simula trayectoria completa  [steps+1, n_paths]
        path = self.asset.simulate(self.T, self.market,
                                   use_cache=self.use_cache)
        n_paths = tf.shape(path)[1]

        # 2) Precios en todas las fechas de observación  -> [m, n_paths]
        spots = tf.gather(path, self.obs_idx)                   # gather filas

        # 3) Matriz de disparo de la barrera
        #    hits[k, p] = True si path p se autocalla en obs k
        hits = spots >= self.barrier[:, None]

        # 4) Para cada path: ¿se disparó alguna vez?
        hit_any = tf.reduce_any(hits, axis=0)                  # [n_paths] bool

        # 5) Primer índice k donde se dispara
        #    tf.argmax sobre int32 devuelve el primer índice con valor 1
        first_hit_idx = tf.argmax(tf.cast(hits, tf.int32), axis=0,
                                   output_type=tf.int32)       # [n_paths]

        # 6) Cupón descontado por observación
        df_vec      = tf.exp(-self.market.r * self.obs_t)      # [m]
        coupon_disc = self.coupon * df_vec                     # [m]

        # 7) Valor del cupón por path (0 si nunca se disparó)
        pv_coupon = tf.where(hit_any,
                             tf.gather(coupon_disc, first_hit_idx),
                             tf.zeros_like(tf.cast(hit_any, dtype=self.asset.dtype)))

        # 8) Payoff residual en T para paths que no se autocall
        ST = path[-1]                                          # [n_paths]
        residual = tf.where(self.is_put,
                            tf.nn.relu(self.K - ST),
                            tf.nn.relu(ST - self.K))
        pv_terminal = tf.where(~hit_any,
                               residual * tf.exp(-self.market.r * self.T),
                               tf.zeros_like(residual))

        # 9) Valor presente por path y media
        pv_total = pv_coupon + pv_terminal
        return tf.reduce_mean(pv_total)

    # ---------------- precio + Greeks ------------------------------------- #
    def _price_and_grads(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.asset.S0)                       # delta
            if self.market._flat_sigma is not None:         # vega plana
                tape.watch(self.market._flat_sigma)
            elif self.market._dupire_grid is not None:      # vega Dupire
                tape.watch(self.market._dupire_grid)

            price = self._price_core()

        delta = tape.gradient(price, self.asset.S0)
        if delta is None:
            delta = tf.constant(0.0, dtype=self.asset.dtype)

        if self.market._flat_sigma is not None:
            vega = tape.gradient(price, self.market._flat_sigma)
            vega = tf.constant(0.0, dtype=self.asset.dtype) if vega is None else vega
        else:
            grid_grad = tape.gradient(price, self.market._dupire_grid)
            vega = tf.reduce_sum(grid_grad) if grid_grad is not None else tf.constant(0.0, dtype=self.asset.dtype)

        return price, delta, vega

    # ---------------- API pública ------------------------------------------ #
    def price(self):
        p, d, v = self._price_and_grads()
        self._last_price, self._last_delta, self._last_vega = p, d, v
        return p

    def delta(self):
        return self._last_delta if self._last_delta is not None else self._price_and_grads()[1]

    def vega(self):
        return self._last_vega if self._last_vega is not None else self._price_and_grads()[2]
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ---------------- parámetros demo -------------------------------------- #
    S0       = 100.0
    K        = 90.0
    r        = 0.03
    q        = 0.0
    T        = 2.0
    n_paths  = 200_000
    n_steps  = 60
    seed     = 42
    dt       = T / n_steps
    sigma    = 0.22

    obs_times      = [0.5, 1.0, 1.5]
    barrier_levels = [110., 110., 110.]
    coupons        = [3.0, 6.0, 9.0]

    # ---------------- objetos de mercado / simulación ---------------------- #
    market = MarketData(r, sigma)
    asset  = AmericanAsset(S0, q,
                           T=T, dt=dt,
                           n_paths=n_paths,
                           antithetic=True,
                           seed=seed)

    option = MCAutocallBarrierOption(asset, market,
                                     K, T,
                                     obs_times, barrier_levels, coupons,
                                     is_put=True)

    # ---------------- resultado ------------------------------------------- #
    tf.print("Autocall 1‑activo  |  paths:", n_paths)
    tf.print("Precio :", option.price())
    tf.print("Delta  :", option.delta())
    tf.print("Vega   :", option.vega())
