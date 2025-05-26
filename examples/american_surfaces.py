# american_surfaces.py

import os
import time
from pathlib import Path

import tensorflow as tf
import pandas as pd

from ml_greeks_pricers.volatility.discrete import DupireLocalVol
from ml_greeks_pricers.common.constants import USE_XLA
from ml_greeks_pricers.pricers.american import MCAmericanOption  # tu módulo american personalizado

# ------------------------------
# Supresión de logs innecesarios
# ------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')

# ------------------------------
# Configuración de precisión y JIT
# ------------------------------
tf.keras.backend.set_floatx('float64')
tf.config.optimizer.set_jit(True)   # XLA JIT

# ------------------------------
# Parámetros de mercado y simulación
# ------------------------------
S0, r, q = 110.0, 0.06, 0.0        # Spot, tasa libre de riesgo, dividendo
iv_vol    = 0.212                  # Volatilidad plana (flat)
n_paths   = 100_000
n_steps   = 60
antithetic = True
dtype     = tf.float64
seed      = 42

# ------------------------------
# Definición de strikes, maturities e IV surface
# ------------------------------
strikes = [60, 70, 80, 90, 100, 110, 120, 130, 140]
mats    = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

iv_matrix = [
    [0.248,0.233,0.220,0.209,0.200,0.193,0.188,0.185,0.184],
    [0.251,0.236,0.223,0.212,0.203,0.196,0.191,0.188,0.187],
    [0.254,0.239,0.226,0.215,0.206,0.199,0.194,0.191,0.190],
    [0.257,0.242,0.229,0.218,0.209,0.202,0.197,0.194,0.193],
    [0.260,0.245,0.232,0.221,0.212,0.205,0.200,0.197,0.196],
    [0.263,0.248,0.235,0.224,0.215,0.208,0.203,0.200,0.199],
    [0.266,0.251,0.238,0.227,0.218,0.211,0.206,0.203,0.202],
    [0.269,0.254,0.241,0.230,0.221,0.214,0.209,0.206,0.205],
]
# Construimos el modelo de volatilidad Dupire usando argumentos posicionales
dupire_vol = DupireLocalVol(strikes, mats, iv_matrix, S0, r, q)

# ------------------------------
# Directorio para guardar las superficies en CSV
# ------------------------------
csv_dir = Path(__file__).with_name('american_surfaces')
csv_dir.mkdir(exist_ok=True)

# ------------------------------
# Funciones compiladas con tf.function
# ------------------------------
@tf.function(input_signature=[
    tf.TensorSpec([], dtype),
    tf.TensorSpec([], dtype),
])
def price_delta_vega_flat(K, T):
    opt = MCAmericanOption(
        S0, K, T, r, q, iv_vol,
        n_paths=n_paths, n_steps=n_steps,
        antithetic=antithetic, dtype=dtype, seed=seed
    )
    p, d, v = opt.price_and_grads()
    return p, d, v

@tf.function(input_signature=[
    tf.TensorSpec([], dtype),
    tf.TensorSpec([], dtype),
])
def price_delta_vega_dup(K, T):
    opt = MCAmericanOption(
        S0, K, T, r, q, dupire_vol,
        n_paths=n_paths, n_steps=n_steps,
        antithetic=antithetic, dtype=dtype, seed=seed
    )
    p, d, v = opt.price_and_grads()
    return p, d, v

# Warm-up para compilar ambos grafos una sola vez
_ = price_delta_vega_flat(
    tf.constant(strikes[0], dtype=dtype),
    tf.constant(mats[0],    dtype=dtype)
)
_ = price_delta_vega_dup(
    tf.constant(strikes[0], dtype=dtype),
    tf.constant(mats[0],    dtype=dtype)
)

# ------------------------------
# Función para generar superficies
# ------------------------------
def surface(price_func):
    price_rows, delta_rows, vega_rows = [], [], []
    for T in mats:
        pr, de, ve = [], [], []
        for K in strikes:
            p, d, v = price_func(
                tf.constant(K, dtype=dtype),
                tf.constant(T, dtype=dtype)
            )
            pr.append(p); de.append(d); ve.append(v)
        price_rows.append(pr)
        delta_rows.append(de)
        vega_rows.append(ve)
    idx, cols = mats, strikes
    return (
        pd.DataFrame(price_rows, index=idx, columns=cols),
        pd.DataFrame(delta_rows, index=idx, columns=cols),
        pd.DataFrame(vega_rows,  index=idx, columns=cols),
    )

# ------------------------------
# Medición y guardado de resultados
# ------------------------------
def measure_and_save(prefix, price_func):
    start = time.perf_counter()
    prices, deltas, vegas = surface(price_func)
    elapsed = time.perf_counter() - start
    print(f"{prefix}: {elapsed:.2f}s")
    prices.to_csv(csv_dir / f"{prefix}_prices.csv")
    deltas.to_csv(csv_dir / f"{prefix}_deltas.csv")
    vegas.to_csv(csv_dir / f"{prefix}_vegas.csv")
    return prices, deltas, vegas

# ------------------------------
# Bloque principal
# ------------------------------
if __name__ == "__main__":
    # Superficie con volatilidad plana
    prices_flat, deltas_flat, vegas_flat = measure_and_save(
        "flat", price_delta_vega_flat
    )

    # Superficie con volatilidad Dupire
    prices_dup, deltas_dup, vegas_dup = measure_and_save(
        "dupire", price_delta_vega_dup
    )

    # Mostrar resúmenes
    print("\nFlat MC Prices:\n", prices_flat)
    print("\nDupire MC Prices:\n", prices_dup)
    print("\nFlat MC Delta:\n", deltas_flat)
    print("\nDupire MC Delta:\n", deltas_dup)
    print("\nFlat MC Vega:\n", vegas_flat)
    print("\nDupire MC Vega:\n", vegas_dup)
