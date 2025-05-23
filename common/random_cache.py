from __future__ import annotations

import tensorflow as tf
from typing import Dict, Tuple

# Cache for stateless random normals. Keys are (shape_tuple, seed, dtype_name)
_RANDOM_CACHE: Dict[Tuple[Tuple[int, ...], int, str], tf.Tensor] = {}


def cached_normal(shape: tuple[int, ...], seed: int, dtype: tf.dtypes.DType = tf.float64) -> tf.Tensor:
    """Return a cached tensor of stateless normal samples."""
    key = (tuple(shape), int(seed), tf.as_dtype(dtype).name)
    tensor = _RANDOM_CACHE.get(key)
    if tensor is None:
        tensor = tf.random.stateless_normal(shape, [seed, 0], dtype=dtype)
        _RANDOM_CACHE[key] = tensor
    return tensor


def mc_noise(
    n_steps: int,
    n_paths: int,
    seed: int,
    *,
    antithetic: bool = False,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return cached random normals for Monte Carlo simulations."""
    base_n = n_paths // 2 if antithetic else n_paths
    if n_steps > 1:
        base_shape = (n_steps, base_n)
        noise = cached_normal(base_shape, seed, dtype)
        if antithetic:
            noise = tf.concat([noise, -noise], axis=1)
    else:
        base_shape = (base_n,)
        noise = cached_normal(base_shape, seed, dtype)
        if antithetic:
            noise = tf.concat([noise, -noise], axis=0)
    return noise
