import tensorflow as tf
tf.keras.backend.set_floatx('float64')
@tf.function
def bs_price(
    S:     tf.Tensor,
    K:     tf.Tensor,
    T:     tf.Tensor,
    t:     tf.Tensor,
    r:     tf.Tensor,
    q:     tf.Tensor,
    sigma: tf.Tensor,
    is_call: bool
) -> tf.Tensor:
    dt    = S.dtype
    half  = tf.constant(0.5, dtype=dt)
    one   = tf.constant(1.0, dtype=dt)
    two   = tf.constant(2.0, dtype=dt)

    tau       = tf.maximum(T - t, tf.constant(0.0, dtype=dt))
    sqrt_tau  = tf.sqrt(tau)
    d1 = (
        tf.math.log(S / K)
        + (r - q + half * tf.square(sigma)) * tau
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    cdf = lambda x: half * (one + tf.math.erf(x / tf.sqrt(two)))
    df_r = tf.exp(-r * tau)
    df_q = tf.exp(-q * tau)

    call = S * df_q * cdf(d1) - K * df_r * cdf(d2)
    put  = K * df_r * cdf(-d2) - S * df_q * cdf(-d1)

    return tf.where(is_call, call, put)
