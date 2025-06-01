import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def vanilla_net(input_dim: int, hidden_units: int, hidden_layers: int, output_dim: int = 1) -> tf.keras.Model:
    """Return a simple feed-forward network used for option pricing."""
    net = tf.keras.Sequential()
    net.add(Input((input_dim,)))
    for _ in range(hidden_layers):
        net.add(Dense(hidden_units, activation="softplus"))
    net.add(Dense(output_dim))
    return net


@tf.keras.utils.register_keras_serializable(package="ml_greeks_pricers")
class TwinNetwork(tf.keras.Model):
    """Model returning the value and its first derivatives."""

    def __init__(self, vanilla: tf.keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vanilla = vanilla

    def call(self, x: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.vanilla(x)
        dy = tape.gradient(y, x)
        return y, dy

    def get_config(self):
        config = super().get_config()
        config["vanilla"] = tf.keras.utils.serialize_keras_object(self.vanilla)
        return config

    @classmethod
    def from_config(cls, config):
        vanilla_cfg = config.pop("vanilla")
        vanilla = tf.keras.utils.deserialize_keras_object(vanilla_cfg)
        return cls(vanilla, **config)


@tf.keras.utils.register_keras_serializable(package="ml_greeks_pricers")
class WeightedMeanSquaredError(tf.keras.losses.Loss):
    """Mean squared error weighted by \lambda_j."""

    def __init__(self, lam: tf.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lam = tf.reshape(lam, (1, -1))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true = tf.cast(y_true, y_pred.dtype)
        lam = tf.cast(self.lam, y_pred.dtype)
        diff = lam * (y_true - y_pred)
        return tf.reduce_mean(tf.square(diff), axis=-1)

    def get_config(self):
        config = super().get_config()
        config["lam"] = self.lam.numpy().tolist()
        return config

    @classmethod
    def from_config(cls, config):
        lam = tf.constant(config.pop("lam"), dtype=tf.float32)
        return cls(lam, **config)
