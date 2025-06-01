import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler


def lambda_j(dy_s: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.math.sqrt(tf.reduce_mean(tf.square(dy_s), axis=0))


def alpha_beta(n: int, l: float = 1.0):
    return [1.0 / (1 + l * n), l * n / (1 + l * n)]


def dataset(x, y, dy, bs):
    return tf.data.Dataset.from_tensor_slices((x, (y, dy))).batch(bs).repeat()


def lr_callback(schedule, epochs):
    t, r = zip(*schedule)

    def f(e):
        return np.interp(e / (epochs - 1), t, r)

    return LearningRateScheduler(f)
