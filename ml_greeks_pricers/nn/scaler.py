import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class TwinScaler:
    """Standardise ``x`` and ``y`` and rescale derivatives consistently."""

    def __init__(self) -> None:
        self.xs = StandardScaler()
        self.ys = StandardScaler()
        self.dy_scale = None

    @staticmethod
    def _f32(a):
        return tf.cast(a, tf.float32)

    def fit(self, x, y) -> None:
        self.xs.fit(x)
        self.ys.fit(y)
        self.dy_scale = self.xs.scale_ / self.ys.scale_

    def transform(self, x, y, dy):
        return (
            self._f32(self.xs.transform(x)),
            self._f32(self.ys.transform(y)),
            self._f32(dy * self.dy_scale),
        )

    def x_transform(self, x):
        return self._f32(self.xs.transform(x))

    def inverse(self, y_s, dy_s):
        return (
            self.ys.inverse_transform(y_s),
            dy_s / self.dy_scale,
        )
