import tensorflow as tf

class NetworkNLSM(tf.keras.Model):
    """Simple feed-forward network used in the TensorFlow NLSM implementation."""

    def __init__(self, nb_stocks, hidden_size=10):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.leaky_relu)
        self.layer3 = tf.keras.layers.Dense(1)
        self.build((None, nb_stocks))

    def call(self, x, training=False):
        x = self.layer1(x)
        x = self.layer3(x)
        return x
