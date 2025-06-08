import numpy as np
import tensorflow as tf
from .backward_induction_pricer import AmericanOptionPricer
from .neural_networks import NetworkNLSM


class NeuralRegressionTF:
    """Train and evaluate a small neural network in TensorFlow."""

    def __init__(self, nb_stocks, nb_paths, hidden_size=10, nb_epochs=20, batch_size=2000):
        del nb_paths
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.model = NetworkNLSM(nb_stocks, hidden_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_network(self, X_inputs, Y_labels):
        X_inputs = tf.convert_to_tensor(X_inputs, dtype=tf.float64)
        Y_labels = tf.convert_to_tensor(Y_labels, dtype=tf.float64)
        Y_labels = tf.expand_dims(Y_labels, axis=1)
        dataset = tf.data.Dataset.from_tensor_slices((X_inputs, Y_labels)).batch(self.batch_size)
        for _ in range(self.nb_epochs):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, predictions)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def evaluate_network(self, X_inputs):
        X_inputs = tf.convert_to_tensor(X_inputs, dtype=tf.float64)
        preds = self.model(X_inputs, training=False)
        return tf.squeeze(preds, axis=1).numpy()


class NeuralNetworkPricerTF(AmericanOptionPricer):
    """TensorFlow implementation of Neural Least Squares Monte Carlo."""

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10, train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only, use_payoff_as_input=use_payoff_as_input)
        self.neural_regression = NeuralRegressionTF(
            model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1,
            model.nb_paths,
            hidden_size=hidden_size,
            nb_epochs=nb_epochs,
        )

    def calculate_continuation_value(self, values, immediate_exercise_value, stock_paths_at_timestep):
        inputs = stock_paths_at_timestep
        if self.train_ITM_only:
            in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
            in_the_money_all = np.where(immediate_exercise_value > 0)
        else:
            in_the_money = np.where(immediate_exercise_value[:self.split] < np.infty)
            in_the_money_all = np.where(immediate_exercise_value < np.infty)
        continuation_values = np.zeros(stock_paths_at_timestep.shape[0])
        self.neural_regression.train_network(inputs[in_the_money[0]], values[in_the_money[0]])
        continuation_values[in_the_money_all[0]] = self.neural_regression.evaluate_network(inputs[in_the_money_all[0]])
        return continuation_values
