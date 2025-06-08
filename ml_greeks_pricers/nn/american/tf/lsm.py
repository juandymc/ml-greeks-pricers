"""TensorFlow-friendly wrapper for Least Squares Monte Carlo."""

import numpy as np
from ..algorithms.backward_induction import regression
from .backward_induction_pricer import AmericanOptionPricer


class LeastSquaresPricer(AmericanOptionPricer):
    """LSM implementation exposed via the tf package."""

    def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
                 train_ITM_only=True, use_payoff_as_input=False):
        del nb_epochs, nb_batches
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        self.regression = regression.LeastSquares(
            model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        )

    def calculate_continuation_value(self, values, immediate_exercise_value,
                                     stock_paths_at_timestep):
        if self.train_ITM_only:
            in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
            in_the_money_all = np.where(immediate_exercise_value > 0)
        else:
            in_the_money = np.where(immediate_exercise_value[:self.split] < np.inf)
            in_the_money_all = np.where(immediate_exercise_value < np.inf)
        return_values = np.zeros(stock_paths_at_timestep.shape[0])
        return_values[in_the_money_all[0]] = self.regression.calculate_regression(
            stock_paths_at_timestep,
            values,
            in_the_money,
            in_the_money_all,
        )
        return return_values
