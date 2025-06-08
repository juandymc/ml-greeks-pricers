import numpy as np
import math
import time
from optimal_stopping.run import configs


class AmericanOptionPricer:
    """Simplified base class for backward induction algorithms."""

    def __init__(self, model, payoff, use_rnn=False, train_ITM_only=True,
                 use_path=False, use_payoff_as_input=False):
        self.model = model
        self.use_var = bool(getattr(self.model, "return_var", False))
        self.payoff = payoff
        self.use_rnn = use_rnn
        self.use_path = use_path
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.which_weight = 0

    def calculate_continuation_value(self, values, immediate_exercise_value, stock_paths_at_timestep):
        raise NotImplementedError

    def stop(self, stock_values, immediate_exercise_values,
             discounted_next_values, h=None, var_paths=None,
             return_continuation_values=False):
        stopping_rule = np.zeros(len(stock_values))
        if self.use_rnn:
            continuation_values = self.calculate_continuation_value(
                discounted_next_values,
                immediate_exercise_values, h)
        else:
            if self.use_var:
                stock_values = np.concatenate([stock_values, var_paths], axis=1)
            continuation_values = self.calculate_continuation_value(
                discounted_next_values,
                immediate_exercise_values, stock_values)
        if self.train_ITM_only:
            which = (immediate_exercise_values > continuation_values) & (
                immediate_exercise_values > np.finfo(float).eps)
        else:
            which = immediate_exercise_values > continuation_values
        stopping_rule[which] = 1
        if return_continuation_values:
            return stopping_rule, continuation_values
        return stopping_rule

    def price(self, train_eval_split=2):
        model = self.model
        t1 = time.time()
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
        stock_paths, var_paths = self.model.generate_paths()
        payoffs = self.payoff(stock_paths)
        stock_paths_with_payoff = np.concatenate(
            [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        time_for_path_gen = time.time() - t1
        self.split = int(len(stock_paths)/train_eval_split)
        print(f"time path gen: {time_for_path_gen}", end=" ")
        if self.use_rnn:
            if self.use_payoff_as_input:
                hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
            else:
                hs = self.compute_hs(stock_paths, var_paths=var_paths)
        disc_factor = math.exp((-model.rate) * model.maturity /
                               (model.nb_dates))
        immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
        values = immediate_exercise_value
        for i, date in enumerate(range(stock_paths.shape[2] - 2, 0, -1)):
            self.which_weight = i
            immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])
            if self.use_rnn:
                h = hs[date]
            else:
                h = None
            if self.use_path:
                varp = None
                if self.use_var:
                    varp = var_paths[:, :, :date+1]
                if self.use_payoff_as_input:
                    paths = stock_paths_with_payoff[:, :, :date+1]
                else:
                    paths = stock_paths[:, :, :date+1]
            else:
                varp = None
                if self.use_var:
                    varp = var_paths[:, :, date]
                if self.use_payoff_as_input:
                    paths = stock_paths_with_payoff[:, :, date]
                else:
                    paths = stock_paths[:, :, date]
            stopping_rule = self.stop(
                paths, immediate_exercise_value,
                values*disc_factor, h=h, var_paths=varp)
            which = stopping_rule > 0.5
            values[which] = immediate_exercise_value[which]
            values[~which] *= disc_factor
        payoff_0 = self.payoff.eval(stock_paths[:, :, 0])[0]
        return max(payoff_0, np.mean(values[self.split:]) * disc_factor), time_for_path_gen
