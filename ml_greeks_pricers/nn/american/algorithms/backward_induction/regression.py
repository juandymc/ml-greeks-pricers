"""Minimal regression routines for the TensorFlow LSM example."""

import numpy as np

# ``regression.py`` lives in ``ml_greeks_pricers.nn.american.algorithms.backward_induction``.
# The basis functions utilities are located in ``ml_greeks_pricers.nn.american.algorithms.utils``.
# Using three leading dots would search for ``ml_greeks_pricers.nn.american.utils`` which
# does not exist. The correct relative import goes up one level to the ``algorithms``
# package and then into ``utils``.
from ..utils import basis_functions


class LeastSquares:
    """Least squares regression using simple basis functions."""

    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        self.bf = basis_functions.BasisFunctions(self.nb_stocks)

    def calculate_regression(self, X, Y, in_the_money, in_the_money_all):
        nb_paths = X.shape[0]
        reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))
        for coeff in range(self.bf.nb_base_fcts):
            reg_vect_mat[:, coeff] = self.bf.base_fct(coeff, X[:, :], d2=True)
        coefficients = np.linalg.lstsq(
            reg_vect_mat[in_the_money[0]], Y[in_the_money[0]], rcond=None
        )
        continuation_values = np.dot(
            reg_vect_mat[in_the_money_all[0]], coefficients[0]
        )
        return continuation_values
