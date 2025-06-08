from ml_greeks_pricers.nn.american.data.stock_model import BlackScholes
from ml_greeks_pricers.nn.american.payoffs.payoff import MaxCall
from ml_greeks_pricers.nn.american.tf import (
    LeastSquaresPricer,
    NeuralNetworkPricerTF,
)
from ml_greeks_pricers.nn.american.run import configs
import numpy as np
import tensorflow as tf


def price_option(pricer):
    price, _ = pricer.price()
    return price


def main():
    # To keep the example runtime reasonable we use a reduced number of paths
    # and training epochs compared to the original values.
    nb_paths = 4000
    nb_dates = 50
    strike = 100
    spot = 100
    drift = 0.05
    volatility = 0.2
    maturity = 1

    configs.path_gen_seed.set_seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    payoff = MaxCall(strike)

    model_lsm = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_paths=nb_paths,
        nb_stocks=1,
        nb_dates=nb_dates,
        spot=spot,
        maturity=maturity,
    )
    lsm_pricer = LeastSquaresPricer(model_lsm, payoff)
    price_lsm = price_option(lsm_pricer)

    model_tf = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_paths=nb_paths,
        nb_stocks=1,
        nb_dates=nb_dates,
        spot=spot,
        maturity=maturity,
    )
    nlsm_pricer_tf = NeuralNetworkPricerTF(
        model_tf,
        payoff,
        nb_epochs=50,
        hidden_size=32,
    )
    price_nlsm_tf = price_option(nlsm_pricer_tf)

    diff = price_lsm - price_nlsm_tf

    output_lines = [
        f"LSM price (tf pkg): {price_lsm:.4f}",
        f"NLSM price (TensorFlow): {price_nlsm_tf:.4f}",
        f"Difference: {diff:.4f}",
    ]
    for l in output_lines:
        print(l)
    with open("lsm_vs_nlsm_tf_only.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))


if __name__ == "__main__":
    main()
