import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')

tf.keras.backend.clear_session()

from ml_greeks_pricers.pricers.tf import MarketData, BasketAsset, MCWorstOfOption

if __name__ == "__main__":
    S0 = [100.0, 95.0]
    q = [0.0, 0.0]
    sigma = [0.25, 0.20]
    corr = [[1.0, 0.3], [0.3, 1.0]]
    K = 90.0
    T = 1.0
    r = 0.03

    market = MarketData(r, sigma)
    asset = BasketAsset(S0, q, corr, T=T, dt=1/252, n_paths=200_000, seed=42)
    option = MCWorstOfOption(asset, market, K, T, is_call=False)

    price = option().numpy()
    delta = option.delta().numpy()
    vega = option.vega().numpy()

    tf.print("price", price)
    tf.print("delta", delta)
    tf.print("vega", vega)
