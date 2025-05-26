import tensorflow as tf
print("Using TensorFlow version", tf.__version__)

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import time


def vanilla_net(input_dim, hidden_units, hidden_layers, output_dim=1):
    net = tf.keras.Sequential()
    net.add(Input((input_dim,)))
    for _ in range(hidden_layers):
        net.add(Dense(hidden_units, activation="softplus"))
    net.add(Dense(output_dim))
    return net

"""| Parámetro       | ¿Qué representa?                                                                                                                                                                                                                                     | Ejemplo en Differential-ML                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `input_dim`     | **Tamaño del vector de entrada** que la red recibirá en cada   `call`. Aquí será la longitud del estado $x=[S_{T_1},U_{T_1}]$. <br>Si tu modelo tiene $n_0$ variables de mercado y $n_1$ variables de transacción, entonces `input_dim = n_0 + n_1`. | • Europea simple: $n_0=1,\;n_1=0$ ⇒ `input_dim=1`. <br>• Americana con flag: $1+1$ ⇒ `input_dim=2`. |
| `hidden_units`  | Número de **neuronas** (anchura) en **cada** capa oculta.  Todas las ocultas usan ese mismo valor.                                                                                                                                                   | El paper original usa 20; el repo de pmdanton usa 64.                                               |
| `hidden_layers` | Número de **capas ocultas** que se van a apilar en el bucle `for`. Cada una tendrá `hidden_units` neuronas y activación *softplus*.                                                                                                                  | 4 es la cifra habitual en los ejemplos de Huge-Savine.                                              |
| `output_dim`    | Dimensión de la capa de salida. En pricing suele ser 1 (un único precio). Podrías poner >1 si quisieras predecir varios valores simultáneamente.                                                                                                     | Se deja por defecto en 1 (`output_dim=1`).                                                          |

"""

class TwinNetwork(tf.keras.Model):

    def __init__(self, vanilla_net):
        super(TwinNetwork, self).__init__()
        self.vanilla_net = vanilla_net

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.vanilla_net(inputs)
        derivs_predictions = tape.gradient(predictions, inputs)
        return predictions, derivs_predictions

class WeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, lambda_j):
        super(WeightedMeanSquaredError, self).__init__()
        self.lambda_j = tf.reshape(lambda_j, (1,-1))

    def call(self, y_true, y_pred):
        return tf.keras.losses.MSE(self.lambda_j*y_true, self.lambda_j*y_pred)

class TwinScaler():
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, x, y):
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)
        self.dy_dx_scale = self.x_scaler.scale_ / self.y_scaler.scale_

    def transform(self, x, y, dy_dx):
        # make sur to cast to Tensorflow float32, to avoid conflict with NumPy float64
        x_scaled = self.x_scaler.transform(x)
        x_scaled = tf.cast(x_scaled, dtype=tf.float32)
        y_scaled = self.y_scaler.transform(y)
        y_scaled = tf.cast(y_scaled, dtype=tf.float32)
        dy_dx_scaled = dy_dx * self.dy_dx_scale
        dy_dx_scaled = tf.cast(dy_dx_scaled, dtype=tf.float32)
        return (x_scaled, y_scaled, dy_dx_scaled)

    def x_transform(self, x):
        x_scaled = self.x_scaler.transform(x)
        x_scaled = tf.cast(x_scaled, dtype=tf.float32)
        return x_scaled

    def inverse_transform(self, y_scaled, dy_dx_scaled):
        y = self.y_scaler.inverse_transform(y_scaled)
        dy_dx = dy_dx_scaled / self.dy_dx_scale
        return (y, dy_dx)

def calc_lambda_j(dy_dx_scaled):
    return 1.0/tf.math.sqrt(tf.reduce_mean(tf.square(dy_dx_scaled), axis=0))

def calc_alpha_beta(n, lambda_hyperparameter=1):
    alpha = 1/(1+lambda_hyperparameter*n)
    beta = 1-alpha
    return [alpha, beta]

def build_dataset(x, y, dydx, batch_size):
    inputs = tf.data.Dataset.from_tensor_slices(x)
    outputs = tf.data.Dataset.from_tensor_slices((y, dydx))
    dataset = tf.data.Dataset.zip((inputs, outputs)).batch(batch_size).repeat()
    return dataset

def build_scheduler_callback(learning_rate_schedule, epochs):
    lr_schedule_epochs = [z[0] for z in learning_rate_schedule]
    lr_schedule_rates = [z[1] for z in learning_rate_schedule]
    lr_schedule_fn = lambda t: np.interp(t/(epochs-1), lr_schedule_epochs, lr_schedule_rates)
    return LearningRateScheduler(lr_schedule_fn)

class Neural_Approximator():
    def __init__(self, x_raw, y_raw, dydx_raw=None):
        self.x_raw = x_raw
        self.y_raw = y_raw
        if dydx_raw is None:
            self.dydx_raw = tf.zeros((y_raw.shape[0], y_raw.shape[1], x_raw.shape[1]))
        else:
            self.dydx_raw = dydx_raw
        self.scaler = TwinScaler()

    def prepare(self, m, differential, lam=1, hidden_units=20, hidden_layers=4, *args, **kwargs):
        self.scaler.fit(self.x_raw[:m], self.y_raw[:m])
        self.x, self.y, self.dydx = self.scaler.transform(self.x_raw[:m], self.y_raw[:m], self.dydx_raw[:m])
        self.m, self.n = self.x.shape
        self.lambda_j = calc_lambda_j(self.dydx)
        net = vanilla_net(self.n, hidden_units, hidden_layers)
        self.twin_net = TwinNetwork(net)
        if differential:
            self.alpha_beta = calc_alpha_beta(self.n, lam)
        else:
            self.alpha_beta = [1, 0]

    def train(self,
          description="training",
          # training params
          reinit=True,
          epochs=100,
          # one-cycle learning rate schedule
          learning_rate_schedule=[
              (0.0, 1.0e-8),
              (0.2, 0.1),
              (0.6, 0.01),
              (0.9, 1.0e-6),
              (1.0, 1.0e-8)],
          batches_per_epoch=16,
          min_batch_size=256, *args, **kwargs):

        # build the dataset
        dataset = build_dataset(self.x, self.y, self.dydx, min_batch_size)

        # Build the weighted mean square error using lambda_j weights
        weighted_mse = WeightedMeanSquaredError(self.lambda_j)
        self.twin_net.compile(optimizer="adam", loss=["mse", weighted_mse], loss_weights=self.alpha_beta)
        # We use Keras LearningRateScheduler callback for the one-cycle learning rate schedule
        lr_scheduler = build_scheduler_callback(learning_rate_schedule, epochs)
        self.training_log = self.twin_net.fit(dataset, epochs=epochs, steps_per_epoch=batches_per_epoch, callbacks=[lr_scheduler], verbose=0)

    def predict_values(self, x):
        x_scaled = self.scaler.x_transform(x)
        y_scaled, _ = self.twin_net.predict(x_scaled)
        y = self.scaler.y_scaler.inverse_transform(y_scaled)
        return y

    def predict_values_and_derivs(self, x):
        x_scaled = self.scaler.x_transform(x)
        pred_scaled = self.twin_net.predict(x_scaled)
        return self.scaler.inverse_transform(*pred_scaled)

import tensorflow as tf
print("Using TensorFlow version", tf.__version__)

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import time

# helper analytics
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)

def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)

def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)
#

# main class
class BlackScholes:

    def __init__(self,
                 vol=0.2,
                 T1=1,
                 T2=2,
                 K=1.10,
                 volMult=1.5):

        self.spot = 1
        self.vol = vol
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult

    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, anti=True, seed=None):

        np.random.seed(seed)

        # 2 sets of normal returns
        returns = np.random.normal(size=[m, 2])

        # SDE
        vol0 = self.vol * self.volMult
        R1 = np.exp(-0.5*vol0*vol0*self.T1 + vol0*np.sqrt(self.T1)*returns[:,0])
        R2 = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1) \
                    + self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
        S1 = self.spot * R1
        S2 = S1 * R2

        # payoff
        pay = np.maximum(0, S2 - self.K)

        # two antithetic paths
        if anti:

            R2a = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1) \
                    - self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
            S2a = S1 * R2a
            paya = np.maximum(0, S2a - self.K)

            X = S1
            Y = 0.5 * (pay + paya)

            # differentials
            Z1 =  np.where(S2 > self.K, R2, 0.0).reshape((-1,1))
            Z2 =  np.where(S2a > self.K, R2a, 0.0).reshape((-1,1))
            Z = 0.5 * (Z1 + Z2)

        # standard
        else:

            X = S1
            Y = pay

            # differentials
            Z =  np.where(S2 > self.K, R2, 0.0).reshape((-1,1))

        return X.reshape([-1,1]), Y.reshape([-1,1]), Z.reshape([-1,1])

    # test set: returns a grid of uniform spots
    # with corresponding ground true prices, deltas and vegas
    def testSet(self, lower=0.35, upper=1.65, num=100, seed=None):

        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, deltas and vegas
        prices = bsPrice(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bsDelta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas = bsVega(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices, deltas, vegas

def test(generator,
         sizes,
         nTest,
         simulSeed=None,
         testSeed=None,
         weightSeed=None,
         deltidx=0):

    # simulation
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSet(max(sizes), seed=simulSeed)
    xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num=nTest, seed=testSeed)
    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")

    predvalues = {}
    preddeltas = {}
    for size in sizes:

        print("\nsize %d" % size)
        regressor.prepare(size, False, weight_seed=weightSeed)

        t0 = time.time()
        regressor.train("standard training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()

        regressor.prepare(size, True, weight_seed=weightSeed)

        t0 = time.time()
        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()

    return xAxis, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas

def graph(title,
          predictions,
          xAxis,
          xAxisName,
          yAxisName,
          targets,
          sizes,
          computeRmse=False,
          weights=None):

    numRows = len(sizes)
    numCols = 2

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

    for i, size in enumerate(sizes):
        ax[i,0].annotate("size %d" % size, xy=(0, 0.5),
          xytext=(-ax[i,0].yaxis.labelpad-5, 0),
          xycoords=ax[i,0].yaxis.label, textcoords='offset points',
          ha='right', va='center')

    ax[0,0].set_title("standard")
    ax[0,1].set_title("differential")

    for i, size in enumerate(sizes):
        for j, regType, in enumerate(["standard", "differential"]):

            if computeRmse:
                errors = 100 * (predictions[(regType, size)] - targets)
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName

            ax[i,j].set_xlabel(t)
            ax[i,j].set_ylabel(yAxisName)

            ax[i,j].plot(xAxis*100, predictions[(regType, size)]*100, 'co', \
                         markersize=2, markerfacecolor='white', label="predicted")
            ax[i,j].plot(xAxis*100, targets*100, 'r.', markersize=0.5, label='targets')

            ax[i,j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    plt.show()

# simulation set sizes to perform
sizes = [1024, 8192]

# show delta?
showDeltas = True

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000)
print("using seed %d" % simulSeed)
weightSeed = None

# number of test scenarios
nTest = 100

# go
generator = BlackScholes()
xAxis, yTest, dydxTest, vegas, values, deltas = \
    test(generator, sizes, nTest, simulSeed, None, weightSeed)

# show predicitions
graph("Black & Scholes", values, xAxis, "", "values", yTest, sizes, True)

# show deltas
graph("Black & Scholes", deltas, xAxis, "", "deltas", dydxTest, sizes, True)

