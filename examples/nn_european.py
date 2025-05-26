import tensorflow as tf
print("Using TensorFlow version", tf.__version__)

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import time

# ----------------------------------------------------------------------------- #
# Model building
# ----------------------------------------------------------------------------- #
def vanilla_net(input_dim, hidden_units, hidden_layers, output_dim=1):
    net = tf.keras.Sequential()
    net.add(Input((input_dim,)))
    for _ in range(hidden_layers):
        net.add(Dense(hidden_units, activation="softplus"))
    net.add(Dense(output_dim))
    return net

class TwinNetwork(tf.keras.Model):
    """Returns value and first derivatives."""
    def __init__(self, vanilla):
        super().__init__()
        self.vanilla = vanilla
    def call(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.vanilla(x)
        dy = tape.gradient(y, x)
        return y, dy

class WeightedMeanSquaredError(tf.keras.losses.Loss):
    """MSE weighted by λⱼ."""
    def __init__(self, lam):          # lam shape (features,)
        super().__init__()
        self.lam = tf.reshape(lam, (1, -1))
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        lam    = tf.cast(self.lam,  y_pred.dtype)
        diff   = lam * (y_true - y_pred)
        return tf.reduce_mean(tf.square(diff), axis=-1)

class TwinScaler:
    """Standardises (x, y) and rescales dy/dx consistently."""
    def __init__(self):
        self.xs = StandardScaler()
        self.ys = StandardScaler()
    @staticmethod
    def _f32(a): return tf.cast(a, tf.float32)
    def fit(self, x, y):
        self.xs.fit(x); self.ys.fit(y)
        self.dy_scale = self.xs.scale_ / self.ys.scale_
    def transform(self, x, y, dy):
        return (self._f32(self.xs.transform(x)),
                self._f32(self.ys.transform(y)),
                self._f32(dy * self.dy_scale))
    def x_transform(self, x):  return self._f32(self.xs.transform(x))
    def inverse(self, y_s, dy_s):
        return (self.ys.inverse_transform(y_s),
                dy_s / self.dy_scale)

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def lambda_j(dy_s):       return 1.0 / tf.math.sqrt(tf.reduce_mean(tf.square(dy_s), 0))
def alpha_beta(n, l=1):   return [1.0/(1+l*n), l*n/(1+l*n)]

def dataset(x, y, dy, bs):
    return tf.data.Dataset.from_tensor_slices((x, (y, dy))).batch(bs).repeat()

def lr_callback(schedule, epochs):
    t, r = zip(*schedule)
    f = lambda e: np.interp(e/(epochs-1), t, r)
    return LearningRateScheduler(f)

# ----------------------------------------------------------------------------- #
# Trainer / predictor
# ----------------------------------------------------------------------------- #
class NeuralApproximator:
    def __init__(self, x, y, dy=None):
        self.x_raw, self.y_raw = x, y
        self.dy_raw = tf.zeros((y.shape[0], y.shape[1], x.shape[1]), tf.float32) if dy is None else dy
        self.scaler = TwinScaler()

    def prepare(self, m, diff, lam=1, hu=20, hl=4):
        self.scaler.fit(self.x_raw[:m], self.y_raw[:m])
        self.x_s, self.y_s, self.dy_s = self.scaler.transform(self.x_raw[:m],
                                                              self.y_raw[:m],
                                                              self.dy_raw[:m])
        n = self.x_s.shape[1]
        self.twin = TwinNetwork(vanilla_net(n, hu, hl))
        self.lam_j = lambda_j(self.dy_s)
        self.loss_w = alpha_beta(n, lam) if diff else [1, 0]

    def train(self, epochs=100, lr_sched=[(0,1e-8),(0.2,1e-1),(0.6,1e-2),(0.9,1e-6),(1,1e-8)],
              steps=16, bs=256):
        ds = dataset(self.x_s, self.y_s, self.dy_s, bs)
        self.twin.compile(optimizer="adam",
                          loss=["mse", WeightedMeanSquaredError(self.lam_j)],
                          loss_weights=self.loss_w)
        self.twin.fit(ds, epochs=epochs, steps_per_epoch=steps,
                      callbacks=[lr_callback(lr_sched, epochs)], verbose=0)

    def predict(self, x):
        x_s = self.scaler.x_transform(x)
        y_s, dy_s = self.twin.predict(x_s, verbose=0)
        return self.scaler.inverse(y_s, dy_s)

# ----------------------------------------------------------------------------- #
# Black-Scholes generator
# ----------------------------------------------------------------------------- #
def bs_price(s, k, v, T):
    d1 = (np.log(s/k)+.5*v*v*T)/(v*np.sqrt(T)); d2 = d1 - v*np.sqrt(T)
    return s*norm.cdf(d1) - k*norm.cdf(d2)
def bs_delta(s, k, v, T):
    d1 = (np.log(s/k)+.5*v*v*T)/(v*np.sqrt(T)); return norm.cdf(d1)
def bs_vega(s, k, v, T):
    d1 = (np.log(s/k)+.5*v*v*T)/(v*np.sqrt(T)); return s*np.sqrt(T)*norm.pdf(d1)

class BlackScholes:
    def __init__(self, vol=.2,T1=1,T2=2,K=1.1,vm=1.5):
        self.s0, self.v, self.T1, self.T2, self.K, self.vm = 1., vol, T1, T2, K, vm
    def training_set(self, m, anti=True, seed=None):
        np.random.seed(seed)
        r = np.random.normal(size=(m,2))
        v0 = self.v*self.vm
        R1 = np.exp(-.5*v0*v0*self.T1 + v0*np.sqrt(self.T1)*r[:,0])
        R2 = np.exp(-.5*self.v*self.v*(self.T2-self.T1) + self.v*np.sqrt(self.T2-self.T1)*r[:,1])
        S1, S2 = self.s0*R1, self.s0*R1*R2
        pay = np.maximum(0., S2-self.K)
        if anti:
            R2a = np.exp(-.5*self.v*self.v*(self.T2-self.T1) - self.v*np.sqrt(self.T2-self.T1)*r[:,1])
            S2a = S1*R2a; paya = np.maximum(0., S2a-self.K)
            Y = .5*(pay+paya)
            Z = .5*(np.where(S2>self.K,R2,0.).reshape(-1,1)+
                    np.where(S2a>self.K,R2a,0.).reshape(-1,1))
        else:
            Y, Z = pay, np.where(S2>self.K,R2,0.).reshape(-1,1)
        return (S1.reshape(-1,1).astype(np.float32),
                Y.reshape(-1,1).astype(np.float32),
                Z.astype(np.float32))
    def test_set(self, lo=.35, hi=1.65, n=100):
        s = np.linspace(lo,hi,n,dtype=np.float32).reshape(-1,1); T=self.T2-self.T1
        return (s, s,
                bs_price(s,self.K,self.v,T).astype(np.float32).reshape(-1,1),
                bs_delta(s,self.K,self.v,T).astype(np.float32).reshape(-1,1),
                bs_vega (s,self.K,self.v,T).astype(np.float32).reshape(-1,1))

# ----------------------------------------------------------------------------- #
# Experiment
# ----------------------------------------------------------------------------- #
def run_test(gen, sizes, n_test, seed):
    x_tr, y_tr, dy_tr = gen.training_set(max(sizes), seed=seed)
    x_te, x_ax, y_te, dy_te, _ = gen.test_set(n=n_test)
    reg = NeuralApproximator(x_tr, y_tr, dy_tr)
    v_pred, d_pred = {}, {}
    for sz in sizes:
        reg.prepare(sz, diff=False); reg.train(); v,d=reg.predict(x_te)
        v_pred[("std",sz)], d_pred[("std",sz)] = v, d[:,0]
        reg.prepare(sz, diff=True);  reg.train(); v,d=reg.predict(x_te)
        v_pred[("diff",sz)], d_pred[("diff",sz)] = v, d[:,0]
    return x_ax, y_te, dy_te[:,0], v_pred, d_pred

def plot(title, pred, x, y, sizes, ylabel):
    rows=len(sizes); fig,ax=plt.subplots(rows,2,figsize=(9,4*rows))
    for i,sz in enumerate(sizes):
        for j,kind in enumerate(("std","diff")):
            ax[i,j].plot(x*100, pred[(kind,sz)]*100,'co',ms=2,mfc='w',label="pred")
            ax[i,j].plot(x*100, y*100,'r.',ms=.5,label="target")
            ax[i,j].set_xlabel("spot (%)"); ax[i,j].set_ylabel(ylabel)
            if i==0: ax[i,j].set_title("standard" if kind=="std" else "differential")
            ax[i,j].legend(prop={"size":8})
    plt.suptitle(title); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    sizes=[1024,8192]; n_test=100; seed=np.random.randint(1e4); print(f"seed {seed}")
    gen=BlackScholes(); x,y,dy,vp,dp=run_test(gen,sizes,n_test,seed)
    plot("Black-Scholes values", vp, x, y, sizes, "value")
    plot("Black-Scholes deltas", dp, x, dy, sizes, "delta")
