import tensorflow as tf

# Enable XLA JIT compilation when a GPU is available
USE_XLA = bool(tf.config.list_physical_devices('GPU'))
