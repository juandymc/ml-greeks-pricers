# ml-greeks-pricers

This repo provides TensorFlow-based pricers for options, including a Monte Carlo European pricer.

## MCEuropeanOption
- Supports flat and local volatility models.
- Default path generation uses `tf.foldl`.
- Set `use_scan=True` to use `tf.scan` instead.
