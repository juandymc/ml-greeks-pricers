# Changelog

## Unreleased
- Fix AmericanAsset.simulate failing under XLA when called from `tf.function` by using cached `n_steps` instead of `tf.get_static_value`.
- Add 2D slice plots for surface visualizations in the examples.
