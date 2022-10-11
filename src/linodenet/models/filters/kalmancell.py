r"""Improved KalmanCell Filter.

The classical Kalman Filter state update has a few nice properties

.. math::
       x' &= x - PH'(HPH' + R)^{-1}(Hx - y)
    \\    &= x - P ∇ₓ½‖(HPH' + R)^{-½}(Hx - y) ‖₂²

- The state update is linear (affine) in the state.
- The state update is linear (affine) in the measurement.
- The state update can be interpreted as a gradient descent step.
- The measurement covariance is used to weight the gradient descent step.
    - If R is large, the gradient descent step is small.
      We cannot trust the measurement due to high variance.
    - If R is small, the gradient descent step is large.
      We can trust the measurement due to low variance.
    - R should be treated as a hyperparameter / observable.
      In particular, often it is given as percentage measurement error.

The KalmanCell filter is a generalization of the classical Kalman Filter.



Probabilistic Filter:

- We need sampling distribution. What if we have multiple independent measurements
  of the same quantity at time t?
  - ⟹ We approximate the observational distribution.
  - Option 1: Empirical distribution
  - Option 2: Posteriors from a Bayesian model

Study Kalman Filter from a probabilistic perspective.
Transfer observations to the latent linear state.

Experiment with parametrized KalmanCell.

ΣH'(HΣ H' + R)^{-1}(Hx - y)

"""
