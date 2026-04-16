r"""Calibration tests and NumPy/SciPy reference formulas for Kalman updates."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import cholesky, expm, logm, solve, sqrtm

type Array = NDArray[np.float64]
type State = tuple[Array, Array]
type UpdateFunction = Callable[[Array, State], State]
type CovarianceFromDirection = Callable[[Array, Array], Array]

STEP_SIZE = 1.0


def observation_precision(dim: int, /) -> Array:
    r"""Return the fixed observation precision $R^{-1}$."""
    variances = np.linspace(0.5, 1.5, dim, dtype=float)
    return np.diag(1.0 / variances)


def _symmetrize(matrix: Array, /) -> Array:
    return 0.5 * (matrix + matrix.T)


def _as_real_symmetric(matrix: NDArray[np.complex128] | Array, /) -> Array:
    real_matrix = np.real_if_close(matrix, tol=1000)
    if np.iscomplexobj(real_matrix):
        msg = "Expected a real matrix after matrix-function evaluation."
        raise ValueError(msg)
    return _symmetrize(np.asarray(real_matrix, dtype=float))


def _inverse_spd(matrix: Array, /) -> Array:
    eye = np.eye(matrix.shape[0], dtype=float)
    return _symmetrize(solve(matrix, eye, assume_a="pos"))


def _matrix_exp_symmetric(matrix: Array, /) -> Array:
    return _as_real_symmetric(expm(matrix))


def _validate_state(y_obs: Array, state: State, /) -> State:
    mu, sigma = state
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    dim = mu.shape[0]
    if y_obs.shape != (dim,):
        msg = f"Expected y_obs.shape == ({dim},), got {y_obs.shape!r}."
        raise ValueError(msg)
    if sigma.shape != (dim, dim):
        msg = f"Expected sigma.shape == ({dim}, {dim}), got {sigma.shape!r}."
        raise ValueError(msg)
    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=1e-10):
        msg = "Expected a symmetric covariance matrix."
        raise ValueError(msg)
    return mu, sigma


def _naive_newton_mean_update(
    y_obs: Array, mu: Array, sigma: Array, obs_precision: Array, /
) -> Array:
    gradient = obs_precision @ (mu - y_obs)
    hessian = _inverse_spd(sigma) + obs_precision
    delta = -solve(hessian, gradient, assume_a="pos")
    return mu + delta


def _mean_update(
    schema: str, y_obs: Array, mu: Array, sigma: Array, obs_precision: Array, /
) -> Array:
    residual = y_obs - mu
    match schema:
        case "euclidean":
            return mu + obs_precision @ residual
        case "natural":
            return mu + sigma @ (obs_precision @ residual)
        case "newton" | "exact":
            return _naive_newton_mean_update(y_obs, mu, sigma, obs_precision)
        case _:
            msg = f"Unknown mean-update schema: {schema!r}."
            raise ValueError(msg)


def _intrinsic_from_direction(direction: Array, sigma: Array, /) -> Array:
    # Take the symmetric principal square root of the covariance.
    sigma_sqrt = _as_real_symmetric(sqrtm(sigma))
    sigma_inv_sqrt = _inverse_spd(sigma_sqrt)
    tangent = sigma_inv_sqrt @ direction @ sigma_inv_sqrt
    return _symmetrize(sigma_sqrt @ _matrix_exp_symmetric(tangent) @ sigma_sqrt)


def _precision_from_direction(direction: Array, sigma: Array, /) -> Array:
    precision = _inverse_spd(sigma)
    precision_post = precision - precision @ direction @ precision
    return _inverse_spd(precision_post)


def _factor_from_direction(direction: Array, sigma: Array, /) -> Array:
    # Start from a lower-triangular factor of the covariance.
    factor = np.asarray(cholesky(sigma, lower=True), dtype=float)
    factor_inv = solve(factor, np.eye(factor.shape[0], dtype=float), assume_a="gen")
    tangent = 0.5 * factor_inv @ direction @ factor_inv.T
    factor_post = factor @ _matrix_exp_symmetric(tangent)
    return _symmetrize(factor_post @ factor_post.T)


def _cholesky_from_direction(direction: Array, sigma: Array, /) -> Array:
    # Start from a lower-triangular factor of the covariance.
    factor = np.asarray(cholesky(sigma, lower=True), dtype=float)
    factor_inv = solve(factor, np.eye(factor.shape[0], dtype=float), assume_a="gen")
    middle = _matrix_exp_symmetric(factor_inv @ direction @ factor_inv.T)
    factor_post = factor @ np.asarray(cholesky(middle, lower=True), dtype=float)
    return _symmetrize(factor_post @ factor_post.T)


def _log_from_direction(direction: Array, sigma: Array, /) -> Array:
    # Work in the symmetric matrix-log coordinates of the covariance.
    sigma_log = _as_real_symmetric(logm(sigma))
    # Apply the Fréchet derivative of log at the covariance to the update direction.
    eigvals, eigvecs = np.linalg.eigh(sigma)
    rotated = eigvecs.T @ direction @ eigvecs
    log_eigvals = np.log(eigvals)
    denom = eigvals[:, None] - eigvals[None, :]
    numer = log_eigvals[:, None] - log_eigvals[None, :]
    factors = np.divide(
        numer,
        denom,
        out=np.zeros_like(denom),
        where=np.abs(denom) > 1e-12,
    )
    diagonal = np.diag_indices_from(factors)
    factors[diagonal] = 1.0 / eigvals
    sigma_log_post = sigma_log + _symmetrize(eigvecs @ (factors * rotated) @ eigvecs.T)
    return _matrix_exp_symmetric(sigma_log_post)


def _apply_update(
    y_obs: Array,
    state: State,
    /,
    *,
    mean_schema: str,
    covariance_from_direction: CovarianceFromDirection,
    covariance_schema: str,
) -> State:
    mu, sigma = _validate_state(y_obs, state)
    obs_precision = observation_precision(mu.shape[0])
    mu_post = _mean_update(mean_schema, y_obs, mu, sigma, obs_precision)
    # Select the schema-specific covariance direction before mapping it into the requested realization.
    match covariance_schema:
        case "euclidean":
            direction = -0.5 * STEP_SIZE * obs_precision
        case "natural":
            direction = -STEP_SIZE * sigma @ obs_precision @ sigma
        case "newton":
            direction = -sigma @ obs_precision @ sigma
        case _:
            msg = f"Unknown covariance-update schema: {covariance_schema!r}."
            raise ValueError(msg)
    sigma_post = covariance_from_direction(direction, sigma)
    return mu_post, sigma_post


def _apply_naive_newton_update(
    y_obs: Array,
    state: State,
    /,
    *,
    covariance_from_direction: CovarianceFromDirection,
) -> State:
    mu, sigma = _validate_state(y_obs, state)
    obs_precision = observation_precision(mu.shape[0])
    mu_post = _naive_newton_mean_update(y_obs, mu, sigma, obs_precision)
    # Form the covariance gradient contributed by the observation precision.
    gradient = 0.5 * obs_precision
    # Apply the inverse covariance Hessian to obtain the Newton direction.
    direction = -_symmetrize(2.0 * sigma @ gradient @ sigma)
    sigma_post = covariance_from_direction(direction, sigma)
    return mu_post, sigma_post


def exact_update(y_obs: Array, state: State, /) -> State:
    r"""Return the exact linear-Gaussian Kalman update."""
    mu, sigma = _validate_state(y_obs, state)
    obs_precision = observation_precision(mu.shape[0])
    # Convert the prior covariance to precision, add the observation precision, and invert back.
    prior_precision = _inverse_spd(sigma)
    sigma_post = _inverse_spd(prior_precision + obs_precision)
    mu_post = _mean_update("exact", y_obs, mu, sigma, obs_precision)
    return mu_post, sigma_post


def intrinsic_euclidean_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Euclidean mean step and intrinsic SPD covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="euclidean",
        covariance_from_direction=_intrinsic_from_direction,
        covariance_schema="euclidean",
    )


def intrinsic_natural_update(y_obs: Array, state: State, /) -> State:
    r"""Use the natural-gradient mean step and intrinsic SPD covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="natural",
        covariance_from_direction=_intrinsic_from_direction,
        covariance_schema="natural",
    )


def intrinsic_newton_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Newton mean step and intrinsic SPD covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="newton",
        covariance_from_direction=_intrinsic_from_direction,
        covariance_schema="newton",
    )


def naive_newton_intrinsic_update(y_obs: Array, state: State, /) -> State:
    r"""Use an explicit Newton direction and the intrinsic SPD covariance map."""
    return _apply_naive_newton_update(
        y_obs, state, covariance_from_direction=_intrinsic_from_direction
    )


def precision_euclidean_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Euclidean mean step and additive precision update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="euclidean",
        covariance_from_direction=_precision_from_direction,
        covariance_schema="euclidean",
    )


def precision_natural_update(y_obs: Array, state: State, /) -> State:
    r"""Use the natural-gradient mean step and additive precision update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="natural",
        covariance_from_direction=_precision_from_direction,
        covariance_schema="natural",
    )


def precision_newton_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Newton mean step and additive precision update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="newton",
        covariance_from_direction=_precision_from_direction,
        covariance_schema="newton",
    )


def naive_newton_precision_update(y_obs: Array, state: State, /) -> State:
    r"""Use an explicit Newton direction and the precision-space realization."""
    return _apply_naive_newton_update(
        y_obs, state, covariance_from_direction=_precision_from_direction
    )


def factor_euclidean_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Euclidean mean step and generic factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="euclidean",
        covariance_from_direction=_factor_from_direction,
        covariance_schema="euclidean",
    )


def factor_natural_update(y_obs: Array, state: State, /) -> State:
    r"""Use the natural-gradient mean step and generic factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="natural",
        covariance_from_direction=_factor_from_direction,
        covariance_schema="natural",
    )


def factor_newton_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Newton mean step and generic factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="newton",
        covariance_from_direction=_factor_from_direction,
        covariance_schema="newton",
    )


def naive_newton_factor_update(y_obs: Array, state: State, /) -> State:
    r"""Use an explicit Newton direction and the generic factor realization."""
    return _apply_naive_newton_update(
        y_obs, state, covariance_from_direction=_factor_from_direction
    )


def cholesky_euclidean_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Euclidean mean step and triangular-factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="euclidean",
        covariance_from_direction=_cholesky_from_direction,
        covariance_schema="euclidean",
    )


def cholesky_natural_update(y_obs: Array, state: State, /) -> State:
    r"""Use the natural-gradient mean step and triangular-factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="natural",
        covariance_from_direction=_cholesky_from_direction,
        covariance_schema="natural",
    )


def cholesky_newton_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Newton mean step and triangular-factor covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="newton",
        covariance_from_direction=_cholesky_from_direction,
        covariance_schema="newton",
    )


def naive_newton_cholesky_update(y_obs: Array, state: State, /) -> State:
    r"""Use an explicit Newton direction and the triangular-factor realization."""
    return _apply_naive_newton_update(
        y_obs, state, covariance_from_direction=_cholesky_from_direction
    )


def log_euclidean_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Euclidean mean step and log-matrix covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="euclidean",
        covariance_from_direction=_log_from_direction,
        covariance_schema="euclidean",
    )


def log_natural_update(y_obs: Array, state: State, /) -> State:
    r"""Use the natural-gradient mean step and log-matrix covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="natural",
        covariance_from_direction=_log_from_direction,
        covariance_schema="natural",
    )


def log_newton_update(y_obs: Array, state: State, /) -> State:
    r"""Use the Newton mean step and log-matrix covariance update."""
    return _apply_update(
        y_obs,
        state,
        mean_schema="newton",
        covariance_from_direction=_log_from_direction,
        covariance_schema="newton",
    )


def naive_newton_log_update(y_obs: Array, state: State, /) -> State:
    r"""Use an explicit Newton direction and the log-matrix realization."""
    return _apply_naive_newton_update(
        y_obs, state, covariance_from_direction=_log_from_direction
    )


UPDATE_FUNCTIONS: dict[str, UpdateFunction] = {
    "exact": exact_update,
    "intrinsic_euclidean": intrinsic_euclidean_update,
    "intrinsic_natural": intrinsic_natural_update,
    "intrinsic_newton": intrinsic_newton_update,
    "naive_newton_intrinsic": naive_newton_intrinsic_update,
    "precision_euclidean": precision_euclidean_update,
    "precision_natural": precision_natural_update,
    "precision_newton": precision_newton_update,
    "naive_newton_precision": naive_newton_precision_update,
    "factor_euclidean": factor_euclidean_update,
    "factor_natural": factor_natural_update,
    "factor_newton": factor_newton_update,
    "naive_newton_factor": naive_newton_factor_update,
    "cholesky_euclidean": cholesky_euclidean_update,
    "cholesky_natural": cholesky_natural_update,
    "cholesky_newton": cholesky_newton_update,
    "naive_newton_cholesky": naive_newton_cholesky_update,
    "log_euclidean": log_euclidean_update,
    "log_natural": log_natural_update,
    "log_newton": log_newton_update,
    "naive_newton_log": naive_newton_log_update,
}

NAIVE_NEWTON_EQUIVALENCE_FUNCTIONS: dict[str, tuple[UpdateFunction, UpdateFunction]] = {
    "intrinsic": (intrinsic_newton_update, naive_newton_intrinsic_update),
    "precision": (precision_newton_update, naive_newton_precision_update),
    "factor": (factor_newton_update, naive_newton_factor_update),
    "cholesky": (cholesky_newton_update, naive_newton_cholesky_update),
    "log": (log_newton_update, naive_newton_log_update),
}


class _CalibrationBase:
    INPUT_SIZE = 16
    BATCH_SIZE = 32
    SEED = 0

    def make_batch(self) -> list[tuple[Array, State]]:
        rng = np.random.default_rng(self.SEED)
        noise = np.diag(np.linspace(0.5, 1.5, self.INPUT_SIZE, dtype=float))
        batch: list[tuple[Array, State]] = []

        for _ in range(self.BATCH_SIZE):
            mu = rng.normal(size=self.INPUT_SIZE)
            factor = rng.normal(size=(self.INPUT_SIZE, self.INPUT_SIZE))
            sigma = factor @ factor.T / self.INPUT_SIZE
            sigma += 0.5 * np.eye(self.INPUT_SIZE)

            predictive_covariance = sigma + noise
            predictive_factor = np.linalg.cholesky(predictive_covariance)
            y_obs = mu + predictive_factor @ rng.normal(size=self.INPUT_SIZE)
            batch.append((y_obs, (mu, sigma)))

        return batch


@pytest.mark.parametrize(("method_name", "method"), list(UPDATE_FUNCTIONS.items()))
class TestKalmanUpdateCalibration(_CalibrationBase):
    def test_calibration(
        self,
        method_name: str,
        method: UpdateFunction,
    ) -> None:
        mean_errors: list[float] = []
        covariance_errors: list[float] = []

        for y_obs, state in self.make_batch():
            mean_exact, covariance_exact = exact_update(y_obs, state)
            mean_updated, covariance_updated = method(y_obs, state)

            assert mean_updated.shape == (self.INPUT_SIZE,)
            assert covariance_updated.shape == (self.INPUT_SIZE, self.INPUT_SIZE)
            assert np.isfinite(mean_updated).all()
            assert np.isfinite(covariance_updated).all()
            np.testing.assert_allclose(
                covariance_updated,
                covariance_updated.T,
                atol=1e-8,
                rtol=1e-8,
            )
            np.linalg.cholesky(covariance_updated)

            mean_error = np.linalg.norm(
                (mean_updated - mean_exact).reshape(1, -1), ord="fro"
            )
            covariance_error = np.linalg.norm(
                covariance_updated - covariance_exact, ord="fro"
            )
            mean_errors.append(float(mean_error))
            covariance_errors.append(float(covariance_error))

        mean_error = float(np.mean(mean_errors))
        covariance_error = float(np.mean(covariance_errors))
        print(
            f"{method_name}: "
            f"mean_frobenius={mean_error:.6e}, "
            f"covariance_frobenius={covariance_error:.6e}"
        )

        assert np.isfinite(mean_error)
        assert np.isfinite(covariance_error)

        if method_name == "exact":
            assert mean_error < 1e-10
            assert covariance_error < 1e-10


@pytest.mark.parametrize(
    ("parametrization", "methods"),
    list(NAIVE_NEWTON_EQUIVALENCE_FUNCTIONS.items()),
)
class TestNaiveNewtonEquivalence(_CalibrationBase):
    def test_naive_newton_matches_closed_form(
        self,
        parametrization: str,
        methods: tuple[UpdateFunction, UpdateFunction],
    ) -> None:
        closed_form, naive = methods

        for y_obs, state in self.make_batch():
            mean_closed, covariance_closed = closed_form(y_obs, state)
            mean_naive, covariance_naive = naive(y_obs, state)

            np.testing.assert_allclose(
                mean_naive,
                mean_closed,
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"{parametrization} mean update mismatch",
            )
            np.testing.assert_allclose(
                covariance_naive,
                covariance_closed,
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"{parametrization} covariance update mismatch",
            )
