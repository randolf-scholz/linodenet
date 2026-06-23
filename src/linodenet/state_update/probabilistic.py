r"""Probabilistic state-update protocol.

NOTE: WIP, not yet implemented.

Probabilistic State Update
--------------------------
- We need sampling distribution. What if we have multiple independent measurements
  of the same quantity at time t?
  - ⟹ We approximate the observational distribution.
  - Option 1: Empirical distribution
  - Option 2: Posteriors from a Bayesian model

Study Kalman Filter from a probabilistic perspective.
Transfer observations to the latent linear state.

Experiment with parametrized KalmanCell.

.. math:: ΣH'(HΣ H' + R)⁻¹(Hx - y)

R is observed or a hyperparameter.
"""

__all__ = [
    # Protocols & ABCs
    "ProbabilisticStateUpdate",
    "EmpiricalStateUpdate",
    "DiracStateUpdate",
    # Classes
    "GradientStepUpdater",
    "NaturalGaussianUpdater",
    "GaussianLatentStateUpdate",
    "probabilistic_kalman_update",
    "discrete_probabilistic_kalman_update",
]

from abc import abstractmethod
from collections.abc import Callable
from math import expm1, log
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.utils import _pytree

from linodenet.distributions import Dirac, Distribution, Empirical
from linodenet.distributions.gaussian import (
    kl,
    multivariate_gaussian_log_likelihood,
)
from linodenet.mappings import Transform


@runtime_checkable
class ProbabilisticStateUpdate[P: Distribution, Q: Distribution](Protocol):
    r"""Protocol for probabilistic state updates.

    The goal of a probabilistic state update is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    @abstractmethod
    def __call__(self, y: Q, x: P, /) -> P: ...


class DiracStateUpdate[D: Distribution](ProbabilisticStateUpdate[D, Dirac]):
    r"""Protocol for probabilistic state update with single-value observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Dirac, x: D, /) -> D: ...


class EmpiricalStateUpdate[D: Distribution](ProbabilisticStateUpdate[D, Empirical]):
    r"""Protocol for probabilistic state update with discrete observations."""

    @abstractmethod
    def __call__(self, y: Tensor | Empirical, x: D, /) -> D: ...


class GradientStepUpdater(nn.Module):
    r"""Single gradient-step updater for latent distribution parameters.

    Given an observation $y$ and latent parameters $θ$, this module pulls the
    observation back through the decoder and performs the Euclidean one-step
    update

    .. math::
        θ' = θ - λ⁻¹ ∇_θ[-\log p_θ(y)]

    where $p_θ(y)$ is induced by the decoder and the latent density.
    """

    decoder: Transform[Tensor, Tensor]
    r"""Decoder used to pull observations back to latent space."""
    latent_dist: Callable[[Tensor, Any], Tensor]
    r"""Callable returning the latent log-density $log p(x∣θ)$."""
    raw_lambda: Tensor
    r"""Unconstrained parameter whose softplus defines the positive $λ$."""

    def __init__(
        self,
        *,
        decoder: Transform[Tensor, Tensor],
        latent_dist: Callable[[Tensor, Any], Tensor],
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_init <= 0:
            raise ValueError(f"Expected lambda_init > 0, got {lambda_init}.")
        self.decoder = decoder
        self.latent_dist = latent_dist
        raw_lambda = log(expm1(lambda_init))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_lambda))

    @property
    def lambda_(self) -> Tensor:
        r"""Return the positive regularization/step-size parameter $λ$."""
        return F.softplus(self.raw_lambda) + torch.finfo(self.raw_lambda.dtype).eps

    def forward(self, y: Tensor, theta: Any, /) -> Any:
        r"""Update the latent parameter pytree using a single observation.

        Args:
            y: Observed value in data space.
            theta: Pytree of tensor parameters defining the latent distribution.

        Returns:
            Updated pytree with the same structure as `theta`.
        """
        leaves, spec = _pytree.tree_flatten(theta)
        differentiable_indices = [
            k
            for k, leaf in enumerate(leaves)
            if isinstance(leaf, Tensor)
            and (leaf.is_floating_point() or leaf.is_complex())
        ]

        if not differentiable_indices:
            return theta

        updated_leaves = list(leaves)
        grad_leaves: list[Tensor] = []
        for k in differentiable_indices:
            leaf = leaves[k]
            assert isinstance(leaf, Tensor)
            cloned = leaf.detach().clone().requires_grad_(True)
            updated_leaves[k] = cloned
            grad_leaves.append(cloned)

        theta_var = _pytree.tree_unflatten(updated_leaves, spec)
        # Pull back y ↦ x via the decoder so log p_θ(y) = log p(x∣θ) + log│det ∂x/∂y│.
        x, logabsdet = self.decoder.decode_and_logabsdet(y)
        log_density = self.latent_dist(x, theta_var)
        # Minimize L(θ) = -log p_θ(y) = -(log p(x∣θ) + log│det ∂x/∂y│).
        loss = -(log_density + logabsdet).sum()
        gradients = torch.autograd.grad(loss, grad_leaves, allow_unused=True)
        scale = self.lambda_.reciprocal()

        for index, leaf, gradient in zip(
            differentiable_indices, grad_leaves, gradients, strict=True
        ):
            # Single Euclidean step: θ' = θ - λ⁻¹∇_θL.
            updated_leaves[index] = (
                leaf if gradient is None else leaf - scale * gradient
            )

        return _pytree.tree_unflatten(updated_leaves, spec)


class NaturalGaussianUpdater(nn.Module):
    r"""Closed-form proximal update for a Gaussian latent state.

    Assumptions:
        1. The latent distribution is Gaussian: $x ∼ 𝓝(μ, Σ)$.
        2. The decoder is an invertible transform $y = h(x)$ providing
           `decode_and_logabsdet`.
        3. We observe a single Dirac value $y$.
        4. The Gaussian parameters are passed explicitly as the tuple $(μ, Σ)$.

    Let

    .. math:: z = h⁻¹(y), \qquad \log p_θ(y) = \log 𝓝(z; μ, Σ) + \log|\det\frac{∂z}{∂y}|

    Since the Jacobian term is independent of $(μ, Σ)$, this module computes the
    exact minimizer of the KL-regularized objective

    .. math:: \min_{μ', Σ'} -\log 𝓝(z; μ', Σ') + λ⋅\mathrm{KL}(𝓝(μ, Σ) ∣ 𝓝(μ', Σ'))

    Writing $η = (1 + λ)⁻¹$ and $δ = z - μ$, the unique Gaussian minimizer is

    .. math:: μ' = (1-η)μ + η z \qquad Σ' = (1-η)Σ + η(1-η)δδᵀ
    """

    decoder: Transform[Tensor, Tensor]
    r"""Decoder used to pull observations back to latent space."""
    raw_lambda: Tensor
    r"""Unconstrained parameter whose softplus defines the positive $λ$."""
    log_prob: Tensor
    r"""BUFFER: The most recent predictive log-likelihood $log p_θ(y)$."""

    def __init__(
        self,
        *,
        decoder: Transform[Tensor, Tensor],
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_init <= 0:
            raise ValueError(f"Expected lambda_init > 0, got {lambda_init}.")
        self.decoder = decoder
        raw_lambda = log(expm1(lambda_init))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_lambda))
        self.register_buffer("log_prob", torch.empty(()), persistent=False)

    @property
    def lambda_(self) -> Tensor:
        r"""Return the positive regularization parameter $λ$."""
        return F.softplus(self.raw_lambda) + torch.finfo(self.raw_lambda.dtype).eps

    def forward(
        self,
        y: Tensor,
        params: tuple[Tensor, Tensor],
        /,
    ) -> tuple[Tensor, Tensor]:
        r"""Return the updated Gaussian parameters $(μ', Σ')$."""
        mu, sigma = params
        if mu.ndim < 1:
            raise ValueError(
                f"Expected μ to have at least one dimension, got {mu.shape}."
            )
        if sigma.ndim < 2 or sigma.shape[-2:] != (mu.shape[-1], mu.shape[-1]):
            raise ValueError(
                "Expected Σ to have shape (..., d, d) matching μ.shape[-1], "
                f"got μ.shape={mu.shape} and Σ.shape={sigma.shape}."
            )
        if mu.shape[:-1] != sigma.shape[:-2]:
            raise ValueError(
                "Expected μ and Σ to share the same batch shape, "
                f"got μ.shape={mu.shape} and Σ.shape={sigma.shape}."
            )

        # Pull back y ↦ z so p_θ(y) = 𝓝(z; μ, Σ) · │det ∂z/∂y│.
        z, logabsdet = self.decoder.decode_and_logabsdet(y)
        self.log_prob = (
            multivariate_gaussian_log_likelihood(
                z,
                mean=mu,
                covariance_matrix=sigma,
            )
            + logabsdet
        )

        eta = (1 + self.lambda_).reciprocal()
        delta = z - mu
        outer = torch.einsum("...i, ...j -> ...ij", delta, delta)

        # Exact proximal solution: μ' = (1-η)μ + ηz and Σ' = (1-η)Σ + η(1-η)δδᵀ.
        mu_new = mu + eta * delta
        sigma_new = (1 - eta) * sigma + eta * (1 - eta) * outer
        return mu_new, sigma_new


def probabilistic_kalman_update(
    obs: MultivariateNormal,
    val: MultivariateNormal,
    /,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    r"""The classical Kalman Filter, rephrased in probabilistic terms.

    .. math::
        μ' = μ - ΣH'(HΣH' + R)⁻¹(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)⁻¹HΣ

    Note that the formula for the update derives from the conditional distribution of a joint
    normal distribution: If $p(x, y) = 𝓝([μ₁, μ₂], [[Σ₁₁, Σ₁₂], [Σ₂₁, Σ₂₂]])$, then
    $p(x∣y) = 𝓝(μ', Σ')$ where $μ' = μ₁ - Σ₁₂Σ₂₂⁻¹(μ₂ - y)$ and $Σ' = Σ₁₁ - Σ₁₂Σ₂⁻¹Σ₂₁$.

    Args:
        obs: The observation distribution.
        val: The state distribution.
        H: The observation matrix.

    Returns:
        The updated state distribution.
    """
    μ: Tensor = val.mean
    Σ: Tensor = val.covariance_matrix
    y: Tensor = obs.mean
    R: Tensor = obs.covariance_matrix

    if H is None:
        yhat = μ
        S = Σ
        Q = S + R
    else:
        yhat = torch.einsum("ij, ...j -> ...i", H, μ)
        S = torch.einsum("mj, jn -> mn", H, Σ)  # S=HΣ
        Q = torch.einsum("mj, nj -> mn", S, H) + R  # Q=HΣH' + R

    # perform cholesky decomposition (stability + ensure psd cov update)
    L = torch.linalg.cholesky(Q)
    # NOTE: compute ΣH'Q⁻¹r = ΣH'(LLᵀ)⁻¹r = ΣH'L⁻ᵀ(L⁻¹r)
    #   so we solve: Lz = r and L⋅P = HΣ
    #   then the update are ∆μ = Pᵀz and ∆Σ = PᵀP
    z = torch.linalg.solve_triangular(L, yhat - y, upper=False)
    P = torch.linalg.solve_triangular(L, S, upper=False)

    # perform the update
    μ = μ - torch.einsum("ji, ...j -> ...i", P, z)
    Σ = Σ - torch.einsum("mj, jn -> mn", P, P)

    return MultivariateNormal(μ, Σ)


def discrete_probabilistic_kalman_update(
    observation: Dirac,
    state: MultivariateNormal,
    *,
    R: Tensor,
    H: Optional[Tensor] = None,
) -> MultivariateNormal:
    """The classical Kalman Filter, rephrased in probabilistic terms.

    Note that the observation is allowed to be sparse, i.e. contain missing values.
    In this case, the update formula is obtained by marginalizing the observation distribution:

    .. math::
        μ' = μ - ΣH'(HΣH' + R)⁻¹(Hμ - y)
        Σ' = Σ - ΣH'(HΣH' + R)⁻¹HΣ

    References:
        Kalman filter with outliers and missing observations,
        T. Cipra & R. Romera, 1997
        https://link.springer.com/article/10.1007/BF02564705
    """
    y = observation.data  # (..., m)
    mask = torch.isnan(y)  # (..., m)

    if H is None:
        H = torch.eye(state.event_shape[-1], device=state.mean.device)

    H = H[..., mask]  # (..., n, m_obs)
    R = R[..., mask][..., mask, :]  # (..., m_obs, m_obs)

    obs_dist = MultivariateNormal(y, R)
    return probabilistic_kalman_update(obs_dist, state, H=H)


class GaussianLatentStateUpdate(nn.Module):
    r"""Perform a gradient based update assuming a latent Gaussian distribution.

    .. math:: Jₜ(θ; θ₋, y_obs) = -\log q(y_obs∣θ) + λ\kl(𝓝(μ, Σ), 𝓝(μ₋, Σ₋))

    where $θ = (μ, Σ)$ are the parameters of the latent Gaussian distribution,
    $θ₋$ are the parameters before the update, and $y_obs$ is the observed value.
    The first term is the negative log-likelihood of the observation under the current parameters,
    and the second term is a KL divergence regularization that encourages the updated parameters
    to stay close to the previous parameters.
    """

    def __init__(
        self,
        decoder: Callable[[tuple[Tensor, Tensor]], Distribution],
        regularization_strength,
        regularization_learnable: bool = True,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def update_covariance(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, Σ)$."""
        raise NotImplementedError

    def update_cholesky(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, L)$, with $Σ=LLᵀ$."""
        mu, sigma = theta
        mu_dash = nn.Parameter(mu)
        sigma_dash = nn.Parameter(sigma)
        grad_fn = torch.func.grad(kl, argnums=0)
        grad_fn((mu_dash, sigma_dash), (mu, sigma), parametrization="cholesky")
        raise NotImplementedError

    def update_precision(
        self, theta: tuple[Tensor, Tensor], y_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Gradient step assuming parameterization $θ=(μ, Λ)$, with $Σ=Λ⁻¹$."""
        raise NotImplementedError
