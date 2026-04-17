r"""Gaussian Distribution.

Note:
    When marginalizing a parameterized distribution, it is often not enough that
    the distribution itself is analytically marginalizable. This is because,
    many distributions require specific constraints on the parameters such as
    for example that the covariance matrix is positive definite.

    However, we often use unconstrained weight tensor $w$, and only obtain the actual
    parameters $θ$ by applying a transformation $θ = f(w)$ called the parametrization.

    Now, marginalizing the distribution requires updating the parameters $θ$,
    but actually, we need to update the weights $w$ and then apply the parametrization.

    In essence, this means we need to be able to find a mapping $w → w'$ such that the
    following diagram commutes::

            parametrizations
        w ──────────────► θ
        │                 │
        │                 │m
        │φ                │a
        │                 │r
        │                 │
        ▼   parametrizations   ▼
        w'──────────────► θ'

    I.e. we need to find a mapping $φ$ on the unconstrained parameters such that
    $f(φ(w)) = m(f(w))$ where $m$ is the marginalization operation on the
    parameters.
"""

__all__ = [
    "kl",
    "kl_cholesky",
    "multivariate_gaussian_log_likelihood",
    "MultivariateNormal",
    "MultiHeadGaussian",
]

import math
from typing import Final, Optional, Self

import torch
import torch.nn.functional as F
from torch import Tensor, distributions as dist, nn
from torch.linalg import cholesky, solve_triangular, vecdot

from .base import DistributionBase


def kl(
    p: tuple[Tensor, Tensor],
    q: tuple[Tensor, Tensor],
    /,
) -> Tensor:
    r"""Return the KL divergence $KL(p, q)$ for multivariate Gaussians.

    Args:
        p: Mean $μᵤ$ with shape `(..., d)` and covariance $Σᵤ$ with shape `(..., d, d)`.
        q: Mean $μᵥ$ with shape `(..., d)` and covariance $Σᵥ$ with shape `(..., d, d)`.

    Returns:
        The KL divergence

        .. math:: KL(p, q) = ½(\tr(Σᵥ⁻¹Σᵤ) -d + (μᵥ-μᵤ)ᵀΣᵥ⁻¹(μᵥ-μᵤ) + \log\det Σᵥ - \log\det Σᵤ)
    """
    mean_p, cov_p = p
    mean_q, cov_q = q
    return kl_cholesky((mean_p, cholesky(cov_p)), (mean_q, cholesky(cov_q)))


def kl_cholesky(
    p: tuple[Tensor, Tensor],
    q: tuple[Tensor, Tensor],
    /,
) -> Tensor:
    r"""Return the KL divergence $KL(p, q)$ for Cholesky-parameterized Gaussians.

    Args:
        p: Mean $μᵤ$ with shape `(..., d)` and lower Cholesky factor $Lᵤ$ with shape `(..., d, d)`.
        q: Mean $μᵥ$ with shape `(..., d)` and lower Cholesky factor $Lᵥ$ with shape `(..., d, d)`.

    Returns:
        The KL divergence where $Σᵤ = LᵤLᵤᵀ$ and $Σᵥ = LᵥLᵥᵀ$.
    """
    mean_p, chol_p = p
    mean_q, chol_q = q
    # (μᵥ - μᵤ)ᵀ Σᵥ⁻¹ (μᵥ - μᵤ)
    delta = mean_q - mean_p
    whitened = solve_triangular(chol_q, delta.unsqueeze(-1), upper=False).squeeze(-1)
    mahalanobis = vecdot(whitened, whitened, dim=-1)

    # tr(Σᵥ⁻¹ Σᵤ) = ⟨Σᵥ⁻¹, Σᵤ⟩ = ⟨Lᵥ⁻ᵀ Lᵥ⁻¹, Lᵤ Lᵤᵀ⟩ = ⟨Lᵥ⁻¹ Lᵤ, Lᵥ⁻¹ Lᵤ⟩ = ‖Lᵥ⁻¹ Lᵤ‖²
    trace_term = solve_triangular(chol_q, chol_p, upper=False)
    trace_term = trace_term.square().sum(dim=(-2, -1))

    # log det Σᵤ and log det Σᵥ
    logdet_p = 2 * chol_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    dim = mean_p.shape[-1]

    return 0.5 * (trace_term + mahalanobis - dim + logdet_q - logdet_p)


def multivariate_gaussian_log_likelihood(
    value: Tensor,
    /,
    *,
    mean: Tensor,
    covariance_matrix: Tensor,
) -> Tensor:
    r"""Return the log-likelihood of a multivariate Gaussian.

    Args:
        value: Evaluation point $x$ with shape `(..., d)`.
        mean: Mean $μ$ with shape `(..., d)`.
        covariance_matrix: Covariance $Σ$ with shape `(..., d, d)`.

    Returns:
        The log-density

        .. math::
            \log 𝓝(x; μ, Σ) = -½(d\log(2π) + \log\det Σ + (x-μ)ᵀΣ⁻¹(x-μ)).
    """
    # Factor Σ = LLᵀ so the Mahalanobis term becomes ‖L⁻¹(x-μ)‖².
    residual = value - mean
    L = cholesky(covariance_matrix)
    whitened = solve_triangular(
        L,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    dim = residual.shape[-1]
    logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    mahalanobis = vecdot(whitened, whitened, dim=-1)
    return -0.5 * (dim * math.log(2 * math.pi) + logdet + mahalanobis)


class MultivariateNormal(dist.MultivariateNormal):
    r"""Augmented Multivariate Normal distribution.

    We add some utilities to the base class.
    """

    def __add__(self, bias: float | Tensor, /) -> Self:
        r"""Add a tensor to the mean."""
        return self.__class__(
            self.mean + bias,
            self.covariance_matrix,
        )

    def __mul__(self, scale: float | Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale * self.mean,
            scale**2 * self.covariance_matrix,
        )

    def __matmul__(self, scale: Tensor, /) -> Self:
        r"""Multiply by a tensor."""
        return self.__class__(
            scale @ self.mean,
            scale @ self.covariance_matrix @ scale.T,
        )


class MultiHeadGaussian(DistributionBase):
    r"""Implements a multi-head Gaussian distribution."""

    normalization_constant: Tensor
    r"""CONST: Normalization constant of a Gaussian distribution."""
    num_heads: Final[int]
    r"""CONST: Shape of heads"""
    num_features: Final[int]
    r"""CONST: Number of features in input."""

    # parameters/buffers
    means: Tensor
    r"""PARAM: Means of the gaussians."""
    scale_tril: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""PARAM: Parameters determining the covariances."""

    # non-permanent buffers
    eye: Tensor
    r"""BUFFER: Identity matrix."""
    covs: Tensor
    r"""BUFFER: Covariances of the gaussians."""
    cholesky_factor: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""BUFFER: Cholesky factor of the covariance matrix."""
    samples: Tensor
    r"""BUFFER: Stored samples when sampling."""
    latents: Tensor
    r"""BUFFER: Stored latents when evaluating log_probs."""
    log_probs: Tensor
    r"""BUFFER: Stored log_probs when evaluating log_probs."""

    def __init__(
        self,
        n_heads: int,
        n_feats: int,
        *,
        means: Optional[Tensor] = None,
        covs: Optional[Tensor] = None,
    ) -> None:
        super().__init__(batch_shape=(n_heads,), event_shape=(n_feats,))
        # CONSTANTS
        self.num_heads = int(n_heads)
        self.num_features = int(n_feats)
        normalization_constant = (
            0.5 * self.num_features * math.log(2 * math.pi)
        )  # -log (2π)^{-k/2}
        self.register_buffer(
            "normalization_constant", torch.tensor(normalization_constant)
        )
        self.register_buffer("eye", torch.eye(n_feats, dtype=torch.bool))

        # BUFFERS
        self.register_buffer("covs", torch.empty(0), persistent=False)
        self.register_buffer("cholesky_factor", torch.empty(0), persistent=False)
        self.register_buffer("samples", torch.empty(0), persistent=False)
        self.register_buffer("latents", torch.empty(0), persistent=False)
        self.register_buffer("log_probs", torch.empty(0), persistent=False)

        # initialize the means
        self.means = nn.Parameter(
            torch.as_tensor(means)
            if means is not None
            else self.sample_default_means(n_heads, n_feats)
        )
        # initialize the covariances
        self.scale_tril = nn.Parameter(  # not a parameter!
            torch.as_tensor(covs)
            if covs is not None
            else self.sample_default_covs(n_heads, n_feats)
        )

    @staticmethod
    def sample_default_means(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default means $μᵢ∼𝓝(0,1)$."""
        return torch.randn(n_heads, n_feats)

    @staticmethod
    def sample_default_covs(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default covariances."""
        return torch.eye(n_feats) + torch.randn(n_heads, n_feats, n_feats) / n_feats

    def get_cholesky(self) -> Tensor:
        r"""Compute cholesky factor of covariance matrix."""
        lower = self.scale_tril.tril()
        diag = lower.diagonal(dim1=-2, dim2=-1)
        # need to make the diagonal positive
        new_diag = F.softplus(diag) + 1e-6  # (M, D)
        # (D, D), (M, D, 1), (M, D, D) -> (M, D, D)
        self.cholesky_factor = torch.where(self.eye, new_diag.unsqueeze(-1), lower)
        return self.cholesky_factor

    def get_covariance(self) -> Tensor:
        r"""Compute covariance matrix from cholesky factor."""
        L = self.get_cholesky()  # M x D x D
        self.covs = torch.einsum("mij,mkj->mik", L, L)
        return self.covs

    def forward(self, x: Tensor, /) -> Tensor:
        r"""Transform $x -> y = Lx + μ$.

        Args:
            x (..., H, D): input tensor

        Returns:
            y (..., H, D): transformed tensor
        """
        L = self.get_cholesky()
        y = self.means + torch.einsum("...mj, mij -> ...mi", x, L)
        return y

    def inverse(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Transform $y -> x = L⁻¹(y-μ)$.

        Args:
            y (B, H, D): input tensor

        Returns:
            x (B, H, D): transformed tensor
            ldj (H): log determinant of the Jacobian
        """
        L = self.get_cholesky()

        # compute z = L⁻¹(x-μ)
        y = y - self.means
        y = y.moveaxis(0, -1)  # (B, H, D) -> (H, D, B)
        # (H, D, D), (H, D, B) -> (H, D, B)
        u = solve_triangular(L, y, upper=False)
        u = u.moveaxis(-1, 0)  # (H, D, B) -> (B, H, D)

        # compute log |det L⁻¹| = - log |det L| = log ∏ᵢ Lᵢᵢ
        ldj = -L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return u, ldj

    def sample(self, size: int | tuple[int, ...] = (), /) -> Tensor:
        r"""Sample from the model.

        Args:
            size (int | tuple[int, ...]): size of the sample

        Returns:
            u (..., H, D): sample
        """
        shape = (size,) if isinstance(size, int) else size
        shape = (*shape, self.num_heads, self.num_features)
        z = torch.randn(*shape, device=self.normalization_constant.device)
        u = self.forward(z)
        self.samples = u  # store buffer for post-hoc analysis
        return u

    def log_prob(self, u: Tensor, /) -> Tensor:
        r"""Compute the log probability of the input.

        Args:
            u (..., H, D): input tensor

        Returns:
            log_prob (..., H): log likelihood
        """
        self.latents = u  # store buffer for post-hoc analysis

        # parse through the gaussians
        z, ldj = self.inverse(u)

        # compute the base log probability
        # ½*log(2π) + ½\log(σ²) + ½‖x-μ‖²/σ² = ½*log(2π) +  ½‖x‖²
        log_prob = self.normalization_constant + 0.5 * vecdot(z, z, dim=-1)  # (..., H)
        log_prob = log_prob - ldj  # (..., H)
        self.log_probs = log_prob  # store buffer for post-hoc analysis
        return log_prob

    def marginalize(self, indices: Tensor, /) -> MultiHeadGaussian:
        r"""Marginalize the distribution over the given indices."""
        # (M, D) -> (M, D), (M, D, D) -> (M, D, D)
        idx = indices.tolist()
        assert len(set(idx)) == len(indices), "Indices must be unique"

        # initialize the marginal model
        marg_model = MultiHeadGaussian(n_feats=len(idx), n_heads=self.num_heads)

        # set the marginal models parameters

        # validate the marginalization
        assert marg_model.means.shape == (self.num_heads, len(idx))
        assert marg_model.covs.shape == (self.num_heads, len(idx), len(idx))
        marg_means = self.means[..., idx]
        marg_covs = self.scale_tril[..., idx, :][..., idx]
        assert torch.allclose(marg_model.means, marg_means)
        assert torch.allclose(marg_model.scale_tril, marg_covs)

        return marg_model
