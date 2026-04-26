r"""Vector and matrix manifolds with explicit base-pointed exp/log maps.

This module intentionally exposes manifold operations at a general base point
$x$, i.e. maps of the form $expₓ(v)$ with $v ∈ TₓM$.

Only manifolds with a standard closed-form choice of metric are implemented:

- `SphereManifold`: round metric on the unit sphere $Sⁿ⁻¹$
- `SpecialOrthogonalManifold`: canonical bi-invariant metric on $SO(n)$
- `PositiveDefiniteManifold`: affine-invariant metric on $SPD(n)$
"""

__all__ = [
    "RiemannManifold",
    "MatrixManifold",
    "VectorManifold",
    "MatrixLieGroup",
    "ManifoldBase",
    "MatrixLieGroupBase",
    # Classes
    "SphereManifold",
    "PositiveDefiniteManifold",
    "SpecialOrthogonalManifold",
]

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.linalg import vecdot, vector_norm
from torch.nn.functional import normalize

from linodenet.domains import MatrixDomains, VectorDomains
from linodenet_special import matrix_log, matrix_sqrt
from signatures import signature


@runtime_checkable
class RiemannManifold(Protocol):
    r"""Protocol for Riemannian manifolds with explicit tangent and exp/log maps."""

    @abstractmethod
    @signature("(...) -> (...)")
    def project_manifold(self, x: Tensor, /) -> Tensor:
        r"""Project an ambient tensor onto the manifold $M$."""
        ...

    @abstractmethod
    @signature("[(...), (...)] -> (...)")
    def project_tangent(self, x: Tensor, v: Tensor, /) -> Tensor:
        r"""Project an ambient tensor onto the tangent space $TₓM$."""
        ...

    @abstractmethod
    @signature("[(...), (...)] -> (...)")
    def exp(self, x: Tensor, v: Tensor, /) -> Tensor:
        r"""Map a tangent vector $v ∈ TₓM$ to the manifold via $expₓ(v)$."""
        ...

    @abstractmethod
    @signature("[(...), (...)] -> (...)")
    def log(self, x: Tensor, y: Tensor, /) -> Tensor:
        r"""Map a manifold point $y ∈ M$ back to a tangent vector in $TₓM$."""
        ...

    @abstractmethod
    @signature("[(...), (...)] -> (...)")
    def retraction(self, x: Tensor, v: Tensor, /) -> Tensor:
        r"""Map a tangent vector $v ∈ TₓM$ to $M$ via a retraction at $x$."""
        ...


@runtime_checkable
class VectorManifold(RiemannManifold, Protocol):
    r"""Protocol for vector manifolds."""

    MANIFOLD: ClassVar[VectorDomains]


@runtime_checkable
class MatrixManifold(RiemannManifold, Protocol):
    r"""Protocol for matrix manifolds."""

    MANIFOLD: ClassVar[MatrixDomains]


@runtime_checkable
class MatrixLieGroup(MatrixManifold, Protocol):
    r"""Protocol for matrix Lie groups with a fixed Lie algebra at the identity."""

    LIE_ALGEBRA: ClassVar[MatrixDomains]

    @abstractmethod
    @signature("(..., n, n) -> (..., n, n)")
    def project_algebra(self, x: Tensor, /) -> Tensor:
        r"""Project an ambient matrix onto the Lie algebra $𝔤 = TᴵG$."""
        ...

    @abstractmethod
    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def to_algebra(self, x: Tensor, v: Tensor, /) -> Tensor:
        r"""Identify a tangent vector $v ∈ TₓG$ with an algebra element in $𝔤$."""
        ...

    @abstractmethod
    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def from_algebra(self, x: Tensor, a: Tensor, /) -> Tensor:
        r"""Map an algebra element $a ∈ 𝔤$ to the tangent space $TₓG$."""
        ...

    @abstractmethod
    @signature("(..., n, n) -> (..., n, n)")
    def exp_identity(self, a: Tensor, /) -> Tensor:
        r"""Map an algebra element $a ∈ 𝔤$ to the group via the identity exponential."""
        ...

    @abstractmethod
    @signature("(..., n, n) -> (..., n, n)")
    def log_identity(self, g: Tensor, /) -> Tensor:
        r"""Map a group element $g ∈ G$ to the Lie algebra $𝔤$ via the identity logarithm."""
        ...


class ManifoldBase(nn.Module, RiemannManifold):
    r"""Base class for Riemannian manifolds with `expₓ(v)`-style operations."""

    @signature("[(...), (...)] -> (...)")
    def forward(self, x: Tensor, v: Tensor, /) -> Tensor:
        return self.exp(x, v)

    @signature("[(...), (...)] -> (...)")
    def retraction(self, x: Tensor, v: Tensor, /) -> Tensor:
        return self.exp(x, v)


class MatrixLieGroupBase(ManifoldBase, MatrixLieGroup):
    r"""Base class for matrix Lie groups using left trivialization."""

    MANIFOLD: ClassVar[MatrixDomains] = MatrixDomains.INVERTIBLE
    LIE_ALGEBRA: ClassVar[MatrixDomains] = MatrixDomains.SQUARE

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def to_algebra(self, x: Tensor, v: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        return self.project_algebra(torch.linalg.solve(x, v))

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def from_algebra(self, x: Tensor, a: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        return x @ self.project_algebra(a)

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def project_tangent(self, x: Tensor, v: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        return self.from_algebra(x, self.project_algebra(self.to_algebra(x, v)))

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def exp(self, x: Tensor, v: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        tangent = self.project_tangent(x, v)
        algebra = self.project_algebra(self.to_algebra(x, tangent))
        return self.project_manifold(x @ self.exp_identity(algebra))

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def log(self, x: Tensor, y: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        y = self.project_manifold(y)
        relative = torch.linalg.solve(x, y)
        algebra = self.project_algebra(self.log_identity(relative))
        return self.from_algebra(x, algebra)


class SphereManifold(ManifoldBase):
    r"""The unit sphere $Sⁿ⁻¹ ⊂ ℝⁿ$ with the round Riemannian metric."""

    MANIFOLD: ClassVar[VectorDomains] = VectorDomains.UNIT_VECTOR

    @signature("(..., n) -> (..., n)")
    def project_manifold(self, x: Tensor, /) -> Tensor:
        return normalize(x, dim=-1)

    @signature("[(..., n), (..., n)] -> (..., n)")
    def project_tangent(self, x: Tensor, v: Tensor, /) -> Tensor:
        x = self.project_manifold(x)
        inner = vecdot(x, v, dim=-1).unsqueeze(-1)
        return v - inner * x

    @signature("[(..., n), (..., n)] -> (..., n)")
    def exp(self, x: Tensor, v: Tensor, /) -> Tensor:
        r"""Compute the round-metric exponential on the sphere.

        .. math:: expₓ(v) = \cos(‖v‖)x + \sin(‖v‖)(v/‖v‖)
        """
        x = self.project_manifold(x)
        v = self.project_tangent(x, v)
        theta = vector_norm(v, dim=-1, keepdim=True)
        direction = v / theta.clamp_min(torch.finfo(x.dtype).eps)
        y = torch.cos(theta) * x + torch.sin(theta) * direction
        return self.project_manifold(y)

    @signature("[(..., n), (..., n)] -> (..., n)")
    def log(self, x: Tensor, y: Tensor, /) -> Tensor:
        r"""Compute the round-metric logarithm on the sphere.

        .. math:: logₓ(y) = (y - \cos(θ)x)\frac{θ}{\sin(θ)}

        where $θ = \arccos(⟨x, y⟩)$.
        """
        x = self.project_manifold(x)
        y = self.project_manifold(y)
        eps = torch.finfo(x.dtype).eps
        tangent = self.project_tangent(x, y)
        sine = vector_norm(tangent, dim=-1, keepdim=True)
        cosine = vecdot(x, y, dim=-1).clamp(-1, 1).unsqueeze(-1)
        theta = torch.atan2(sine, cosine)
        scale = torch.where(sine > eps, theta / sine, torch.zeros_like(theta))
        return scale * tangent


class PositiveDefiniteManifold(ManifoldBase):
    r"""The SPD manifold with the affine-invariant Riemannian metric."""

    MANIFOLD: ClassVar[MatrixDomains] = MatrixDomains.POSITIVE_DEFINITE

    def _sqrt_and_inv_sqrt(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self._ensure_positive_definite(x)
        sqrt_x = matrix_sqrt(x).real.to(dtype=x.dtype)
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        eye = eye.expand(x.shape[:-2] + eye.shape).clone()
        inv_sqrt_x = torch.linalg.solve(sqrt_x, eye)
        return sqrt_x, inv_sqrt_x

    def _ensure_positive_definite(self, x: Tensor, /) -> Tensor:
        x = (x + x.mT) / 2
        eigenvalues, V = torch.linalg.eigh(x)
        eigenvalues = eigenvalues.clamp_min(torch.finfo(x.dtype).eps)
        return torch.einsum("...ik, ...k, ...jk -> ...ij", V, eigenvalues, V)

    @signature("(..., n, n) -> (..., n, n)")
    def project_manifold(self, x: Tensor, /) -> Tensor:
        x = (x + x.mT) / 2
        y = torch.matrix_exp(x)
        return (y + y.mT) / 2

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def project_tangent(self, _: Tensor, v: Tensor, /) -> Tensor:
        return (v + v.mT) / 2

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def exp(self, x: Tensor, v: Tensor, /) -> Tensor:
        v = self.project_tangent(x, v)
        sqrt_x, inv_sqrt_x = self._sqrt_and_inv_sqrt(x)
        mid = inv_sqrt_x @ v @ inv_sqrt_x
        mid = (mid + mid.mT) / 2
        return self._ensure_positive_definite(sqrt_x @ torch.matrix_exp(mid) @ sqrt_x)

    @signature("[(..., n, n), (..., n, n)] -> (..., n, n)")
    def log(self, x: Tensor, y: Tensor, /) -> Tensor:
        sqrt_x, inv_sqrt_x = self._sqrt_and_inv_sqrt(x)
        mid = self._ensure_positive_definite(inv_sqrt_x @ y @ inv_sqrt_x)
        log_mid = matrix_log(mid).real.to(dtype=mid.dtype)
        return self.project_tangent(x, sqrt_x @ log_mid @ sqrt_x)


class SpecialOrthogonalManifold(MatrixLieGroupBase):
    r"""The special orthogonal group with the Frobenius bi-invariant metric."""

    MANIFOLD: ClassVar[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL
    LIE_ALGEBRA: ClassVar[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC

    @signature("(..., n, n) -> (..., n, n)")
    def project_manifold(self, x: Tensor, /) -> Tensor:
        u, _, vh = torch.linalg.svd(x)
        size = x.shape[-1]
        correction = torch.eye(size, dtype=x.dtype, device=x.device)
        correction = correction.expand(x.shape[:-2] + correction.shape).clone()
        det = torch.linalg.det(u @ vh)
        correction[..., -1, -1] = torch.where(det < 0, -1, 1).to(dtype=x.dtype)
        return u @ correction @ vh

    @signature("(..., n, n) -> (..., n, n)")
    def project_algebra(self, x: Tensor, /) -> Tensor:
        return (x - x.mT) / 2

    @signature("(..., n, n) -> (..., n, n)")
    def exp_identity(self, a: Tensor, /) -> Tensor:
        return torch.matrix_exp(self.project_algebra(a))

    @signature("(..., n, n) -> (..., n, n)")
    def log_identity(self, g: Tensor, /) -> Tensor:
        return self.project_algebra(matrix_log(g).real.to(dtype=g.dtype))
