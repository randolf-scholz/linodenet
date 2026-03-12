__all__ = [
    "DEVICES",
    "DTYPES",
    "SEEDS",
    "SEED",
    "TestCase",
    "make_quasi_gaussian",
    "make_rank_one_matrix",
    "make_diagonal_matrix",
    "make_orthogonal_matrix",
]

from typing import Final, NamedTuple

import torch
from numpy.random import default_rng
from scipy.stats import ortho_group
from torch import Tensor, nn

from linodenet_special.marchenko_pastur import MarchenkoPastur

DEVICES: Final[list[str]] = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DTYPES: Final[list[torch.dtype]] = [torch.float32, torch.float64]
SEEDS: Final[list[int]] = [1000, 1001, 1002, 1003, 1004]
SEED: Final[int] = 0


class TestCase(NamedTuple):
    r"""Test matrix with known SVD."""

    U: Tensor  # left singular vectors (..., m, k)
    S: Tensor  # singular values (..., k)
    V: Tensor  # right singular vectors (..., n, k)

    @property
    def value(self) -> nn.Parameter:
        r"""Reconstruct the matrix A = U diag(S) Vᵀ."""
        A = torch.einsum("...mk, ...k, ...nk -> ...mn", self.U, self.S, self.V)
        return nn.Parameter(A, requires_grad=True)

    @property
    def spectral_norm(self) -> Tensor:
        r"""Return the spectral norm of the matrix."""
        return self.S.max(dim=-1).values  # (...,)

    @property
    def spectral_norm_gradient(self):
        r"""Return the gradient of the spectral norm of the matrix."""
        u, _, v = self.singular_triplet
        return torch.einsum("...m, ...n -> ...mn", u, v)

    @property
    def singular_triplet(self) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return the maximum singular value and its vectors."""
        U, S, V = self.U, self.S, self.V
        idx_star = S.argmax(dim=-1, keepdim=True)  # (..., 1)
        s = S.gather(dim=-1, index=idx_star)  # (..., 1)
        idx_vec = idx_star.unsqueeze(-1)  # (..., 1, 1)
        u = U.gather(dim=-1, index=idx_vec.expand(*U.shape[:-1], 1))  # (..., m, 1)
        v = V.gather(dim=-1, index=idx_vec.expand(*V.shape[:-1], 1))  # (..., n, 1)
        return u.squeeze(-1), s.squeeze(-1), v.squeeze(-1)


def make_quasi_gaussian(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> TestCase:
    r"""Generates a random m×n matrix with known spectral norm and gradient.

    We sample random singular values from an MP distribution,
    as well as random orthogonal matrices U and V from the haar distribution.

    Values should approximately be sampled from N(0, 1/n)
    """
    m, n = shape
    k = min(m, n)
    gamma = m / n
    rng = default_rng(seed)

    # only the first k vectors
    U_numpy = ortho_group(m).rvs(random_state=rng)[..., :k]
    V_numpy = ortho_group(n).rvs(random_state=rng)[..., :k]
    U = torch.from_numpy(U_numpy).to(dtype=dtype, device=device)
    V = torch.from_numpy(V_numpy).to(dtype=dtype, device=device)
    dist = MarchenkoPastur(gamma=gamma, sigma2=1.0, validate_args=False)
    S = dist.sample([k]).to(dtype=dtype, device=device).sqrt()
    return TestCase(U=U, S=S, V=V)


def make_rank_one_matrix(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> TestCase:
    r"""Generate a rank-one matrix with known SVD."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed or 0)

    m, n = shape
    sigma = 1000 * torch.rand((), device=device, generator=generator) + 1
    u = torch.randn(m, device=device, generator=generator)
    u = u / u.norm()  # (m,)
    U = u.unsqueeze(-1).to(dtype=dtype)  # (m, 1)
    v = torch.randn(n, device=device, generator=generator)
    v = v / v.norm()  # (n,)
    V = v.unsqueeze(-1).to(dtype=dtype)  # (n, 1)
    S = sigma.unsqueeze(-1).to(dtype=dtype)  # (1,)
    return TestCase(U=U, S=S, V=V)


def make_diagonal_matrix(
    size: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> TestCase:
    r"""Generate a diagonal matrix with known SVD."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed or 0)

    S = 10 * torch.randn(size, dtype=dtype, device=device, generator=generator)
    U = torch.eye(size, device=device, dtype=dtype)
    V = torch.eye(size, device=device, dtype=dtype)
    return TestCase(U=U, S=S, V=V)


def make_orthogonal_matrix(
    size: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> TestCase:
    r"""Generate an orthogonal matrix with known SVD."""
    rng = default_rng(seed)
    U_numpy = ortho_group.rvs(size, random_state=rng)
    U = torch.from_numpy(U_numpy).to(dtype=dtype, device=device)
    V = torch.eye(size, device=device, dtype=dtype)
    S = torch.ones(size, device=device, dtype=dtype)
    return TestCase(U=U, S=S, V=V)
