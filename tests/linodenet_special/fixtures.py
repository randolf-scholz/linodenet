__all__ = [
    "DEVICES",
    "DTYPES",
    "SEEDS",
    "SEED",
    "ExampleWithKnownSVD",
    "Fixture",
    "make_test_case_quasi_gaussian",
    "make_test_case_rank_one",
    "make_test_case_diagonal",
    "make_test_case_repeated_singular_values",
]

import warnings
from functools import cache
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


class Fixture:
    ATOL = 1e-6
    RTOL = 1e-6

    def assert_upper_bounded(
        self,
        value: Tensor | float,
        bound: Tensor | float,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| < (1+rtol) |right| + atol."""
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        bound = torch.as_tensor(bound)
        upper_bound = (1 + rtol) * bound + atol
        assert upper_bound >= 0.0
        ok = value <= upper_bound

        abs_violation = (value - upper_bound).clamp_min(0)
        rel_violation = abs_violation / upper_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            msg = (
                f"Values exceed bound! "
                f"\n\tvalue: {value.tolist()}"
                f"\n\tbound: {bound.tolist()}"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
            )
            raise AssertionError(msg)

        if warn_loose and (max_abs_err < 1e-3 or max_rel_err < 1e-3):
            warnings.warn(
                f"Bounds are loose:"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})",
                stacklevel=2,
            )

    def assert_lower_bounded(
        self,
        value: Tensor | float,
        bound: Tensor | float,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| ≥ (1-rtol) |right| - atol."""
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        bound = torch.as_tensor(bound)
        lower_bound = (1 - rtol) * bound - atol
        assert (1 - rtol) >= 0.0
        ok = value >= lower_bound

        abs_violation = (lower_bound - value).clamp_min(0)
        rel_violation = abs_violation / lower_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            msg = (
                f"Values exceed bound! "
                f"\n\tvalue: {value.tolist()}"
                f"\n\tbound: {bound.tolist()}"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
            )
            raise AssertionError(msg)

        if warn_loose and (max_abs_err < 1e-3 or max_rel_err < 1e-3):
            warnings.warn(
                f"Bounds are loose:"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})",
                stacklevel=2,
            )

    def assert_close(
        self,
        value: Tensor | float,
        true_value: Tensor | float,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> None:
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        true_value = torch.as_tensor(true_value)
        residual = (value - true_value).abs()
        magnitude = true_value.abs()
        ok = residual <= rtol * magnitude + atol

        if not ok.all():
            max_abs_err = residual.max().item()
            mean_abs_err = residual.mean().item()
            median_abs_err = residual.median().item()
            max_rel_err = (residual / magnitude).max().item()
            mean_rel_err = (residual / magnitude).nanmean().item()
            median_rel_err = (residual / magnitude).nanmedian().item()
            msg = (
                f"Values not close! "
                # f"\n\tleft:  {value.tolist()}"
                # f"\n\tright: {true_value.tolist()}"
                f"\n\tmax    abs error={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs error={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs error={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel error={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel error={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel error={median_rel_err:8.2e}  (expected {rtol})"
            )
            raise AssertionError(msg)


class ExampleWithKnownSVD(NamedTuple):
    r"""Test matrix with known SVD."""

    U: Tensor  # left singular vectors (..., m, k)
    S: Tensor  # singular values (..., k)
    V: Tensor  # right singular vectors (..., n, k)

    @property
    def sigma(self) -> Tensor:
        assert (self.S[0] >= self.S[1:]).all()
        return self.S[0]

    @property
    def u(self) -> Tensor:
        return self.U[..., 0]

    @property
    def v(self) -> Tensor:
        return self.V[..., 0]

    @property
    @cache  # noqa: B019
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
        r"""Return the gradient of the spectral norm of the matrix.

        The gradient is analytically given as:

        ..math:: \dv{‖A‖₂}{A} = uvᵀ
        """
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

    def singular_triplet_vjp(self, g_u: Tensor, g_s: Tensor, g_v: Tensor) -> Tensor:
        r"""Return the VJP of the dominant singular triplet with respect to $A$.

        For a simple dominant singular triplet $(u, σ, v)$ satisfying
        $Av = σu$, $Aᵀu = σv$, $uᵀu = 1$, and $vᵀv = 1$, the backward map is

        .. math::

           g_A = gₛ\,uvᵀ + (𝕀ₘ - uuᵀ) p vᵀ + u qᵀ (𝕀ₙ - vvᵀ),

        where $p ∈ ℝᵐ$ and $q ∈ ℝⁿ$ solve the augmented linear system

        .. math::

           \begin{bmatrix}
               σ𝕀ₘ & -A & u & 0 \\
               -Aᵀ & σ𝕀ₙ & 0 & v
           \end{bmatrix}
           \begin{bmatrix}
               p \\ q \\ μ \\ ν
           \end{bmatrix}
           =
           \begin{bmatrix}
               gᵤ \\ gᵥ
           \end{bmatrix}.

        Equivalently, the individual VJP contributions are

        .. math::

           gₛᵀ \frac{∂σ}{∂A} = gₛ\,uvᵀ,
           \qquad
           gᵤᵀ \frac{∂u}{∂A} = (𝕀ₘ - uuᵀ) p vᵀ,
           \qquad
           gᵥᵀ \frac{∂v}{∂A} = u qᵀ (𝕀ₙ - vvᵀ).

        This formula assumes the dominant singular value is isolated. When the top
        singular value is repeated, the singular vectors are not uniquely defined
        and the full triplet VJP is not unique either.
        """
        A = self.value
        u, sigma, v = self.singular_triplet

        g_sigma_out = g_s * torch.outer(u, v)
        if not (g_u.any() or g_v.any()).item():
            return g_sigma_out

        m, n = A.shape
        zero_u = torch.zeros((m, 1), dtype=A.dtype, device=A.device)
        zero_v = torch.zeros((n, 1), dtype=A.dtype, device=A.device)
        eye_m = torch.eye(m, dtype=A.dtype, device=A.device)
        eye_n = torch.eye(n, dtype=A.dtype, device=A.device)

        k_top = torch.cat((sigma * eye_m, -A, u.unsqueeze(-1), zero_u), dim=1)
        k_bottom = torch.cat((-A.T, sigma * eye_n, zero_v, v.unsqueeze(-1)), dim=1)
        k_mat = torch.cat((k_top, k_bottom), dim=0)
        c_vec = torch.cat((g_u, g_v), dim=0)

        x = torch.linalg.lstsq(k_mat, c_vec).solution
        p = x[:m]
        q = x[m : m + n]

        g_u_out = torch.outer(p - torch.dot(u, p) * u, v)
        g_v_out = torch.outer(u, q - torch.dot(v, q) * v)
        return g_sigma_out + g_u_out + g_v_out

    @staticmethod
    def dyad_loss(g_matrix: Tensor, u, v) -> Tensor:
        r"""Return the gauge-invariant loss induced by the dominant dyad $uvᵀ$."""
        return torch.einsum("mn, mn ->", torch.outer(u, v), g_matrix)

    @classmethod
    def singular_triplet_loss(
        cls, g_sigma: Tensor, g_matrix: Tensor, sigma: Tensor, u: Tensor, v: Tensor
    ) -> Tensor:
        r"""Return the gauge-invariant scalar loss for a singular triplet."""
        return g_sigma * sigma + cls.dyad_loss(g_matrix, u, v)


def make_test_case_quasi_gaussian(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> ExampleWithKnownSVD:
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
    S = dist.sample_positive([k]).to(dtype=dtype, device=device).sqrt()
    # ensure a minimum gap between singular values.
    S = torch.sort(S, descending=True).values
    eps = 1e-6 * max(1.0, S.max().item())
    S = S + eps * torch.arange(k, 0, -1, dtype=dtype, device=device)
    assert (S[0] > S[1:] + eps).all()
    return ExampleWithKnownSVD(U=U, S=S, V=V)


def make_test_case_rank_one(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> ExampleWithKnownSVD:
    r"""Generate a rank-one matrix with known SVD."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed or 0)

    m, n = shape
    sigma = 10 * torch.rand((), device=device, dtype=dtype, generator=generator) + 1
    u = torch.randn(m, device=device, dtype=dtype, generator=generator)
    u = u / u.norm()  # (m,)
    U = u.unsqueeze(-1)  # (m, 1)
    v = torch.randn(n, device=device, dtype=dtype, generator=generator)
    v = v / v.norm()  # (n,)
    V = v.unsqueeze(-1)  # (n, 1)
    S = sigma.unsqueeze(-1)  # (1,)
    return ExampleWithKnownSVD(U=U, S=S, V=V)


def make_test_case_diagonal(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> ExampleWithKnownSVD:
    r"""Generate a diagonal matrix with known SVD."""
    m, n = shape
    generator = torch.Generator(device=device)
    generator.manual_seed(seed or 0)

    k = min(m, n)
    diag = 10 * torch.randn(k, dtype=dtype, device=device, generator=generator)
    signs = torch.sign(diag)
    U = torch.eye(m, device=device, dtype=dtype)[:, :k]
    V = torch.eye(n, device=device, dtype=dtype)[:, :k] * signs
    # ensure a minimum gap between singular values.
    S = torch.sort(diag.abs(), descending=True).values
    eps = 1e-6 * max(1.0, S.max().item())
    S = S + eps * torch.arange(k, 0, -1, dtype=dtype, device=device)
    assert (S[0] > S[1:] + eps).all()
    return ExampleWithKnownSVD(U=U, S=S, V=V)


def make_test_case_repeated_singular_values(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> ExampleWithKnownSVD:
    r"""Generate an orthogonal matrix with known SVD."""
    m, n = shape
    k = min(m, n)
    rng = default_rng(seed)
    U_numpy = ortho_group.rvs(m, random_state=rng)[:, :k]
    V_numpy = ortho_group.rvs(n, random_state=rng)[:, :k]
    U = torch.from_numpy(U_numpy).to(dtype=dtype, device=device)
    V = torch.from_numpy(V_numpy).to(dtype=dtype, device=device)
    S = torch.ones(k, device=device, dtype=dtype)
    return ExampleWithKnownSVD(U=U, S=S, V=V)
