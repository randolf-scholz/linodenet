r"""Closed-form state propagation for linear continuous-time dynamics.

The module provides functional and `nn.Module` interfaces for deterministic
linear ODE flows and linear-Gaussian SDE flows. All propagators evaluate the
state at one or more time deltas relative to the supplied initial state.
"""

__all__ = [
    "LinearFlow",
    "LinearGaussianFlow",
    "linear_gaussian_flow",
    "linear_flow",
]

from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet.initializations import resolve_kernel_initialization
from linodenet.mappings import PositiveDefinite
from linodenet.nn import ReZero
from linodenet.nn.parametrize import register_parametrization
from linodenet.parametrizations import resolve_matrix_parametrization
from signatures import signature

from .base import ContinuousFlow


@signature("[(..., $n), (..., d), (d, d), (d,)?] -> (..., $n, d)")
def linear_flow(
    timedeltas: Tensor,  # (..., $n)
    x0: Tensor,  # (..., d)
    kernel: Tensor,  # (d, d)
    bias: Optional[Tensor] = None,
    /,
) -> Tensor:  # (..., $n, d)
    r"""Propagate an affine linear ODE by a matrix exponential.

    Solves the time-homogeneous system

    .. math:: dxₜ/dt = Axₜ + b

    for each requested time delta $t$. If `bias` is omitted, the solution is
    $xₜ = eᴬᵗx₀$. Otherwise, the affine term is computed through the
    augmented matrix exponential

    .. math:: \exp\left(\bmat{ A & b \\ 0 & 0 }t\right) = \bmat{ eᴬᵗ & φ₁(At)bt \\ 0 & 1 }

    where $φ₁(Z) = ∑ₖ₌₀^∞ Zᵏ/(k+1)!$.

    Args:
        timedeltas: Evaluation time deltas, with shape `(..., n)`.
        x0: Initial state, with shape `(..., d)`.
        kernel: Linear drift matrix $A$, with shape `(d, d)`.
        bias: Optional affine drift vector $b$, with shape `(d,)`.

    Returns:
        Propagated states at each requested time delta, with shape `(..., n, d)`.
    """
    if bias is None:
        Adt = torch.einsum("..., kl -> ...kl", timedeltas, kernel)
        expAdt = torch.linalg.matrix_exp(Adt)  # (*bs, n)
        return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0)

    # use augmented block matrix [[A, b], [0, 0]]
    n = bias.shape[-1]
    M = torch.cat(
        [
            torch.cat([kernel, bias.unsqueeze(-1)], dim=-1),
            torch.zeros((1, n + 1), device=kernel.device, dtype=kernel.dtype),
        ],
        dim=0,
    )
    Mdt = torch.einsum("..., kl -> ...kl", timedeltas, M)
    expMdt = torch.linalg.matrix_exp(Mdt)
    expAdt = expMdt[..., :n, :n]
    phi1bt = expMdt[..., :n, -1]
    return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0) + phi1bt


class LinearFlow(nn.Module, ContinuousFlow):
    r"""Learnable affine linear flow with optional kernel constraints.

    The module solves

    .. math:: x_{t+∆t} = e^{A∆t}xₜ

    or, when `use_bias=True`,

    .. math:: x_{t+∆t} = e^{A∆t}xₜ + ∫₀^{∆t} e^{A(∆t-s)} b \dd{s}.

    The drift matrix may be constrained by a matrix parametrization $π$, for
    example to enforce skew-symmetric or otherwise structured dynamics. When
    enabled, the ReZero gate $ρ$ scales the parametrized drift, so the effective
    system matrix is $ρ(π(A))$.
    """

    # Constants
    input_size: Final[int]
    r"""CONST: State dimensionality `d`."""
    output_size: Final[int]
    r"""CONST: Output state dimensionality, equal to `input_size`."""
    use_rezero: Final[bool]
    r"""CONST: Whether to apply a `ReZero` gate to the effective kernel."""
    use_bias: Final[bool]
    r"""CONST: Whether to include a learnable affine drift vector."""

    # Parameters
    weight: Tensor
    r"""PARAM: Cached drift matrix after parametrization and before ReZero gating."""
    bias: Tensor | None
    r"""PARAM: Optional learnable affine drift vector."""
    rezero: nn.Module
    r"""MODULE: ReZero gate or identity map applied to the effective kernel."""
    # Buffers
    kernel: Tensor
    r"""BUFFER: Cached effective drift matrix used by the latest forward pass."""
    kernel_initialization: nn.Module
    r"""MODULE: Sampler used to initialize the raw drift matrix."""
    kernel_parametrization: nn.Module | None
    r"""MODULE: Optional parametrization applied to the raw drift matrix."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "kernel_initialization": self.kernel_initialization,
            "kernel_parametrization": self.kernel_parametrization,
            "use_rezero": self.use_rezero,
            "use_bias": self.use_bias,
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
        use_rezero: bool = True,
        use_bias: bool = False,
    ) -> None:
        r"""Initialize the linear flow.

        Args:
            input_size: State dimensionality `d`.
            kernel_initialization: Initialization specification for the drift matrix.
                Strings and tensors are resolved by `resolve_kernel_initialization`.
            kernel_parametrization: Optional matrix parametrization specification
                applied to the raw drift matrix.
            use_rezero: Whether to gate the effective drift matrix with `ReZero`.
            use_bias: Whether to add a learnable affine drift vector.
        """
        super().__init__()
        ContinuousFlow.__init__(self, input_shape=(input_size,))

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self.use_rezero = use_rezero
        self.use_bias = use_bias
        self.kernel_initialization = resolve_kernel_initialization(
            input_size, kernel_initialization
        )

        # initialize parameters
        self.weight = nn.Parameter(self.kernel_initialization())
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(input_size)) if self.use_bias else None,
        )

        # apply parametrization if given
        self.kernel_parametrization = resolve_matrix_parametrization(
            kernel_parametrization
        )
        register_parametrization(self, "weight", self.kernel_parametrization)

        # initialize buffers
        self.rezero = ReZero() if self.use_rezero else nn.Identity()
        self.register_buffer("kernel", self.rezero(self.weight))

    @signature("[(...), (..., d)] -> (..., d)")
    def step(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Propagate the flow for one scalar time delta per batch item.

        Args:
            timedeltas: Scalar time delta or batched scalar deltas, with shape `(...)`.
            x0: Initial state, with shape `(..., d)`.

        Returns:
            Propagated state after the requested delta, with shape `(..., d)`.
        """
        return self.forward(timedeltas.unsqueeze(-1), x0).squeeze(-2)

    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forward(self, timedeltas: Tensor, x0: Tensor) -> Tensor:
        r"""Evaluate the flow at one or more time deltas.

        The cached effective kernel is refreshed from the parametrized weight and
        ReZero gate before evaluating the closed-form solution.

        Args:
            timedeltas: Evaluation time deltas, with shape `(..., n)`.
            x0: Initial state, with shape `(..., d)`.

        Returns:
            Propagated states at each requested delta, with shape `(..., n, d)`.
        """
        # update buffer
        self.kernel = self.rezero(self.weight)
        return linear_flow(timedeltas, x0, self.kernel, self.bias)

    @signature("[(..., $n), (..., d)] -> (..., $n, d)")
    def forecast(
        self,
        timestamps: Tensor,
        x0: Tensor,
        *,
        t0: Tensor | float,
    ) -> Tensor:
        r"""Evaluate the flow at absolute timestamps relative to an origin.

        Converts timestamps to time deltas via `timestamps - t0` and delegates to
        `forward`.

        Args:
            timestamps: Evaluation timestamps, with shape `(..., n)`.
            x0: Initial state at `t0`, with shape `(..., d)`.
            t0: Reference timestamp of `x0`.

        Returns:
            Propagated states at each timestamp, with shape `(..., n, d)`.
        """
        return self(timestamps - t0, x0)


def linear_gaussian_flow(
    delta_t: Tensor,  # (..., $n)
    z0: tuple[Tensor, Tensor],  # (..., d), (..., d, d)
    A: Tensor,  # (d, d)
    Q: Tensor,  # (d, d)
    b: Optional[Tensor] = None,  # (d,) | None
    /,
) -> tuple[Tensor, Tensor]:  # (...., $n, d), (..., $n, d, d)
    r"""Propagate a Gaussian state through a linear-Gaussian SDE.

    The dynamics are

    .. math:: \dd{Zₜ} = (AZₜ + b)\dd{t} + L\dd{Wₜ}

    with diffusion covariance $Q = LLᵀ$. If $Z₀ ∼ 𝓝(μ₀, Σ₀)$, then
    $Zₜ ∼ 𝓝(μₜ, Σₜ)$ for all time deltas $t$, where

    .. math::
        μₜ &= eᴬᵗμ₀ + ∫₀ᵗ e^{(t-s)A}b \dd{s} \\
        Σₜ &= eᴬᵗΣ₀e^{tAᵀ} + ∫₀ᵗ e^{(t-s)A}Qe^{(t-s)Aᵀ} \dd{s}

    The covariance integral is evaluated with Van Loan's block-matrix
    exponential. For `b is None`, the implementation computes

    .. math:: \exp(\bmat{ A & Q \\ 0 & -Aᵀ }t) = \bmat{ F & C \\ 0 & F⁻ᵀ }

    and returns $Σₜ = FΣ₀Fᵀ + CFᵀ$.

    Args:
        delta_t: Evaluation time deltas, with shape `(..., n)`.
        z0: Initial Gaussian state `(μ₀, Σ₀)`, with shapes `(..., d)` and `(..., d, d)`.
        A: Linear drift matrix, with shape `(d, d)`.
        Q: Diffusion covariance matrix $LLᵀ$, with shape `(d, d)`.
        b: Optional affine drift vector, with shape `(d,)`.

    Returns:
        Pair `(μₜ, Σₜ)` containing propagated means and covariances, with
        shapes `(..., n, d)` and `(..., n, d, d)`.
    """
    mu_0, sigma_0 = z0
    n = A.shape[-1]

    if b is None:
        # [[A, Q], [0, -Aᵀ]]
        # -> [[F, C], [0, F⁻ᵀ]]
        M = torch.cat(
            [
                torch.cat([A, Q], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT], dim=-1),
            ],
            dim=0,
        )

    else:
        # use augmented block matrix
        # [[A, Q, b], [0, -Aᵀ, 0], [0, 0, 0]]
        # -> [[F, C, r], [0, F⁻ᵀ, 0], [0, 0, 1]]
        b = b.unsqueeze(-1)
        M = torch.cat(
            [
                torch.cat([A, Q, b], dim=-1),
                torch.cat([torch.zeros_like(A), -A.mT, torch.zeros_like(b)], dim=-1),
                torch.zeros((1, 2 * n + 1), dtype=A.dtype, device=A.device),
            ],
            dim=0,
        )

    # exp(M∆t) is a block matrix
    Mdt = torch.einsum("..., kl -> ...kl", delta_t, M)
    P = torch.linalg.matrix_exp(Mdt)
    F = P[..., :n, :n]  # top left block
    C = P[..., :n, n : 2 * n]  # top center block
    r = P[..., :n, -1] if b is not None else 0.0  # top right block
    mu_t = torch.einsum("...nkl, ...l -> ...nk", F, mu_0) + r  # eᴬᵗμ₀ + φ₁(At)bt
    sigma_t = F @ sigma_0.unsqueeze(-3) @ F.mT + C @ F.mT  # eᴬᵗΣ₀eᴬᵀᵗ + CFᵀ

    return mu_t, sigma_t


class LinearGaussianFlow(nn.Module, ContinuousFlow):
    r"""Learnable linear-Gaussian flow for Gaussian states.

    The module parameterizes the drift matrix $A$, diffusion covariance $Q$, and
    affine drift vector $b$ in the linear SDE

    .. math:: dZₜ = (AZₜ + b)\,dt + L\,dWₜ, \quad Q = LLᵀ.

    A Gaussian initial state remains Gaussian. For $Z₀ ∼ 𝓝(μ₀, Σ₀)$,
    the propagated state satisfies $Zₜ ∼ 𝓝(μₜ, Σₜ)$ with

    .. math::
        μₜ &= eᴬᵗμ₀ + ∫₀ᵗ e^{(t-s)A}b\,ds \\
        Σₜ &= eᴬᵗΣ₀e^{Aᵀt} + ∫₀ᵗ eᴬ⁽ᵗ⁻ˢ⁾ Q e^{Aᵀ(t-s)} ds

    The covariance integral is computed exactly with Van Loan's block-matrix
    exponential, as implemented by `linear_gaussian_flow`.

    References:
        - | Computing integrals involving the matrix exponential
          | Van Loan
          | IEEE Transactions on Automatic Control, 1978
    """

    A: Tensor
    r"""PARAM: Drift matrix of the linear SDE."""
    b: Tensor | None
    r"""PARAM: Affine drift vector of the linear SDE."""
    Q: Tensor  # C = √Q
    r"""PARAM: Positive-definite diffusion covariance matrix."""
    kernel_initialization: nn.Module
    r"""MODULE: Sampler used to initialize the drift matrix."""
    kernel_parametrization: nn.Module | None
    r"""MODULE: Optional parametrization applied to the drift matrix."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "kernel_initialization": self.kernel_initialization,
            "kernel_parametrization": self.kernel_parametrization,
        }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
    ) -> None:
        r"""Initialize the linear-Gaussian flow.

        Args:
            input_size: State dimensionality `d`.
            kernel_initialization: Initialization specification for the drift
                matrix. Strings and tensors are resolved by `resolve_kernel_initialization`.
            kernel_parametrization: Optional matrix parametrization specification
                applied to the drift matrix.
        """
        super().__init__()
        ContinuousFlow.__init__(self, input_shape=(input_size,))

        self.input_size = input_size
        self.kernel_initialization = resolve_kernel_initialization(
            input_size, kernel_initialization
        )

        self.A = nn.Parameter(self.kernel_initialization())
        self.kernel_parametrization = resolve_matrix_parametrization(
            kernel_parametrization
        )
        register_parametrization(self, "A", self.kernel_parametrization)
        self.Q = nn.Parameter(torch.randn(input_size, input_size))
        self.b = nn.Parameter(torch.randn(input_size))
        register_parametrization(self, "Q", PositiveDefinite())

    def forward(
        self, delta_t: Tensor, z_0: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        r"""Evaluate the Gaussian flow at one or more time deltas.

        Args:
            delta_t: Evaluation time deltas, with shape `(..., n)`.
            z_0: Initial Gaussian state `(μ₀, Σ₀)`,
                with shapes `(..., d)` and `(..., d, d)`.

        Returns:
            Pair `(μₜ, Σₜ)` containing propagated means and covariances,
            with shapes `(..., n, d)` and `(..., n, d, d)`.
        """
        return linear_gaussian_flow(delta_t, z_0, self.A, self.Q, self.b)
