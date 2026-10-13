r"""Kalman-inspired prototype state updates.

For a Gaussian prior $z ∼ 𝓝(μ, Σ)$ and observations $y = Hz + v$ with
independent noise $v ∼ 𝓝(0, R)$, the classical measurement update is

$$\begin{aligned}
    K &= ΣHᵀ(HΣHᵀ + R)⁻¹ \\
    μ' &= μ + K(y - Hμ)  \\
    Σ' &= Σ - KHΣ
\end{aligned}$$

For missing observations, let $Πₘ$ select the coordinates marked true in an
observation mask $m$. The restricted update uses

$$\begin{aligned}
    Sₘ &= Πₘ(HΣHᵀ + R)Πₘᵀ  \\
    Kₘ &= ΣHᵀΠₘᵀSₘ⁻¹  \\
    μ' &= μ + KₘΠₘ(y - Hμ)  \\
    Σ' &= Σ - KₘΠₘHΣ
\end{aligned}$$

For a new independent observation with the same noise model, the posterior
predictive mean and covariance are $Hμ'$ and $HΣ'Hᵀ + R$. These describe a
future observation, not the already conditioned measurement.

The learned variants below replace covariance-derived gains with free
matrices; they need not inherit the classical filter's covariance properties.

Note:
    These classes are prototypes. Their constructors currently pass unsupported
    state-size keywords to `nn.Module.__init__`. The expressions documented
    below describe their forward calculations.
"""

__all__ = [
    "PseudoKalmanUpdate",
    "NonLinearKalmanUpdate",
    "NonLinearUpdate",
]

from enum import Enum
from math import sqrt
from typing import Optional, SupportsFloat

import torch
from torch import Tensor, nn

from signatures import signature

from .base import VectorStateUpdate


class _Alpha(float, Enum):
    FIRST_VALUE = 0.0
    AVERAGE = 0.5
    LAST_VALUE = 1.0

    @classmethod
    def new(cls, arg: str | SupportsFloat) -> float:
        if isinstance(arg, str):
            return cls[arg.replace("-", "_").upper()]
        return float(arg)


class PseudoKalmanUpdate(nn.Module, VectorStateUpdate):
    r"""Prototype update with learned residual transforms.

    For an observed-coordinate diagonal mask $Dₘ$, the forward calculation is

    $$ x' = x - α(𝕀 + ε_B B)HᵀDₘ(𝕀 + ε_A A)Dₘ(Hx - y) $$

    The learnable residual scales $ε_A$ and $ε_B$ start at zero. True mask entries
    mark observed coordinates; if no mask is supplied, observations are identified
    by `~y.isnan()`.

    For direct observations with $H = 𝕀$ and identity residual transforms, $α=1$
    takes the observation, $α=0$ keeps the state, and $α=½$ averages the two.
    A general Kalman gain depends on the prior and observation noise covariances;
    it is not universally one half.

    Note:
        The transpose-observation method expects an `autoregressive` attribute
        that is not initialized by this constructor.
    """

    # PARAMETERS
    H: Tensor
    r"""PARAM: Observation matrix $H$."""
    kernel: Tensor
    r"""Legacy kernel annotation; no `kernel` tensor is initialized."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""
    alpha: Tensor
    r"""PARAM: Learnable or frozen correction strength $α$."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "alpha": float(self.alpha),
            "alpha_learnable": self.alpha.requires_grad,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        n: int = self.input_size
        m: int = self.hidden_size

        # PARAMETERS
        alpha_ = torch.tensor(_Alpha(alpha))
        self.alpha = nn.Parameter(alpha_, requires_grad=alpha_learnable)
        self.epsilonA = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.epsilonB = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.A = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.B = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(n, n)))
        self.H = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        # SEE: https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function."""
        if self.autoregressive:
            return x

        # SEE: https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ji, ...j -> ...i", H, x)

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Apply masked residual transforms and subtract the scaled correction.

        The residual transforms are $𝕀 + ε_A A$ and $𝕀 + ε_B B$, rather than the
        unscaled matrices $A$ and $B$. True mask entries mark observed coordinates.
        """
        mask = ~torch.isnan(y) if mask is None else mask
        z = self.h(x)
        z = torch.where(mask, z - y, self.ZERO)  # → [..., m]
        z = z + self.epsilonA * torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)
        z = self.ht(z)
        z = z + self.epsilonB * torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.alpha * z


class NonLinearKalmanUpdate(nn.Module, VectorStateUpdate):
    r"""Prototype learned gain returning a correction vector.

    For an observed-coordinate diagonal mask $Dₘ$, the forward calculation is

    $$ c = BHᵀDₘADₘ(Hx - y) $$

    Despite the class name, this expression contains no nonlinear activation.
    The method returns $c$ directly; it does not subtract the correction from
    $x$. True mask entries mark observed coordinates, and `~y.isnan()` supplies
    the default mask.

    The declared matrix shapes are compatible with this expression in the square
    case `input_size == hidden_size`.

    References:
        - Kalman filter with outliers and missing observations \
          T. Cipra and R. Romera \
          TEST, 6, 379–395, 1997. \
          <https://doi.org/10.1007/BF02564705>
    """

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: Observation matrix $H$."""
    kernel: Tensor
    r"""Legacy kernel annotation; no `kernel` tensor is initialized."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "autoregressive": self.autoregressive,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        autoregressive: bool = False,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.autoregressive = autoregressive

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.A = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
        nn.init.kaiming_normal_(self.B, nonlinearity="linear")

        self.H = nn.Parameter(torch.empty(hidden_size, input_size))
        nn.init.kaiming_normal_(self.H, nonlinearity="linear")

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Return the correction $BHᵀDₘADₘ(Hx - y)$.

        Here $Dₘ$ is the diagonal mask of observed coordinates. This method returns
        the correction vector directly.
        """
        mask = ~torch.isnan(y) if mask is None else mask
        yhat = torch.einsum("ij, ...j -> ...i", self.H, x)
        r = torch.where(mask, yhat - y, self.ZERO)  # → [..., m]
        z = torch.where(mask, torch.einsum("ij, ...j -> ...i", self.A, r), self.ZERO)
        q = torch.einsum("ji, ...j -> ...i", self.H, z)
        return torch.einsum("ij, ...j -> ...i", self.B, q)


class NonLinearUpdate(nn.Module, VectorStateUpdate):
    r"""Prototype residual update with a nonlinear correction.

    $$ x' = x - αψ\Bigl(BHᵀDₘADₘ(Hx - y)\Bigr) $$

    Here $ψ$ is `nonlinearity` and $Dₘ$ is the diagonal mask of observed
    coordinates. True mask entries mark observed values, and `~y.isnan()`
    supplies the default mask. The learnable or fixed scalar $α$ controls the
    correction strength.
    """

    # PARAMETERS
    H: Tensor
    r"""PARAM: Observation matrix $H$."""
    kernel: Tensor
    r"""Legacy kernel annotation; no `kernel` tensor is initialized."""
    alpha: Tensor
    r"""PARAM: Learnable or frozen correction strength $α$."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "nonlinearity": self.nonlinearity,
            "alpha_value": float(self.alpha),
            "alpha_learnable": self.alpha.requires_grad,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        nonlinearity: nn.Module,
        alpha_value: float = 0.0,
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        m = self.input_size
        n = self.hidden_size
        # Modules
        self.nonlinearity = nonlinearity
        # PARAMETERS
        self.alpha = nn.Parameter(
            torch.tensor(alpha_value),
            requires_grad=alpha_learnable,
        )
        # self.epsilonA = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.epsilonB = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.A = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.B = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(n, n)))
        self.H = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Return the state after the scaled nonlinear correction.

        Args:
            y: Observation tensor, possibly containing NaNs at missing coordinates.
            x: Current state tensor.
            mask: Boolean mask with true entries at observed coordinates.
                If `None`, uses `~y.isnan()`.

        Returns:
            The updated state $x' = x - αψ(BHᵀDₘADₘ(Hx - y))$.
        """
        mask = ~torch.isnan(y) if mask is None else mask
        yhat = torch.einsum("ij, ...j -> ...i", self.H, x)  # (..., m)
        r = torch.where(mask, yhat - y, self.ZERO)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.A, r)
        z = torch.where(mask, z, self.ZERO)  # (..., m)
        z = torch.einsum("ji, ...j -> ...i", self.H, z)  # (..., n)
        z = torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.alpha * self.nonlinearity(z)
