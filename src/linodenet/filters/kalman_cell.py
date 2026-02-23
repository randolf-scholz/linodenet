r"""Implementations of Kalman-inspired RNN Cells.

Cells are building blocks for RNNs; as per the Cell Protocol,
a cell essentially is a function $(y, x) -> x'$.

KalmanCell
----------
The classical Kalman Filter state update has a few nice properties

.. math::
       x' &= x - PH'(HPH' + R)⁻¹(Hx - y)
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
"""

__all__ = [
    "KalmanCell",
    "PseudoKalmanCell",
    "NonLinearKalmanCell",
    "NonLinearCell",
]

from enum import Enum
from math import sqrt
from typing import Optional, SupportsFloat

import torch
from torch import Tensor, jit, nn

from linodenet.filters.base import CellBase
from signatures import signature


class _Alpha(float, Enum):
    FIRST_VALUE = 0.0
    AVERAGE = 0.5
    LAST_VALUE = 1.0

    @classmethod
    def new(cls, arg: str | SupportsFloat) -> float:
        if isinstance(arg, str):
            return cls[arg.replace("-", "_").upper()]
        return float(arg)


class KalmanCell(CellBase):
    r"""Implements a Kalman Filter Cell.

    .. math:: F(y，x) =  x - ΣHᵀ(HΣHᵀ + R)⁻¹(Hx - y)

    .. math::
        Σ &= 𝕀 + ε⋅AAᵀ \\
        R &= 𝕀 + ε⋅BBᵀ
    """


class PseudoKalmanCell(CellBase):
    r"""Implements a Kalman-inspired Filter Cell.

    Contrary to the KalmanCell, this cell does not learn a single covariance
    but two separate matrices $A$ and $B$.

    .. math::  x' = x - αBHᵀΠₘᵀAΠₘ(Hx - y)

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.
    """

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""
    alpha: Tensor
    r"""PARAM/BUFFER: The alpha parameter."""

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

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        # SEE: https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    @jit.export
    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function."""
        if self.autoregressive:
            return x

        # SEE: https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ji, ...j -> ...i", H, x)

    @jit.export
    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)$."""
        mask = ~torch.isnan(y)  # → [..., m]
        z = self.h(x)
        z = torch.where(mask, z - y, self.ZERO)  # → [..., m]
        z = z + self.epsilonA * torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)
        z = self.ht(z)
        z = z + self.epsilonB * torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.alpha * z


class NonLinearKalmanCell(CellBase):
    r"""A Kalman-Filter inspired non-linear Filter.

    We assume that $y = h(x)$ and $y = H⋅x$ in the linear case. We adapt  the formula
    provided by the regular Kalman Filter and replace the matrices with learnable
    parameters $A$ and $B$ and insert an neural network block $ψ$, typically a
    non-linear activation function followed by a linear layer $ψ(z)=Wϕ(z)$.

    .. math::
        x̂' &= x̂ + P⋅Hᵀ ∏ₘᵀ (HPHᵀ + R)⁻¹ ∏ₘ (y - Hx̂)    \\
           &⇝ x̂ + B⋅Hᵀ ∏ₘᵀA∏ₘ (y - Hx̂)                 \\
           &⇝ x̂ + ψ(B Hᵀ ∏ₘᵀA ∏ₘ (y - Hx̂))

    Here $yₜ$ is the observation vector. and $x̂$ is the state vector.

    .. math::
        x̂' &= x̂ - P⋅Hᵀ ∏ₘᵀ (HPHᵀ + R)⁻¹ ∏ₘ (Hx̂ - y)    \\
           &⇝ x̂ - B⋅Hᵀ ∏ₘᵀA∏ₘ (Hx̂ - y)                 \\
           &⇝ x̂ - ψ(B Hᵀ ∏ₘᵀA ∏ₘ (Hx̂ - y))

    Note that in the autoregressive case, $H=𝕀$ and $P=R$. Thus

    .. math::
        x̂' &= x̂ - P∏ₘᵀ(2P)⁻¹Πₘ(x̂ - x)        \\
           &= x̂ - ½ P∏ₘᵀP⁻¹Πₘ(x̂ - y)      \\

    We consider a few cases:

    .. math::  x̂' = x̂ - α(x̂ - x)

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.

    So in this case, the filter precisely always chooses the average between the prediction and the measurement.

    The reason for a another linear transform after $ϕ$ is to stabilize the distribution.
    Also, when $ϕ=𝖱𝖾𝖫𝖴$, it is necessary to allow negative updates.

    Note that in the autoregressive case, i.e. $H=𝕀$, the equation can be simplified
    towards $x̂' ⇝ x̂ + ψ( B ∏ₘᵀ A ∏ₘ (y - Hx̂) )$.

    References:
        Kalman filter with outliers and missing observations
        T. Cipra, R. Romera
        https://link.springer.com/article/10.1007/BF02564705
    """

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

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
        self.autoregressive = bool(autoregressive)

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.A = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
        nn.init.kaiming_normal_(self.B, nonlinearity="linear")

        self.H = nn.Parameter(torch.empty(hidden_size, input_size))
        nn.init.kaiming_normal_(self.H, nonlinearity="linear")

    @jit.export
    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $BΠAΠ(x - y)$."""
        mask = ~torch.isnan(y)  # → [..., m]
        yhat = torch.einsum("ij, ...j -> ...i", self.H, x)
        r = torch.where(mask, yhat - y, self.ZERO)  # → [..., m]
        z = torch.where(mask, torch.einsum("ij, ...j -> ...i", self.A, r), self.ZERO)
        q = torch.einsum("ji, ...j -> ...i", self.H, z)
        return torch.einsum("ij, ...j -> ...i", self.B, q)


class NonLinearCell(CellBase):
    r"""Non-linear Layers stacked on top of linear core."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""
    alpha: Tensor
    r"""PARAM: The alpha parameter."""

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

    @jit.export
    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return the updated state tensor.

        Args:
            y: The observation Tensor. May contain NaNs for missing values.
            x: The state tensor

        Returns:
            The updated state tensor $x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)$.
        """
        mask = ~torch.isnan(y)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.H, x)  # (..., m)
        z = torch.where(mask, z - y, self.ZERO)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)  # (..., m)
        z = torch.einsum("ji, ...j -> ...i", self.H, z)  # (..., n)
        z = torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.alpha * self.nonlinearity(z)
