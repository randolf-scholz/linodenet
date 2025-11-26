r"""Implementations of various filter cells.

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
    # Protocols & ABCs
    # Classes
    "NonLinearCell",
    "NonLinearKalmanCell",
    "PseudoKalmanCell",
]

from math import sqrt
from typing import Any, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.filters.base import CellBase
from linodenet.layers import ReverseDense
from linodenet.layers.containers import initialize_from_dict
from linodenet.utils import deep_dict_update


def _set_alpha(alpha: str | float) -> float:
    match alpha:
        case float(value):
            return value
        case "first-value":
            return 0.0
        case "last-value":
            return 1.0
        case "kalman":
            return 0.5
        case str(name):
            raise ValueError(f"Unknown alpha: {name}")
        case _:
            raise TypeError(f"Unknown alpha: {type(alpha)}")


class NonLinearCell(CellBase):
    r"""Non-linear Layers stacked on top of linear core."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
        "num_blocks": 2,
        "block": ReverseDense.HP | {"bias": False},
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        # alpha: str | float = "last-value",
        # alpha_learnable: bool = True,
        **cfg: Any,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        config = deep_dict_update(self.HP, cfg)
        config["block"]["input_size"] = input_size
        config["block"]["output_size"] = input_size

        # CONSTANTS
        n = self.input_size
        m = self.hidden_size

        # MODULES
        blocks: list[nn.Module] = []
        for _ in range(config["num_blocks"]):
            module = initialize_from_dict(config["block"])
            if getattr(module, "bias", None) is not None:
                raise ValueError("Avoid bias term!")
            blocks.append(module)

        self.layers = nn.Sequential(*blocks)

        # PARAMETERS
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.epsilonA = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.epsilonB = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.A = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.B = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(n, n)))
        self.H = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)$.

        .. Signature:: ``[(..., m), (..., n)] -> (..., n)``.
        """
        mask = ~torch.isnan(y)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.H, x)  # (..., m)
        z = torch.where(mask, z - y, self.ZERO)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)  # (..., m)
        z = torch.einsum("ji, ...j -> ...i", self.H, z)  # (..., n)
        z = torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.epsilon * self.layers(z)


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

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
    }
    r"""The HyperparameterDict of this class."""

    def __init__(self, /, **cfg: Any):
        config = deep_dict_update(self.HP, cfg)
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        super().__init__(input_size=input_size, hidden_size=hidden_size)

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
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $BΠAΠ(x - y)$.

        .. Signature:: ``[(..., m), (..., n)] -> (..., n)``.
        """
        mask = ~torch.isnan(y)  # → [..., m]
        yhat = torch.einsum("ij, ...j -> ...i", self.H, x)
        r = torch.where(mask, yhat - y, self.ZERO)  # → [..., m]
        z = torch.where(mask, torch.einsum("ij, ...j -> ...i", self.A, r), self.ZERO)
        q = torch.einsum("ji, ...j -> ...i", self.H, z)
        return torch.einsum("ij, ...j -> ...i", self.B, q)


class PseudoKalmanCell(CellBase):
    r"""A Linear, Autoregressive Filter.

    .. math::  x̂' = x̂ - αP∏ₘᵀP⁻¹Πₘ(x̂ - x)

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.

    One idea: $P = 𝕀 + εA$, where $A$ is symmetric. In this case,
    $𝕀-εA$ is approximately equal to the inverse.

    We define the linearized filter as

    .. math::  x̂' = x̂ - α(𝕀 + εA)∏ₘᵀ(𝕀 - εA)Πₘ(x̂ - x)

    Where $ε$ is initialized as zero.
    """

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "alpha": "last-value",
        "alpha_learnable": False,
        "projection": "Symmetric",
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        # PARAMETERS
        alpha_ = torch.tensor(_set_alpha(alpha))
        self.alpha = nn.Parameter(alpha_, requires_grad=alpha_learnable)
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.weight = nn.Parameter(torch.empty(self.input_size, self.input_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

        # BUFFERS
        with torch.no_grad():
            kernel = self.epsilon * self.weight
            self.register_buffer("kernel", kernel)
            self.register_buffer("ZERO", torch.zeros(1))

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # refresh buffer
        kernel = self.epsilon * self.weight

        # create the mask
        mask = ~torch.isnan(y)  # → [..., m]
        z = torch.where(mask, x - y, self.ZERO)  # → [..., m]
        z = z - torch.einsum("ij, ...j -> ...i", kernel, z)  # → [..., n]
        z = torch.where(mask, z, self.ZERO)
        z = z + torch.einsum("ij, ...j -> ...i", kernel, z)
        return x - self.alpha * z
