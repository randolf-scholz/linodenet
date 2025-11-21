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
    "Cell",
    "CellBase",
    # Classes
    "LinearCell",
    "LinearKalmanCell",
    "LinearResidualCell",
    "NonLinearCell",
    "NonLinearKalmanCell",
    "PseudoKalmanCell",
    "ResidualCell",
]

from abc import abstractmethod
from collections.abc import Callable
from math import sqrt
from typing import Any, Final, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from linodenet.activations import get_activation
from linodenet.layers import ReverseDense
from linodenet.torch_generics import initialize_from_dict
from linodenet.utils import deep_dict_update


@runtime_checkable
class Cell(Protocol):
    r"""Protocol for cells."""

    # FIXME: python 3.14 use ReadOnly instead (https://peps.python.org/pep-0767)
    input_size: Final[int]  # type: ignore[misc]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    r"""The size of the hidden state $x$."""

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter $x' = F(y，x)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


class CellBase(nn.Module):
    r"""Base class for filter-cells."""

    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """
        ...


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


class LinearCell(CellBase):
    r"""Linear RNN Cell.

    .. math:: F(y，x) =  Ux + Vy + b

    where $U$ and $V$ are learnable matrices, and $b$ is a learnable bias vector.
    """

    # PARAMETERS
    U: Tensor
    r"""PARAM: the hidden state matrix."""
    V: Tensor
    r"""PARAM: the observable matrix."""
    bias: Optional[Tensor]
    r"""PARAM: the bias vector."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__(input_size, hidden_size)
        n = self.input_size
        m = self.hidden_size
        self.U = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.V = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        self.bias = nn.Parameter(torch.zeros(m)) if bool(bias) else None

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the cell.

        .. math:: F(y，x) =  Ux + Vy + b

        .. Signature:: ``[(..., n), (..., m)] -> (..., m)``.
        """
        z = torch.einsum("...i,ij->...j", x, self.U)
        z = z + torch.einsum("...i,ij->...j", y, self.V)

        if self.bias is not None:
            z = z + self.bias
        return z


class LinearResidualCell(nn.Module):
    r"""Linear RNN Cell that performs a residual update.

    .. math:: x' = x - F⋅(Hy - x)

    Where $F$ is a learnable square matrix, and $H$ is either a learnable matrix or
    a fixed matrix.
    """

    # CONSTANTS
    input_size: Final[int]
    r"""CONST: The size of the observable $y$."""
    hidden_size: Final[int]
    r"""CONST: The size of the hidden state $x$."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""

    # PARAMETERS
    F: Tensor
    r"""PARAM: the hidden state matrix."""
    H: Optional[Tensor]
    r"""PARAM: the observable matrix."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        autoregressive: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.autoregressive = bool(autoregressive)
        m, n = self.hidden_size, self.input_size
        self.F = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.H = (
            nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
            if autoregressive
            else None
        )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r = (
            y - x
            if self.autoregressive
            else torch.einsum("...i,ij->...j", y, self.H) - x
        )

        return x - torch.einsum("...i,ij->...j", r, self.F)


class LinearKalmanCell(nn.Module):
    r"""A Linear Filter.

    .. math::  x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)

    - $A$ and $B$ are chosen such that

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""
    alpha: Tensor
    r"""PARAM/BUFFER: The alpha parameter."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "alpha": "last-value",
        "alpha_learnable": False,
        "autoregressive": False,
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        *,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
        autoregressive: bool = False,
    ) -> None:
        super().__init__()
        n: int = int(input_size)
        m: int = n if hidden_size is None else int(hidden_size)

        # CONSTANTS
        self.input_size = n
        self.hidden_size = m
        self.autoregressive = autoregressive

        # PARAMETERS
        alpha_ = torch.tensor(_set_alpha(alpha))
        self.alpha = nn.Parameter(alpha_, requires_grad=alpha_learnable)
        self.epsilonA = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.epsilonB = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.A = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.B = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(n, n)))
        self.H = (
            None
            if autoregressive
            else nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        )
        # TODO: PARAMETRIZATIONS

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # validation
        if autoregressive and n != m:
            raise ValueError("Autoregressive filter requires input_size == hidden_size")

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x

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
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)$.

        .. Signature:: ``[(..., m), (..., n)] -> (..., n)``.
        """
        mask = ~torch.isnan(y)  # → [..., m]
        z = self.h(x)
        z = torch.where(mask, z - y, self.ZERO)  # → [..., m]
        z = z + self.epsilonA * torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)
        z = self.ht(z)
        z = z + self.epsilonB * torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.alpha * z


class NonLinearKalmanCell(nn.Module):
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

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""

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
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.autoregressive = config["autoregressive"]
        self.input_size = input_size = config["input_size"]

        hidden_size = config["input_size" if self.autoregressive else "hidden_size"]
        self.hidden_size = hidden_size

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.A = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
        nn.init.kaiming_normal_(self.B, nonlinearity="linear")

        if self.autoregressive and input_size != hidden_size:
            raise ValueError("Autoregressive filter requires input_size == hidden_size")

        if self.autoregressive:
            self.H = None
        else:
            self.H = nn.Parameter(torch.empty(hidden_size, input_size))
            nn.init.kaiming_normal_(self.H, nonlinearity="linear")

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x

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
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $BΠAΠ(x - y)$.

        .. Signature:: ``[(..., m), (..., n)] -> (..., n)``.
        """
        mask = ~torch.isnan(y)  # → [..., m]
        r = torch.where(mask, self.h(x) - y, self.ZERO)  # → [..., m]
        z = torch.where(mask, torch.einsum("ij, ...j -> ...i", self.A, r), self.ZERO)
        return torch.einsum("ij, ...j -> ...i", self.B, self.ht(z))


class NonLinearCell(nn.Module):
    r"""Non-linear Layers stacked on top of linear core."""

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""

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
        "num_blocks": 2,
        "block": ReverseDense.HP | {"bias": False},
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        input_size: int,
        # hidden_size: Optional[int] = None,
        # alpha: str | float = "last-value",
        # alpha_learnable: bool = True,
        # autoregressive: bool = False,
        **cfg: Any,
    ) -> None:
        super().__init__()
        config = deep_dict_update(self.HP, cfg)
        hidden_size = (
            input_size if config["hidden_size"] is None else config["hidden_size"]
        )
        autoregressive = config["autoregressive"]
        config["block"]["input_size"] = input_size
        config["block"]["output_size"] = input_size
        if autoregressive and input_size != hidden_size:
            raise ValueError("Autoregressive filter requires input_size == hidden_size")

        # CONSTANTS
        self.input_size = n = input_size
        self.hidden_size = m = hidden_size
        self.autoregressive = config["autoregressive"]

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
        self.H = (
            None
            if autoregressive
            else nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        )
        # TODO: PARAMETRIZATIONS

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x

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
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Return $x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)$.

        .. Signature:: ``[(..., m), (..., n)] -> (..., n)``.
        """
        mask = ~torch.isnan(y)  # (..., m)
        z = self.h(x)  # (..., m)
        z = torch.where(mask, z - y, self.ZERO)  # (..., m)
        z = torch.einsum("ij, ...j -> ...i", self.A, z)
        z = torch.where(mask, z, self.ZERO)  # (..., m)
        z = self.ht(z)  # (..., n)
        z = torch.einsum("ij, ...j -> ...i", self.B, z)
        return x - self.epsilon * self.layers(z)


class ResidualCell(nn.Module):
    r"""Non-Linear RNN Cell that performs a residual update.

    .. math:: x' = x - F⋅φ(Hy - x)

    Where $F$ is a learnable square matrix, and $H$ is either a learnable matrix or
    a fixed matrix and $φ$ is a non-linear activation function, applied element-wise.
    """

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        autoregressive: bool = False,
        activation: Callable | str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.autoregressive = bool(autoregressive)
        m, n = self.hidden_size, self.input_size

        self.activation = get_activation(activation)
        self.F = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.H = (
            nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
            if autoregressive
            else None
        )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r = (
            y - x
            if self.autoregressive
            else torch.einsum("...i,ij->...j", y, self.H) - x
        )

        return x - torch.einsum("...i,ij->...j", r, self.F)


class PseudoKalmanCell(nn.Module):
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

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

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
        "alpha": "last-value",
        "alpha_learnable": False,
        "projection": "Symmetric",
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        input_size: int,
        *,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
        **cfg: Any,
    ) -> None:
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.input_size = input_size
        self.hidden_size = config["hidden_size"]

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
