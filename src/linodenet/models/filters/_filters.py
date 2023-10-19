r"""Different Filter models to be used in conjunction with LinodeNet.

A Filter takes two positional inputs:
    - An input tensor x: the current estimation of the state of the system
    - An input tensor y: the current measurement of the system
    - An optional input tensor mask: a mask to be applied to the input tensor
"""

__all__ = [
    # Constants
    "CELLS",
    # Types
    "Cell",
    # Base Classes
    "Filter",
    "FilterABC",
    # Classes
    "KalmanCell",
    "KalmanFilter",
    "LinearFilter",
    "NonLinearFilter",
    "PseudoKalmanFilter",
    "RecurrentCellFilter",
    "SequentialFilter",
    "SequentialFilterBlock",
]
from abc import abstractmethod
from collections.abc import Iterable
from math import sqrt
from typing import Any, Final, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn
from typing_extensions import Self

from linodenet.utils import (
    ReverseDense,
    ReZeroCell,
    assert_issubclass,
    deep_dict_update,
    deep_keyval_update,
    initialize_from_config,
)


@runtime_checkable
class Cell(Protocol):
    """Protocol for all cells."""

    def __call__(self, y: Tensor, x: Tensor) -> Tensor:
        """Forward pass of the cell."""
        ...


CELLS: dict[str, type[Cell]] = {
    "RNNCell": nn.RNNCell,
    "GRUCell": nn.GRUCell,
    "LSTMCell": nn.LSTMCell,
}
r"""Lookup table for cells."""


@runtime_checkable
class Filter(Protocol):
    """Protocol for Filters components."""

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@assert_issubclass(Filter)
class FilterABC(nn.Module):
    r"""Base class for all filters.

    All filters should have a signature of the form:

    .. math::  x' = x + ϕ(y-h(x))

    Where $x$ is the current state of the system, $y$ is the current measurement, and
    $x'$ is the new state of the system. $ϕ$ is a function that maps the measurement
    to the state of the system. $h$ is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in Filters
    satisfying the idempotence property: if $y=h(x)$, then $x'=x$.
    """

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """


@assert_issubclass(Filter)
class PseudoKalmanFilter(nn.Module):
    r"""A Linear, Autoregressive Filter.

    .. math::  x̂' = x̂ - αP∏ₘᵀP^{-1}Πₘ(x̂ - x)

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

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "alpha": "last-value",
        "alpha_learnable": False,
        "projection": "Symmetric",
    }
    r"""The HyperparameterDict of this class."""

    # CONSTANTS
    input_size: Final[int]
    r"""CONST: The input size (=dim x)."""
    hidden_size: Final[int]
    r"""CONST: The hidden size (=dim y)."""

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(
        self,
        input_size: int,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
        **cfg: Any,
    ):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.input_size = input_size
        self.hidden_size = config["hidden_size"]

        # PARAMETERS
        match alpha:
            case "first-value":
                alpha = 0.0
            case "last-value":
                alpha = 1.0
            case "kalman":
                alpha = 0.5
            case str():
                raise ValueError(f"Unknown alpha: {alpha}")

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=alpha_learnable)
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


@assert_issubclass(Filter)
class LinearFilter(nn.Module):
    r"""A Linear Filter.

    .. math::  x' = x - αBHᵀ∏ₘᵀAΠₘ(Hx - y)

    - $A$ and $B$ are chosen such that

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.

    TODO: Add parametrization options.
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "alpha": "last-value",
        "alpha_learnable": False,
        "autoregressive": False,
    }
    r"""The HyperparameterDict of this class."""

    # CONSTANTS
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
    input_size: Final[int]
    r"""CONST: The input size (=dim x)."""
    output_size: Final[int]
    r"""CONST: The hidden size (=dim y)."""
    alpha_learnable: Final[bool]
    r"""CONST: Whether alpha is a learnable parameter."""

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

    def __init__(
        self,
        input_size: int,
        # hidden_size: Optional[int] = None,
        # alpha: str | float = "last-value",
        # alpha_learnable: bool = True,
        # autoregressive: bool = False,
        **cfg: Any,
    ):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)
        hidden_size = config.get("hidden_size", input_size)
        alpha = config["alpha"]
        self.alpha_learnable = config["alpha_learnable"]
        autoregressive = config["autoregressive"]
        assert not autoregressive or input_size == hidden_size

        # CONSTANTS
        self.input_size = n = input_size
        self.output_size = m = hidden_size
        self.autoregressive = config["autoregressive"]

        # PARAMETERS
        match alpha:
            case "first-value":
                alpha = 0.0
            case "last-value":
                alpha = 1.0
            case "kalman":
                alpha = 0.5
            case str():
                raise ValueError(f"Unknown alpha: {alpha}")

        # PARAMETERS
        if self.alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
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

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    @jit.export
    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function."""
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
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


@assert_issubclass(Filter)
class NonLinearFilter(nn.Module):
    r"""Non-linear Layers stacked on top of linear core."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
        "num_blocks": 2,
        "block": ReverseDense.HP | {"bias": False},
    }
    r"""The HyperparameterDict of this class."""

    # CONSTANTS
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
    input_size: Final[int]
    r"""CONST: The input size (=dim x)."""
    output_size: Final[int]
    r"""CONST: The hidden size (=dim y)."""

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(
        self,
        input_size: int,
        # hidden_size: Optional[int] = None,
        # alpha: str | float = "last-value",
        # alpha_learnable: bool = True,
        # autoregressive: bool = False,
        **cfg: Any,
    ):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)
        hidden_size = (
            input_size if config["hidden_size"] is None else config["hidden_size"]
        )
        autoregressive = config["autoregressive"]
        config["block"]["input_size"] = input_size
        config["block"]["output_size"] = input_size
        assert not autoregressive or input_size == hidden_size

        # CONSTANTS
        self.input_size = n = input_size
        self.output_size = m = hidden_size
        self.autoregressive = config["autoregressive"]

        # MODULES
        blocks: list[nn.Module] = []
        for _ in range(config["num_blocks"]):
            module = initialize_from_config(config["block"])
            if hasattr(module, "bias"):
                assert module.bias is None, "Avoid bias term!"
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

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    @jit.export
    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function."""
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
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


@assert_issubclass(Filter)
class KalmanFilter(nn.Module):
    r"""Classical Kalman Filter.

    .. math::
        x̂ₜ₊₁ &= x̂ₜ + Pₜ Hₜᵀ(Hₜ Pₜ   Hₜᵀ + Rₜ)⁻¹ (yₜ - Hₜ x̂ₜ) \\
        Pₜ₊₁ &= Pₜ - Pₜ Hₜᵀ(Hₜ Pₜ⁻¹ Hₜᵀ + Rₜ)⁻¹ Hₜ Pₜ⁻¹

    In the case of missing data:

    Substitute $yₜ← Sₜ⋅yₜ$, $Hₜ ← Sₜ⋅Hₜ$ and $Rₜ ← Sₜ⋅Rₜ⋅Sₜᵀ$ where $Sₜ$
    is the $mₜ×m$ projection matrix of the missing values. In this case:

    .. math::
        x̂' &= x̂ + P⋅Hᵀ⋅Sᵀ(SHPHᵀSᵀ + SRSᵀ)⁻¹ (Sy - SHx̂) \\
           &= x̂ + P⋅Hᵀ⋅Sᵀ(S (HPHᵀ + R) Sᵀ)⁻¹ S(y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅(S⁺S)ᵀ (HPHᵀ + R)⁻¹ (S⁺S) (y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅∏ₘᵀ (HPHᵀ + R)⁻¹ ∏ₘ (y - Hx̂) \\
        P' &= P - P⋅Hᵀ⋅Sᵀ(S H P⁻¹ Hᵀ Sᵀ + SRSᵀ)⁻¹ SH P⁻¹ \\
           &= P - P⋅Hᵀ⋅(S⁺S)ᵀ (H P⁻¹ Hᵀ + R)⁻¹ (S⁺S) H P⁻¹ \\
           &= P - P⋅Hᵀ⋅∏ₘᵀ (H P⁻¹ Hᵀ + R)⁻¹ ∏ₘ H P⁻¹


    .. note::
        The Kalman filter is a linear filter. The non-linear version is also possible,
        the so called Extended Kalman-Filter. Here, the non-linearity is linearized at
        the time of update.

        ..math ::
            x̂' &= x̂ + P⋅Hᵀ(HPHᵀ + R)⁻¹ (y - h(x̂)) \\
            P' &= P -  P⋅Hᵀ(HPHᵀ + R)⁻¹ H P

        where $H = \frac{∂h}{∂x}|_{x̂}$. Note that the EKF is generally not an optimal
        filter.
    """

    # CONSTANTS
    input_size: Final[int]
    r"""CONST: The input size."""
    hidden_size: Final[int]
    r"""CONST: The hidden size."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: The observation matrix."""
    R: Tensor
    r"""PARAM: The observation noise covariance matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(self, /, input_size: int, hidden_size: int):
        super().__init__()

        # CONSTANTS
        self.input_size = input_size
        self.hidden_size = hidden_size

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.H = nn.Parameter(torch.empty(input_size, hidden_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.kaiming_normal_(self.H, nonlinearity="linear")
        nn.init.kaiming_normal_(self.R, nonlinearity="linear")

    @jit.export
    def forward(self, y: Tensor, x: Tensor, *, P: Optional[Tensor] = None) -> Tensor:
        r"""Return $x' = x + P⋅Hᵀ∏ₘᵀ(HPHᵀ + R)⁻¹ ∏ₘ (y - Hx)$."""
        P = torch.eye(x.shape[-1]) if P is None else P
        # create the mask
        mask = ~torch.isnan(y)
        H = self.H
        R = self.R
        r = torch.einsum("ij, ...j -> ...i", H, x) - y
        r = torch.where(mask, r, self.ZERO)
        z = torch.linalg.solve(H @ P @ H.t() + R, r)
        z = torch.where(mask, z, self.ZERO)
        return x - torch.einsum("ij, jk, ..k -> ...i", P, H.t(), z)


@assert_issubclass(Filter)
class KalmanCell(nn.Module):
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
           &= x̂ - ½ P∏ₘᵀP^{-1}Πₘ(x̂ - y)      \\

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

    References
    ----------
    - | Kalman filter with outliers and missing observations
      | T. Cipra, R. Romera
      | https://link.springer.com/article/10.1007/BF02564705
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
    }
    r"""The HyperparameterDict of this class."""

    # CONSTANTS
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
    input_size: Final[int]
    r"""CONST: The input size (=dim x)."""
    hidden_size: Final[int]
    r"""CONST: The hidden size (=dim y)."""

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(self, /, **cfg: Any):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.autoregressive = config["autoregressive"]
        self.input_size = input_size = config["input_size"]

        if self.autoregressive:
            hidden_size = config["input_size"]
        else:
            hidden_size = config["hidden_size"]

        self.hidden_size = hidden_size

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.A = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
        nn.init.kaiming_normal_(self.B, nonlinearity="linear")

        if self.autoregressive:
            assert (
                hidden_size == input_size
            ), "Autoregressive filter requires x_dim == y_dim"
            self.H = None
        else:
            self.H = nn.Parameter(torch.empty(hidden_size, input_size))
            nn.init.kaiming_normal_(self.H, nonlinearity="linear")

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    @jit.export
    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function."""
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
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


# class SequentialFilterBlock(FilterABC, nn.ModuleList):
#     r"""Multiple Filters applied sequentially."""
#
#     HP = {
#         "__name__": __qualname__,  # type: ignore[name-defined]
#         "__module__": __module__,  # type: ignore[name-defined]
#         "input_size": None,
#         "filter": KalmanCell.HP | {"autoregressive": True},
#         "layers": [ReverseDense.HP | {"bias": False}, ReZeroCell.HP],
#     }
#     r"""The HyperparameterDict of this class."""
#
#     input_size: Final[int]
#
#     def __init__(self, *args: Any, **HP: Any) -> None:
#         super().__init__()
#         self.CFG = HP = deep_dict_update(self.HP, HP)
#
#         self.input_size = input_size = HP["input_size"]
#         HP["filter"]["input_size"] = input_size
#
#         layers: list[nn.Module] = []
#
#         for layer in HP["layers"]:
#             if "input_size" in layer:
#                 layer["input_size"] = input_size
#             if "output_size" in layer:
#                 layer["output_size"] = input_size
#             module = initialize_from_config(layer)
#             layers.append(module)
#
#         layers = list(args) + layers
#         self.filter: nn.Module = initialize_from_config(HP["filter"])
#         self.layers: Iterable[nn.Module] = nn.Sequential(*layers)
#
#     @jit.export
#     def forward(self, y: Tensor, x: Tensor) -> Tensor:
#         r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
#         z = self.filter(y, x)
#         for module in self.layers:
#             z = module(z)
#         return x + z


class SequentialFilterBlock(FilterABC):
    r"""Non-linear Layers stacked on top of linear core."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "autoregressive": False,
        "filter": KalmanCell.HP,
        "layers": [ReverseDense.HP | {"bias": False}, ReZeroCell.HP],
    }
    r"""The HyperparameterDict of this class."""

    input_size: Final[int]

    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = None, **cfg: Any
    ) -> None:
        super().__init__()
        config = deep_dict_update(self.HP, cfg)
        config["filter"]["autoregressive"] = config["autoregressive"]

        self.input_size = input_size = config["input_size"]
        config["filter"]["input_size"] = input_size

        self.nonlinear: nn.Module = initialize_from_config(config["filter"])

        # self.add_module("nonlinear", nonlinear)

        layers: list[nn.Module] = [] if modules is None else list(modules)
        for layer in config["layers"]:
            if "input_size" in layer:
                layer["input_size"] = input_size
            if "output_size" in layer:
                layer["output_size"] = input_size
            module = initialize_from_config(layer)
            layers.append(module)

        # super().__init__(layers)
        self.layers = nn.Sequential(*layers)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        z = self.nonlinear(y, x)
        for module in self.layers:
            z = module(z)
        return x - z


# @assert_issubclass(Filter)
class SequentialFilter(nn.Sequential):  # FIXME: use ModuleList instead?
    r"""Multiple Filters applied sequentially."""

    input_size: Final[int]
    """The input size of the filter."""
    output_size: Final[int]
    """The output size of the filter."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
        "layers": [LinearFilter.HP, NonLinearFilter.HP, NonLinearFilter.HP],
    }
    r"""The HyperparameterDict of this class."""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        r"""Initialize from hyperparameters."""
        config = deep_dict_update(cls.HP, cfg)
        layers: list[nn.Module] = []

        for layer in config["layers"]:
            if isinstance(layer, nn.Module):
                module = layer
            else:
                layer |= {
                    "input_size": config["input_size"],
                    "hidden_size": config["hidden_size"],
                    "autoregressive": config["autoregressive"],
                }
                module = initialize_from_config(layer)
            layers.append(module)

        return cls(*layers)

    def __new__(cls, *modules: nn.Module, **cfg: Any) -> Self:
        r"""Initialize from hyperparameters."""
        layers: list[nn.Module] = [] if modules is None else list(modules)
        assert len(layers) ^ len(cfg), "Provide either blocks, or hyperparameters!"
        return cls.from_config(cfg) if cfg else super().__new__(cls)

    def __init__(self, *modules: nn.Module, **cfg: Any) -> None:
        if not modules and cfg:
            return  # must have been initialized from config

        self.input_size = modules[0].input_size if modules else None
        self.output_size = modules[-1].output_size if modules else None

        super().__init__(*modules)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for module in self:
            x = module(y, x)
        return x


class RecurrentCellFilter(FilterABC):
    r"""Any Recurrent Cell allowed."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "concat_mask": True,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": True,
        "Cell": {
            "__name__": "GRUCell",
            "__module__": "torch.nn",
            "input_size": None,
            "hidden_size": None,
            "bias": True,
            "device": None,
            "dtype": None,
        },
    }
    r"""The HyperparameterDict of this class."""

    # CONSTANTS
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the inputs."""
    input_size: Final[int]
    r"""CONST: The input size."""
    hidden_size: Final[int]
    r"""CONST: The hidden size."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""

    def __init__(self, /, input_size: int, hidden_size: int, **cfg: Any):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.concat_mask = config["concat_mask"]
        self.input_size = input_size * (1 + self.concat_mask)
        self.hidden_size = hidden_size
        self.autoregressive = config["autoregressive"]

        if self.autoregressive:
            assert (
                hidden_size == input_size
            ), "Autoregressive filter requires x_dim == y_dim"
            self.H = torch.eye(input_size)
        else:
            self.H = nn.Parameter(torch.empty(input_size, hidden_size))
            nn.init.kaiming_normal_(self.H, nonlinearity="linear")

        deep_keyval_update(
            config, input_size=self.input_size, hidden_size=self.hidden_size
        )

        # MODULES
        self.cell = initialize_from_config(config["Cell"])

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x
        return torch.einsum("ij, ...j -> ...i", self.H, x)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        mask = torch.isnan(y)

        # impute missing value in observation with state estimate
        if self.autoregressive:
            y = torch.where(mask, x, y)
        else:
            y = torch.where(mask, self.h(x), y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Flatten for RNN-Cell
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        # Apply filter
        result = self.cell(y, x)

        # De-Flatten return value
        return result.view(mask.shape)
