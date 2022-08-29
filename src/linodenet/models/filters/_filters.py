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
    # Classes
    "FilterABC",
    "KalmanFilter",
    "KalmanCell",
    "RecurrentCellFilter",
    "SequentialFilterBlock",
    "SequentialFilter",
]
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.util import (
    LookupTable,
    ReverseDense,
    ReZeroCell,
    deep_dict_update,
    deep_keyval_update,
    initialize_from_config,
)

Cell = nn.Module
r"""Type hint for Cells."""

CELLS: Final[LookupTable[Cell]] = {
    "RNNCell": nn.RNNCell,
    "GRUCell": nn.GRUCell,
    "LSTMCell": nn.LSTMCell,
}
r"""Lookup table for cells."""


class FilterABC(nn.Module):
    r"""Base class for all filters.

    All filters should have a signature of the form:

    .. math::
       x' = x + ϕ(y-h(x))

    Where `x` is the current state of the system, `y` is the current measurement, and
    `x'` is the new state of the system. `ϕ` is a function that maps the measurement
    to the state of the system. `h` is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in Filters
    satisfying the idempotence property: if `y=h(x)`, then `x'=x`.
    """

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        """Forward pass of the filter.

        Parameters
        ----------
        x: Tensor
            The current estimation of the state of the system.
        y: Tensor
            The current measurement of the system.

        Returns
        -------
        Tensor:
            The updated state of the system.
        """


class KalmanFilter(FilterABC):
    r"""Classical Kalman Filter.

    .. math::
        x̂ₜ₊₁ &= x̂ₜ + Pₜ Hₜᵀ(Hₜ Pₜ   Hₜᵀ + Rₜ)⁻¹ (yₜ - Hₜ x̂ₜ) \\
        Pₜ₊₁ &= Pₜ - Pₜ Hₜᵀ(Hₜ Pₜ⁻¹ Hₜᵀ + Rₜ)⁻¹ Hₜ Pₜ⁻¹

    In the case of missing data:

    Substitute `yₜ← Sₜ⋅yₜ`, `Hₜ ← Sₜ⋅Hₜ` and `Rₜ ← Sₜ⋅Rₜ⋅Sₜᵀ` where `Sₜ`
    is the `mₜ×m` projection matrix of the missing values. In this case:

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

        where `H = \frac{∂h}{∂x}|_{x̂}`. Note that the EKF is generally not an optimal
        filter.
    """

    # CONSTANTS
    input_size: Final[int]
    """CONST: The input size."""
    hidden_size: Final[int]
    """CONST: The hidden size."""

    # PARAMETERS
    H: Tensor
    """PARAM: The observation matrix."""
    R: Tensor
    """PARAM: The observation noise covariance matrix."""

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
        self.R = nn.Parameter(torch.empty(input_size, hidden_size))
        nn.init.kaiming_normal_(self.H, nonlinearity="linear")
        nn.init.kaiming_normal_(self.R, nonlinearity="linear")

    @jit.export
    def forward(self, y: Tensor, x: Tensor, *, P: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass of the filter.

        Parameters
        ----------
        x: Tensor
        y: Tensor
        P: Optional[Tensor] = None

        Returns
        -------
        Tensor
        """
        P = torch.eye(x.shape[-1]) if P is None else P
        # create the mask
        mask = ~torch.isnan(y)
        H = self.H
        R = self.R
        r = y - torch.einsum("ij, ...j -> ...i", H, x)
        r = torch.where(mask, r, self.ZERO)
        z = torch.linalg.solve(H @ P @ H.t() + R, r)
        z = torch.where(mask, z, self.ZERO)
        return x + torch.einsum("ij, jk, ..k -> ...i", P, H.t(), z)


class KalmanCell(FilterABC):
    r"""A Kalman-Filter inspired non-linear Filter.

    We assume that `y = h(x)` and `y = H⋅x` in the linear case. We adapt  the formula
    provided by the regular Kalman Filter and replace the matrices with learnable
    parameters `A` and `B` and insert an neural network block `ψ`, typically a
    non-linear activation function followed by a linear layer `ψ(z)=Wϕ(z)`.

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

    x̂' &= x̂ - ½ (x̂ - y)    \\
       &= ½(x̂ + y)

    So in this case, the filter precisely always chooses the average between the prediction and the measurement.

    The reason for a another linear transform after ϕ is to stabilize the distribution.
    Also, when `ϕ=𝖱𝖾𝖫𝖴`, it is necessary to allow negative updates.

    Note that in the autoregressive case, i.e. `H=𝕀`, the equation can be simplified
    towards `x̂' ⇝ x̂ + ψ( B ∏ₘᵀ A ∏ₘ (y - Hx̂) )`.

    References
    ----------
    - | Kalman filter with outliers and missing observations
      | T. Cipra, R. Romera
      | https://link.springer.com/article/10.1007/BF02564705
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "hidden_size": int,
        "autoregressive": False,
    }
    """The HyperparameterDict of this class."""

    # CONSTANTS
    autoregressive: Final[bool]
    """CONST: Whether the filter is autoregressive or not."""
    input_size: Final[int]
    """CONST: The input size (=dim x)."""
    hidden_size: Final[int]
    """CONST: The hidden size (=dim y)."""

    # PARAMETERS
    H: Optional[Tensor]
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(self, /, **HP: Any):
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        # CONSTANTS
        self.autoregressive = HP["autoregressive"]
        self.input_size = input_size = HP["input_size"]

        if self.autoregressive:
            hidden_size = HP["input_size"]
        else:
            hidden_size = HP["hidden_size"]

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
        r"""Apply the observation function.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ij, ...j -> ...i", H, x)

    @jit.export
    def ht(self, x: Tensor) -> Tensor:
        r"""Apply the transpose observation function.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        if self.autoregressive:
            return x

        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        H = self.H  # need to assign to local for torchscript....
        assert H is not None, "H must be given in non-autoregressive mode!"
        return torch.einsum("ji, ...j -> ...i", H, x)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: `[...,m], [...,n] ⟶ [...,n]`.

        Parameters
        ----------
        y: Tensor
        x: Tensor

        Returns
        -------
        Tensor
        """
        # create the mask
        mask = ~torch.isnan(y)  # → [..., m]
        r = torch.where(mask, y - self.h(x), self.ZERO)  # → [..., m]
        z = torch.where(mask, torch.einsum("ij, ...j -> ...i", self.A, r), self.ZERO)
        return torch.einsum("ij, ...j -> ...i", self.B, self.ht(z))


class SequentialFilterBlock(FilterABC, nn.ModuleList):
    """Multiple Filters applied sequentially."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "filter": KalmanCell.HP | {"autoregressive": True},
        "layers": [ReverseDense.HP | {"bias": False}, ReZeroCell.HP],
    }
    """The HyperparameterDict of this class."""

    input_size: Final[int]

    def __init__(self, *args: Any, **HP: Any) -> None:
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        self.input_size = input_size = HP["input_size"]
        HP["filter"]["input_size"] = input_size

        layers: list[nn.Module] = []

        for layer in HP["layers"]:
            if "input_size" in layer:
                layer["input_size"] = input_size
            if "output_size" in layer:
                layer["output_size"] = input_size
            module = initialize_from_config(layer)
            layers.append(module)

        layers = list(args) + layers
        self.filter: nn.Module = initialize_from_config(HP["filter"])
        self.layers: Iterable[nn.Module] = nn.Sequential(*layers)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        """Signature: `[...,m], [...,n] ⟶ [...,n]`."""
        z = self.filter(y, x)
        for module in self.layers:
            z = module(z)
        return x + z


class SequentialFilter(FilterABC, nn.ModuleList):
    """Multiple Filters applied sequentially."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "independent": True,
        "copies": 2,
        "input_size": int,
        "module": SequentialFilterBlock.HP,
    }
    """The HyperparameterDict of this class."""

    def __init__(self, **HP: Any) -> None:
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        HP["module"]["input_size"] = HP["input_size"]

        copies: list[nn.Module] = []

        for _ in range(HP["copies"]):
            if isinstance(HP["module"], nn.Module):
                module = HP["module"]
            else:
                module = initialize_from_config(HP["module"])

            if HP["independent"]:
                copies.append(module)
            else:
                copies = [module] * HP["copies"]
                break

        HP["module"] = str(HP["module"])
        nn.ModuleList.__init__(self, copies)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        """Signature: `[...,m], [...,n] ⟶ [...,n]`."""
        for module in self:
            x = module(y, x)
        return x


class RecurrentCellFilter(FilterABC):
    """Any Recurrent Cell allowed."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "concat": True,
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
    """The HyperparameterDict of this class."""

    # CONSTANTS
    concat_mask: Final[bool]
    """CONST: Whether to concatenate the mask to the inputs."""
    input_size: Final[int]
    """CONST: The input size."""
    hidden_size: Final[int]
    """CONST: The hidden size."""
    autoregressive: Final[bool]
    """CONST: Whether the filter is autoregressive or not."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""

    def __init__(self, /, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        # CONSTANTS
        self.concat_mask = HP["concat"]
        self.input_size = input_size * (1 + self.concat_mask)
        self.hidden_size = hidden_size
        self.autoregressive = HP["autoregressive"]

        if self.autoregressive:
            assert (
                hidden_size == input_size
            ), "Autoregressive filter requires x_dim == y_dim"
            self.H = torch.eye(input_size)
        else:
            self.H = nn.Parameter(torch.empty(input_size, hidden_size))
            nn.init.kaiming_normal_(self.H, nonlinearity="linear")

        deep_keyval_update(HP, input_size=self.input_size, hidden_size=self.hidden_size)

        # MODULES
        self.cell = initialize_from_config(HP["Cell"])

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        if self.autoregressive:
            return x
        return torch.einsum("ij, ...j -> ...i", self.H, x)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: `[...,m], [...,n] ⟶ [...,n]`.

        Parameters
        ----------
        y: Tensor
        x: Tensor

        Returns
        -------
        Tensor
        """
        mask = torch.isnan(y)

        # impute missing value in observation with state estimate
        if self.autoregressive:
            y = torch.where(mask, x, y)
        else:
            # TODO: something smarter in non-autoregressive case
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
