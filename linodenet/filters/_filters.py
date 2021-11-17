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
    "KalmanBlockCell",
    "KalmanFilter",
    "KalmanCell",
    "RecurrentCellFilter",
]

import logging
from abc import abstractmethod
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.util import (
    ACTIVATIONS,
    LookupTable,
    autojit,
    deep_dict_update,
    deep_keyval_update,
    initialize_from,
)

__logger__ = logging.getLogger(__name__)

Cell = nn.Module
r"""Type hint for Cells."""

CELLS: Final[LookupTable[Cell]] = {
    "RNNCell": nn.RNNCell,
    "GRUCell": nn.GRUCell,
    "LSTMCell": nn.LSTMCell,
}
r"""Lookup table for cells."""


class FilterABC(nn.Module):
    r"""Base class for all filters."""

    def __init__(self) -> None:
        super().__init__()

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


@autojit
class KalmanFilter(FilterABC):
    r"""Classical Kalman Filter."""

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
    def forward(self, /, y: Tensor, x: Tensor, *, P: Optional[Tensor] = None) -> Tensor:
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
        P = torch.eye(len(x)) if P is None else P
        # create the mask
        mask = ~torch.isnan(y)
        H = self.H
        R = self.R
        r = y - torch.einsum("ij, ...j -> ...i", H, x)
        r = torch.where(mask, r, self.ZERO)
        z = torch.linalg.solve(H @ P @ H.t() + R, r)
        z = torch.where(mask, z, self.ZERO)
        return x + torch.einsum("ij, jk, ..k -> ...i", P, H.t(), z)


@autojit
class KalmanCell(FilterABC):
    r"""A Kalman-Filter inspired non-linear Filter."""

    HP: dict = {
        "activation": "Identity",
        "autoregressive": False,
    }

    # CONSTANTS
    autoregressive: Final[bool]
    """CONST: Whether the filter is autoregressive or not."""
    input_size: Final[int]
    """CONST: The input size."""
    hidden_size: Final[int]
    """CONST: The hidden size."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(self, /, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        # CONSTANTS
        self.autoregressive = HP["autoregressive"]
        self.input_size = input_size
        self.hidden_size = hidden_size

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.kernel = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_normal_(self.kernel, nonlinearity="linear")

        # MODULES
        self.activation = ACTIVATIONS[HP["activation"]]()

        if self.autoregressive:
            assert (
                hidden_size == input_size
            ), "Autoregressive filter requires x_dim == y_dim"
            self.H = torch.eye(input_size)
        else:
            self.H = nn.Parameter(torch.empty(input_size, hidden_size))
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
        # create the mask
        mask = ~torch.isnan(y)
        r = torch.where(mask, y - self.h(x), self.ZERO)
        z = torch.where(
            mask, torch.einsum("ij, ...j -> ...i", self.kernel, r), self.ZERO
        )
        return x + self.activation(self.h(z))


class KalmanBlockCell(FilterABC):
    r"""Multiple KalmanCells."""

    HP: dict = {
        "nblocks": 5,
        "activation": "Identity",
        "autoregressive": True,
        "Cell": {
            "__name__": "GRUCell",
            "input_size": None,
            "hidden_size": None,
            "autoregressive": True,
            "activation": "Identity",
        },
    }

    # CONSTANTS
    autoregressive: Final[bool]
    """CONST: Whether the filter is autoregressive or not."""
    nblocks: Final[int]
    """CONST: The number of blocks."""
    input_size: Final[int]
    """CONST: The input size."""
    hidden_size: Final[int]
    """CONST: The hidden size."""

    def __init__(self, /, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        # CONSTANTS
        self.nblocks = HP["nblocks"]
        self.autoregressive = HP["autoregressive"]
        self.input_size = input_size
        self.hidden_size = hidden_size

        # MODULES
        self.activation = ACTIVATIONS[HP["activation"]]()
        self.blocks = nn.Sequential(
            *(
                KalmanCell(input_size=input_size, hidden_size=hidden_size, **HP)
                for _ in range(self.nblocks)
            )
        )

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: `[...,m], [...,n] ⟶ [...,n]`.

        Parameters
        ----------
        x: Tensor
        y: Tensor

        Returns
        -------
        xhat: Tensor
        """
        for block in self.blocks:
            x = block(y, x)
        return x


@autojit
class RecurrentCellFilter(FilterABC):
    """Any Recurrent Cell allowed."""

    HP: dict = {
        "concat": True,
        "Cell": {
            "__name__": "GRUCell",
            "input_size": None,
            "hidden_size": None,
            "bias": True,
            "device": None,
            "dtype": None,
        },
    }

    # CONSTANTS
    concat_mask: Final[bool]
    """CONST: Whether to concatenate the mask to the inputs."""
    input_size: Final[int]
    """CONST: The input size."""
    hidden_size: Final[int]
    """CONST: The hidden size."""

    def __init__(self, /, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        # CONSTANTS
        self.concat_mask = HP["concat"]
        self.input_size = input_size * (1 + self.concat_mask)
        self.hidden_size = hidden_size

        deep_keyval_update(
            self.HP, input_size=self.input_size, hidden_size=self.hidden_size
        )

        # MODULES
        self.cell = initialize_from(CELLS, HP["Cell"])

    @jit.export
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Perform the forward pass.

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
        y = torch.where(mask, x, y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Flatten for GRU-Cell
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        # Apply filter
        return self.cell(y, x).view(x.shape)
