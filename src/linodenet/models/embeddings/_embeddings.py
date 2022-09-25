r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
]

from typing import Any, Final

import torch
from torch import Tensor, jit, nn


class ConcatEmbedding(nn.Module):
    r"""Maps $x ⟼ [x,w]$.

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    pad_size:    int
    padding: Tensor
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "hidden_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        assert (
            input_size <= output_size
        ), f"ConcatEmbedding requires {input_size=} ≤ {output_size=}!"
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = output_size - input_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = list(x.shape[:-1]) + [self.padding_size]
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return y[..., : self.input_size]


class ConcatProjection(nn.Module):
    r"""Maps $z = [x,w] ⟼ x$."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "hidden_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        assert (
            input_size >= output_size
        ), f"ConcatEmbedding requires {input_size=} ≥ {output_size=}!"
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = input_size - output_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return x[..., : self.input_size]

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = list(y.shape[:-1]) + [self.padding_size]
        return torch.cat([y, self.padding.expand(shape)], dim=-1)

    # TODO: Add variant with filter in latent space
    # TODO: Add Controls
