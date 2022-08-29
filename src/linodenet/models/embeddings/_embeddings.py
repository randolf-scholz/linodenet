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
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    pad_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(self, input_size: int, hidden_size: int, **HP: Any) -> None:
        super().__init__()
        assert (
            input_size <= hidden_size
        ), f"ConcatEmbedding requires {input_size=} < {hidden_size=}!"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size
        self.padding = nn.Parameter(torch.randn(self.pad_size))

    @jit.export
    def forward(self, X: Tensor) -> Tensor:
        r""".. Signature:: ``(..., d) -> (..., d+e)``.

        Parameters
        ----------
        X: Tensor, shape=(...,DIM)

        Returns
        -------
        Tensor, shape=(...,LAT)
        """
        shape = list(X.shape[:-1]) + [self.pad_size]
        return torch.cat([X, self.padding.expand(shape)], dim=-1)

    @jit.export
    def inverse(self, Z: Tensor) -> Tensor:
        r""".. Signature: ``(..., d+e) -> (..., d)``.

        The reverse of the forward. Satisfies inverse(forward(x)) = x for any input.

        Parameters
        ----------
        Z: Tensor, shape=(...,LEN,LAT)

        Returns
        -------
        Tensor, shape=(...,LEN,DIM)
        """
        return Z[..., : self.input_size]


class ConcatProjection(nn.Module):
    r"""Maps $z = [x,w] ⟼ x$.

    Attributes
    ----------
    input_size:  int
    hidden_size: int
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
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    pad_size: Final[int]
    r"""CONST: The size of the padding."""

    def __init__(self, input_size: int, hidden_size: int, **HP: Any) -> None:
        super().__init__()
        assert input_size <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size

    @jit.export
    def forward(self, Z: Tensor) -> Tensor:
        r""".. Signature: ``(..., d+e) -> (..., d)``.

        Parameters
        ----------
        Z: Tensor, shape=(...,LEN,LAT)

        Returns
        -------
        Tensor, shape=(...,LEN,DIM)
        """
        return Z[..., : self.input_size]

    # TODO: Add variant with filter in latent space
    # TODO: Add Controls
