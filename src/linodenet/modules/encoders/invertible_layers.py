r"""Implementation of invertible layers.

Layers:

- Affine: $y = Ax+b$ and $x = A⁻¹(y-b)$
    - A diagonal
    - A triangular
    - A tridiagonal
- Element-wise:
    - Monotonic
- Shears (coupling flows): $y_A = f(x_A, x_B)$ and $y_B = x_B$
    - Example: $y_A = x_A + e^{x_B}$ and $y_B = x_B$
- Residual: $y = x + F(x)$
    - Contractive: $y = F(x)$ with $‖F‖<1$
    - Low Rank Perturbation: $y = x + ABx$
- Continuous Time Flows: $ẋ=f(t, x)$
"""

__all__ = [
    # Protocols & ABCs
    "InvertibleModule",
    "InvertibleModuleABC",
    # Classes
    "iResNetBlock",
    "iSequential",
]

import warnings
from abc import abstractmethod
from typing import Any, Final, Optional, Protocol, Self, runtime_checkable

import torch
from torch import Tensor, jit, nn


@runtime_checkable
class InvertibleModule[U, V](Protocol):
    r"""Protocol for invertible layers."""

    # NOTE: Theoretically, this must be a subclass of nn.Module.
    # but typing system currently does not support this.

    @abstractmethod
    def inverse(self) -> "InvertibleModule[V, U]":
        r"""Return the inverse of the layer."""
        ...

    @abstractmethod
    def encode(self, u: U, /) -> V:
        r"""Encode the data."""
        ...

    @abstractmethod
    def decode(self, v: V, /) -> U:
        r"""Decode the data."""
        ...

    def transform(self, u: U, /) -> V:
        r"""Alias for encode."""
        return self.encode(u)

    def inverse_transform(self, v: V, /) -> U:
        r"""Alias for decode."""
        return self.decode(v)


class InvertibleModuleABC[U, V](nn.Module, InvertibleModule[U, V]):
    r"""Abstract base class for invertible layers."""

    @abstractmethod
    def encode(self, u: U, /) -> V:
        r"""Encode the data."""
        ...

    @abstractmethod
    def decode(self, v: V, /) -> U:
        r"""Decode the data."""
        ...


class iResNetBlock(nn.Module):
    r"""A single block of an iResNet.

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | http://proceedings.mlr.press/v97/behrmann19a.html
    """

    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""
    atol: Final[float]
    r"""CONST: Absolute tolerance for fixed point iteration"""
    rtol: Final[float]
    r"""CONST: Relative tolerance for fixed point iteration"""
    is_inverse: Final[bool]
    r"""CONST: Whether to use the inverse or the forward map"""
    converged: Tensor
    r"""BUFFER: Boolean tensor indicating convergence"""

    def __init__(
        self,
        layer: nn.Module,
        *,
        maxiter: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        inverse: Optional[Self] = None,
    ) -> None:
        super().__init__()
        self.block = layer
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.register_buffer("converged", torch.tensor(False))

        self.is_inverse = inverse is not None
        if inverse is None:
            cls = type(self)
            self.inverse = cls(
                layer,
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
                inverse=self,
            )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n) -> (..., n)``."""
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Compute the forward map with residual connection."""
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        if self.is_inverse:
            return self._encode(y)
        return self._decode(y)

    @jit.export
    def _encode(self, x: Tensor) -> Tensor:
        return x + self.block(x)

    @jit.export
    def _decode(self, y: Tensor) -> Tensor:
        x = y.clone().detach()
        # m = torch.isnan(y)
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.block(x)
            with torch.no_grad():
                residual = torch.sqrt(torch.nansum((x - x_prev).pow(2)))
                tol = self.atol + self.rtol * torch.sqrt(torch.nansum(x_prev.pow(2)))
                self.converged = residual < tol
                if self.converged:
                    break
        if not self.converged:
            warnings.warn(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual.item()} > {self.atol}.",
                stacklevel=2,
            )
        return x


class iSequential(nn.Module):
    r"""Invertible Sequential model."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    is_inverse: Final[bool]
    r"""CONST: Whether to use the inverse or the forward map"""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "maxiter": 10,
        "input_size": None,
        "dropout": None,
        "bias": True,
        "nblocks": 5,
        "rezero": False,
        "iResNetBlock": {
            "input_size": None,
            "activation": "ReLU",
            "activation_config": {"inplace": False},
            "bias": True,
            "hidden_size": None,
            "maxiter": 100,
        },
    }
    r"""The hyperparameter dictionary"""

    @classmethod
    def from_config(cls, **cfg: Any) -> Self:
        r"""Initialize from hyperparameters."""
        raise NotImplementedError

    def __new__(
        cls, *modules: nn.Module, inverse: Optional[Self] = None, **cfg: Any
    ) -> Self:
        r"""Initialize from hyperparameters."""
        blocks: list[nn.Module] = [] if modules is None else list(modules)
        assert len(blocks) ^ len(cfg), "Provide either blocks, or hyperparameters!"

        if cfg:
            return cls.from_config(**cfg)

        return super().__new__(cls)

    def __init__(self, *modules: nn.Module, inverse: Optional[Self] = None) -> None:
        r"""Initialize from hyperparameters."""
        super().__init__()

        blocks: list[nn.Module] = [] if modules is None else list(modules)
        # assert len(blocks) ^ len(hparams), "Provide either blocks, or hyperparameters!"
        # if hparams:
        #     raise ValueError

        self.input_size = -1
        self.output_size = -1

        # self.input_size = blocks[0].input_size  # type: ignore[assignment]
        # self.output_size = blocks[-1].output_size  # type: ignore[assignment]
        self.blocks = nn.Sequential(*blocks)

        self.is_inverse = inverse is not None
        if not self.is_inverse:
            cls = type(self)
            self.inverse = cls(
                *[block.inverse for block in reversed(self.blocks)], inverse=self
            )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Compute the encoding."""
        return self.blocks(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Compute the encoding."""
        return self.blocks(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fix point iteration in each block in reversed order."""
        for layer in self.blocks[::-1]:  # traverse in reverse
            y = layer.decode(y)
        return y
