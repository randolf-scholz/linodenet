#!/usr/bin/env python

from typing import Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.nn import functional
from torch.optim import SGD
from typing_extensions import Self


class LinearEmbedding(nn.Module):
    r"""Maps $x ↦ Ax$ and $y ↦ A⁺y$."""

    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    is_inverse: Final[bool]
    r"""CONST: Whether this is the inverse of another module."""

    # PARAMS
    weight: Tensor
    r"""PARAM: The weight matriz."""
    weight_pinv: Tensor
    r"""BUFFER: The pseudo-inverse of the weight matriz."""

    def __init__(
        self, input_size: int, output_size: int, inverse: Optional[Self] = None
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        self.register_buffer("weight_pinv", torch.empty((input_size, output_size)))
        self.reset_parameters()

        self.is_inverse = inverse is not None
        if inverse is None:
            self.inverse = LinearEmbedding(
                input_size=output_size, output_size=input_size, inverse=self
            )
            self.inverse.weight = self.weight
            self.inverse.weight_pinv = self.weight_pinv

    @jit.export
    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)

    @jit.export
    def recompute_pinv(self) -> None:
        self.weight_pinv = self.weight.pinverse()

    @jit.export
    def reset_cache(self) -> None:
        self.weight_pinv.detach_()
        self.recompute_pinv()

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        if self.is_inverse:
            return self._encode(y)
        return self._decode(y)

    @jit.export
    def _encode(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.weight)

    @jit.export
    def _decode(self, y: Tensor) -> Tensor:
        return functional.linear(y, self.weight_pinv)


if __name__ == "__main__":
    N, m, n = 8, 32, 16
    x = torch.randn(8, m)
    model = LinearEmbedding(m, n)
    optim = SGD(model.parameters(), lr=1e-3)
    model = jit.script(model)

    for k in range(3):
        model.zero_grad()
        y = model(x).norm()
        y.backward()
        optim.step()
