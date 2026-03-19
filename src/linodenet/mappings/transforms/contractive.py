r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = ["ContractiveNew", "ContractiveTransform"]

import warnings

import torch
from torch import Tensor, nn

from linodenet.mappings.base import TransformBase


class ContractiveTransform(TransformBase):
    r"""A residual flow based on a contraction layer.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration.

    The jacobian determinant of the forward transformation is:

    .. math:: \log\det(∂y/∂x) = \log\det(𝕀 + ∂g/∂x) = \tr\log(𝕀 + ∂g/∂x)

    Using the power series, this is

    .. math:: ∑_k (-1)ᵏ⁺¹ \tr((∂g/∂x)ᵏ)/k

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | https://proceedings.mlr.press/v97/behrmann19a.html
        - https://github.com/jhjacobsen/invertible-resnet
    """

    def __init__(
        self,
        contraction: nn.Module,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.contraction = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or the elementwise tolerance threshold
        $|x'-x| ≤ \text{rtol}⋅|x| + \text{atol}$ is reached.
        """
        x = y.clone()

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.contraction(x_prev)
            residual = torch.abs(x - x_prev)
            tolerance = self.rtol * torch.abs(x) + self.atol

            if torch.all(residual <= tolerance):
                return x

        warnings.warn(
            f"No convergence in {self.maxiter} iterations. ",
            stacklevel=2,
        )
        return x

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ContractiveNew(TransformBase):
    r"""A residual flow based on a contraction layer.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration implemented with `torch.while_loop`.

    The jacobian determinant of the forward transformation is:

    .. math:: \log\det(∂y/∂x) = \log\det(𝕀 + ∂g/∂x) = \tr\log(𝕀 + ∂g/∂x)

    Using the power series, this is

    .. math:: ∑_k (-1)ᵏ⁺¹ \tr((∂g/∂x)ᵏ)/k

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | https://proceedings.mlr.press/v97/behrmann19a.html
        - https://github.com/jhjacobsen/invertible-resnet
    """

    maxiter: Tensor
    rtol: Tensor
    atol: Tensor

    def __init__(
        self,
        contraction: nn.Module,
        *,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.contraction = contraction
        self.register_buffer("maxiter", torch.as_tensor(maxiter, dtype=torch.int32))
        self.register_buffer("atol", torch.as_tensor(atol))
        self.register_buffer("rtol", torch.as_tensor(rtol))

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    type State = tuple[Tensor, Tensor, Tensor, Tensor]
    #                  budget, x,      residual, y

    def cond_fn(self, state: State, /) -> Tensor:
        budget, x, residual, _ = state
        tolerance = self.rtol * x.abs() + self.atol
        return (budget > 0) & (residual > tolerance).any()

    def body_fn(self, state: State, /) -> State:
        budget, x, _, y = state

        # PERF: unroll 8 iterations (convergence check expensive)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x = y - self.contraction(x)
        x_new = y - self.contraction(x)

        return budget - 1, x_new, (x_new - x).abs(), y.clone()

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or the elementwise tolerance threshold
        $|x'-x| ≤ \text{rtol}⋅|x| + \text{atol}$ is reached.
        """
        x0 = y.clone()
        r0 = torch.full_like(x0, torch.inf)

        initial_state = (self.maxiter, x0, r0, y)
        final_state = torch.while_loop(self.cond_fn, self.body_fn, (initial_state,))

        _, x, _, _ = final_state  # pyright: ignore[reportGeneralTypeIssues]

        return x

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
