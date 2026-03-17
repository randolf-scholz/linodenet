r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = ["ContractiveFlow"]

from torch import Tensor, nn

from .base import TransformBase


class ContractiveFlow(TransformBase):
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
        # TODO: fix point iteration x = y - g(x)
        pass

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
