r"""Diffeomorphisms, i.e. differentiable bijections with differentiable inverse.

We call a module a bijection if it satisfies 3 properties:

1. It has both an `encode` and `decode` method.
2. It is invertible, i.e. `decode(encode(x)) = x` and `encode(decode(y)) = y`
3. Both encode and decode are differentiable.
"""

__all__ = [
    # constants
    "BIJECTIONS",
    # protocols & base classes
    "Bijection",
    "BijectionABC",
    # classes
    "iResNet",
    "iResNetBlock",
    "iResNetLayer",
    "iLowRankLayer",
]


from linodenet.bijections.base import Bijection, BijectionABC
from linodenet.bijections.iresnet import iResNet, iResNetBlock, iResNetLayer
from linodenet.bijections.low_rank_perturbation import iLowRankLayer

BIJECTIONS: dict[str, type[Bijection]] = {
    "iResNet": iResNet,
}  # fmt: skip
r"""Dictionary containing all available bijections."""
