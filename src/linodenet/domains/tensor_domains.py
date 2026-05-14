r"""Tensor domain labels and their partial-order relations."""

__all__ = ["TensorDomains"]

from types import MappingProxyType

from torch import Tensor

from .base import PosetEnum


class TensorDomains(PosetEnum):
    r"""Enumeration of some tensor domains for tensors of arbitrary rank."""

    ANY = "any"  # top node
    NONE = "none"  # bottom node

    REAL = "real"
    COMPLEX = "complex"
    BOOLEAN = "boolean"

    SPARSE = "sparse"

    ZERO = "zero"  # xᵢ...ⱼ = 0 for all entries
    ONE = "one"  # xᵢ...ⱼ = 1 for all entries
    NONZERO = "nonzero"  # x ≠ 0

    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError


T = TensorDomains  # temporary alias
TensorDomains.KNOWN_SUPERTYPES = MappingProxyType({
    T.BOOLEAN: {T.REAL},
    T.ONE: {T.BOOLEAN, T.NONZERO},
    T.REAL: {T.COMPLEX},
    T.ZERO: {T.BOOLEAN, T.SPARSE},
})  # fmt: skip
TensorDomains.KNOWN_SUBTYPES = MappingProxyType({})  # pyright: ignore[reportAttributeAccessIssue]
del T  # remove alias
