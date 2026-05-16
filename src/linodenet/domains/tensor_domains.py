r"""Tensor-specific domain primitives and partial-order labels."""

__all__ = [
    "TensorDomains",
    "Boolean",
    "Complex",
    "Empty",
    "NonZero",
    "One",
    "Real",
    "Sparse",
    "Tensor",
    "Zero",
]

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self, overload

import torch
from torch import Tensor as TorchTensor

from . import tensor_tests as tests
from .base import Indeterminate, PosetEnum, TensorDomain


@dataclass(frozen=True)
class Tensor(TensorDomain):
    r"""Domain of tensors with optional fixed trailing shape."""

    shape: Final[tuple[int, ...] | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_tensor(value, shape=self.shape)

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(self, shape: tuple[int, ...], /) -> Self: ...
    def __call__(self, shape: tuple[int, ...] | None = None, /) -> Self:
        return self.__class__(shape)


@dataclass(frozen=True)
class Complex(Tensor):
    r"""Domain of tensors with values in the complex numbers."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_complex_tensor(value, shape=self.shape)


@dataclass(frozen=True)
class Real(Complex):
    r"""Domain of tensors with real dtype."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_real_tensor(value, shape=self.shape)


@dataclass(frozen=True)
class Boolean(Tensor):
    r"""Domain of tensors whose entries are only zeros and ones."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_boolean_tensor(value, shape=self.shape)


@dataclass(frozen=True)
class Empty(Tensor):
    r"""Domain of tensors with no admissible values."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return value.new_full(Tensor.check(self, value).shape, False, dtype=torch.bool)


@dataclass(frozen=True)
class Zero(Tensor):
    r"""Domain of tensors whose entries are only zeros."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_zero_tensor(value, shape=self.shape)


@dataclass(frozen=True)
class One(Tensor):
    r"""Domain of tensors whose entries are only ones."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_one_tensor(value, shape=self.shape)


@dataclass(frozen=True)
class Sparse(Tensor):
    r"""Domain of tensors with sufficiently many exact zero entries."""

    _: KW_ONLY
    sparsity: Final[float | None] = None

    def __post_init__(self) -> None:
        if self.sparsity is not None and not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("Expected sparsity in [0, 1].")

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_sparse_tensor(
            value,
            self.sparsity,
            shape=self.shape,
        )

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(
        self,
        shape: tuple[int, ...],
        /,
        *,
        sparsity: float | None = None,
    ) -> Self: ...
    def __call__(
        self,
        shape: tuple[int, ...] | None = None,
        /,
        *,
        sparsity: float | None = None,
    ) -> Self:
        return self.__class__(shape, sparsity=sparsity)


@dataclass(frozen=True)
class NonZero(Tensor):
    r"""Domain of tensors that are not identically zero."""

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return tests.is_nonzero_tensor(value, shape=self.shape)


class TensorDomains(TensorDomain, PosetEnum):
    r"""Enumeration of some tensor domains for tensors of arbitrary rank."""

    ALIASES: ClassVar[Mapping[str, Self]]

    def __new__(cls, value: Tensor) -> Self:
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value: Tensor) -> None:
        del value

    ANY = Tensor()  # top node
    NONE = Empty()  # bottom node

    REAL = Real()
    COMPLEX = Complex()
    BOOLEAN = Boolean()

    SPARSE = Sparse()

    ZERO = Zero()  # xᵢ...ⱼ = 0 for all entries
    ONE = One()  # xᵢ...ⱼ = 1 for all entries
    NONZERO = NonZero()  # x ≠ 0

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            return cls.ALIASES.get(value)
        return None

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.value.shape

    def check(self, value: TorchTensor, /) -> TorchTensor:
        return self.value.check(value)

    def __eq__(self, other: object, /) -> bool:
        return PosetEnum.__eq__(self, other)

    def __hash__(self) -> int:
        return PosetEnum.__hash__(self)

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__le__(self, other)

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__lt__(self, other)

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__ge__(self, other)

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__gt__(self, other)


T = TensorDomains  # temporary alias
T.KNOWN_SUPERTYPES = MappingProxyType({
    T.BOOLEAN: {T.REAL},
    T.ONE: {T.BOOLEAN, T.NONZERO},
    T.REAL: {T.COMPLEX},
    T.ZERO: {T.BOOLEAN, T.SPARSE},
})  # fmt: skip
T.KNOWN_SUBTYPES = MappingProxyType({
    T.SPARSE: {T.ZERO},
})  # fmt: skip
T.ALIASES = MappingProxyType({
    "any"     : T.ANY,
    "boolean" : T.BOOLEAN,
    "complex" : T.COMPLEX,
    "empty"   : T.NONE,
    "none"    : T.NONE,
    "nonzero" : T.NONZERO,
    "one"     : T.ONE,
    "real"    : T.REAL,
    "sparse"  : T.SPARSE,
    "zero"    : T.ZERO,
})  # fmt: skip
del T
