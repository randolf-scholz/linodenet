r"""Functional Algebra.

| operator | meaning  | i-operator | iterated   |
|----------|----------|------------|------------|
| `>>`     | series   | `**`       | repeat     |
| `^`      | parallel | `//`       | concurrent |
| `&`      | meet     | `%`        | fork       |
| `|`      | join     | `%`        | fork       |

# @: tensor product?
# +: sum-reduce
# -: ?
# / num: ? reduce?
# % num: ? reduce?
# // num: ? concurrent/duplicate?
# ** num: repeat (>>)
# * num: duplicate? (⇝ similar to (x,) * n)
"""

__all__ = [
    "Fn",
    "FnSequence",
    "SupportsLenAndGetItem",
    "FunctionalMixin",
    "Seq",
    "Reduction",
    "WrappedFn",
    # Classes
    "Choice",
    "Replicate",
    "Diagonal",
    "Duplicate",
    "Identity",
    "Map",
    "Fork",
    "Parallel",
    "Reduce",
    "Repeat",
    "Series",
    "Sum",
    # Functions
    "choice",
    "replicate",
    "diagonal",
    "duplicate",
    "identity",
    "fork",
    "parallel",
    "reduce",
    "repeat",
    "series",
]

import random
from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Reversible, Sequence
from copy import deepcopy
from typing import (
    Any,
    Final,
    Optional,
    Protocol,
    Self,
    SupportsIndex,
    overload,
    runtime_checkable,
)


class SupportsLenAndGetItem[T](Protocol):
    r"""Protocol for types that support `__len__` and `__getitem__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsIndex, /) -> T: ...


@runtime_checkable
class Seq[T](Collection[T], Reversible[T], Protocol):  # +T
    r"""Protocol version of `collections.abc.Sequence`.

    Note:
        Only compatible with `tuple[T, ...]`, not `tuple[*Ts]` when using pyright.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
        - https://github.com/python/cpython/blob/main/Lib/_collections_abc.py
    """

    @overload
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...


# region base classes ------------------------------------------------------------------


@runtime_checkable
class Fn(Protocol):
    r"""Base protocol for functional modules."""

    def __invert__(self) -> "Fn":
        return NotImplemented

    @abstractmethod
    def __call__(self, arg: Any, /) -> Any: ...


# type Reduction[T] = Callable[[SupportsLenAndGetItem[T]], T]


class Reduction[T](Protocol):
    r"""Protocol for reduction functions."""

    @abstractmethod
    def __call__(self, xs: Seq[T], /) -> T: ...


class FnSequence[M: Fn](Sequence[M], Fn):
    r"""Base class for sequences of functional modules."""

    @classmethod
    def from_iterable(cls, seq: Iterable[M], /) -> Self:
        raise NotImplementedError(
            "Subclasses must override from_iterable to enable slicing"
        )

    def __init__(self, seq: Iterable[M] = (), /) -> None:
        super().__init__()
        self._data: Final[tuple[M, ...]] = tuple(seq)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> M: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __getitem__(self, index: SupportsIndex | slice, /) -> M | Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, slice):
            return self.from_iterable(self._data[index])
        return self._data[index]


class FunctionalMixin(Fn, Protocol):
    r"""Mixin for functional modules.

    This allows a nice way to chain modules together.
    """

    def __invert__(self) -> "FunctionalMixin":
        r"""Invert the module."""
        raise NotImplementedError

    # region series --------------------------------------------------------------------
    def __rshift__[N: Fn](self, other: N | Sequence[N], /) -> "Series[Self | N]":
        r"""Execute modules in series (`>>`).

        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
        """
        return series(self, other)

    def __rrshift__[N: Fn](self, other: N | Sequence[N], /) -> "Series[Self | N]":
        r"""Execute modules in series (`>>`).

        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
        """
        return series(other, self)

    def __pow__(self, n: int, /) -> "Series[Self]":
        r"""Repeat a module `n` times (`**`).

            x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

        Equivalent to `f >> f >> ... >> f` (n times).
        """
        return repeat(self, n)

    # endregion series -----------------------------------------------------------------

    # region parallel ------------------------------------------------------------------
    def __xor__[N: Fn](self: Self, other: N | Sequence[N], /) -> "Parallel[Self | N]":
        r"""Execute modules in parallel (`|`).

        x₁ ───▶ f₁(x₁)
        x₂ ───▶ f₂(x₂)
            ⋮
        xₙ ───▶ fₙ(xₙ)
        """
        return parallel(self, other)

    def __rxor__[N: Fn](self, other: N | Sequence[N], /) -> "Parallel[Self | N]":
        r"""Execute modules in parallel (`|`).

        x₁ ───▶ f₁(x₁)
        x₂ ───▶ f₂(x₂)
            ⋮
        xₙ ───▶ fₙ(xₙ)
        """
        return parallel(other, self)

    @overload
    def __floordiv__(self, num: int, /) -> "Replicate[Self]": ...
    @overload
    def __floordiv__(self, num: None = ..., /) -> "Map[Self]": ...
    def __floordiv__(self, num: int | None = None, /) -> "Replicate[Self] | Map[Self]":
        r"""Repeat a single module in parallel (`//`).

            x₁ ─────▶ f(x₁)
            x₂ ─────▶ f(x₂)
                 ⋮
            xₙ ─────▶ f(xₙ)

        Note: If `num` is `None`, the module will be executed for each input.
        """
        if num is None:
            return Map(self)
        return replicate(self, num)

    # endregion parallel ---------------------------------------------------------------

    # region meet ----------------------------------------------------------------------
    def __and__[N: Fn](self, other: N | Sequence[N], /) -> "Fork[Self | N]":
        r"""Execute multiple modules with the same input (`&`).

                  ┌────▶ f₁(x)
            x ────┼────▶ f₂(x)
                  │       ⋮
                  └────▶ fₙ(x)

        meet:
            Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∩…∩Xₙ，Y₁×…×Yₙ)
            (f₁，…，fₙ) ⟼ meet(f₁，…，fₙ)

        meet(f₁，…，fₙ):
            X₁∩…∩Xₙ ⟶ Y₁×…×Yₙ
            x ⟼ (f₁(x), ..., fₙ(x))
        """
        return fork(self, other)

    def __rand__[N: Fn](self, other: N | Sequence[N], /) -> "Fork[Self | N]":
        r"""Execute multiple modules with the same input (`&`).

                  ┌────▶ f₁(x)
            x ────┼────▶ f₂(x)
                  │       ⋮
                  └────▶ fₙ(x)

        meet:
            Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∩…∩Xₙ，Y₁×…×Yₙ)
            (f₁，…，fₙ) ⟼ meet(f₁，…，fₙ)

        meet(f₁，…，fₙ):
            X₁∩…∩Xₙ ⟶ Y₁×…×Yₙ
            x ⟼ (f₁(x), ..., fₙ(x))
        """
        return fork(other, self)

    def __mod__(self, n: int, /) -> "Duplicate[Self]":
        r"""Execute multiple copies of the same module with the same input (`%`).

              ┌────▶ f(x)
        x ────┼────▶ f(x)
              │       ⋮
              └────▶ f(x)
        """
        return duplicate(self, n)

    # endregion meet -------------------------------------------------------------------

    # region join ----------------------------------------------------------------------
    def __or__[N: Fn](self, other: N | Sequence[N], /) -> "Fork[Self | N]":
        r"""Join multiple outputs into a single output (`|`).

                  ┌────▶ f₁(x) or None
            x ────┼────▶ f₂(x) or None
                  │       ⋮
                  └────▶ fₙ(x) or None

        join:
            Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∪…∪Xₙ，Y₁'×…×Yₙ')
            (f₁，…，fₙ) ⟼ union(f₁，…，fₙ)

        join(f₁，…，fₙ):
            X₁∪…∪Xₙ ⟶ Y₁'×…×Yₙ'
            x ⟼ (f₁(x) or None , ..., fₙ(x) or None)
        """
        return fork(self, other)

    def __ror__[N: Fn](self, other: N | Sequence[N], /) -> "Fork[Self | N]":
        r"""Join multiple outputs into a single output (`|`).

                ┌────▶ f₁(x) or None
            x ──┼────▶ f₂(x) or None
                │       ⋮
                └────▶ fₙ(x) or None

        join:
            Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∪…∪Xₙ，Y₁'×…×Yₙ')
            (f₁，…，fₙ) ⟼ union(f₁，…，fₙ)

        join(f₁，…，fₙ):
            X₁∪…∪Xₙ ⟶ Y₁'×…×Yₙ'
            x ⟼ (f₁(x) or None , ..., fₙ(x) or None)
        """
        return fork(other, self)

    # endregion join -------------------------------------------------------------------


# endregion base classes ---------------------------------------------------------------


class Identity(Fn):
    r"""Identity module."""

    def __call__[T](self, x: T, /) -> T:
        return x


identity: Final[Identity] = Identity()  # canonical instance


class WrappedFn[T: Fn](Fn):
    r"""Wrap a function to make it a Fn-member."""

    def __init__(self, fn: T, /) -> None:
        r"""Wrap a function to make it callable."""
        super().__init__()
        self.fn: Final[T] = fn

    def __invert__(self) -> "WrappedFn":
        r"""Invert the wrapped function."""
        return WrappedFn(~self.fn)

    def __call__(self, arg: Any, /) -> Any:
        r"""Call the wrapped function."""
        return self.fn(arg)


class Series[M: Fn](FnSequence[M]):
    r"""Execute modules in series (`>>`).

        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y

    .. math:: series(f₁, ..., fₙ):
        X₁ ⟶ X₂ ⟶ ... ⟶ Xₙ ⟶ Y
        x ⟼ f₁(x) ⟼ f₂(f₁(x)) ⟼ ... ⟼ fₙ(fₙ₋₁(...f₁(x)...)) ⟼ y
    """

    def __invert__(self) -> "Series":
        return Series(~module for module in reversed(self))

    def __call__(self, x: Any, /) -> Any:
        for module in self:
            x = module(x)
        return x


@overload
def series[M: Fn, N: Fn](x: M, y: N, /) -> Series[M | N]: ...
@overload
def series[M: Fn, N: Fn](x: M, y: Sequence[N], /) -> Series[M | N]: ...
@overload
def series[M: Fn, N: Fn](x: Sequence[M], y: N, /) -> Series[M | N]: ...
@overload
def series[M: Fn, N: Fn](x: Sequence[M], y: Sequence[N], /) -> Series[M | N]: ...
def series[M: Fn, N: Fn](x: M | Sequence[M], y: N | Sequence[N], /) -> Series[M | N]:
    r"""Execute modules in series (`>>`).

        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y

    .. math:: series(f₁, ..., fₙ):
        X₁ ⟶ X₂ ⟶ ... ⟶ Xₙ ⟶ Y
        x ⟼ f₁(x) ⟼ f₂(f₁(x)) ⟼ ... ⟼ fₙ(fₙ₋₁(...f₁(x)...)) ⟼ y
    """
    match x, y:
        case Fn(), Fn():
            return Series((x, y))
        case Fn(), [*modules]:
            return Series((x, *modules))
        case [[*modules], Fn()]:
            return Series((*modules, y))
        case [[*left_modules], [*right_modules]]:
            return Series((*left_modules, *right_modules))
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Repeat[M: Fn](Series[M]):
    r"""Repeat a module `n` times (`**`).

        x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

    NOTE: Equivalent to `f >> f >> ... >> f` (n times).
    """

    def __invert__(self) -> "Repeat":
        return Repeat(self.module, -self.num)

    def __init__(self, module: M, num: int, /) -> None:
        if num < 0:
            raise ValueError("num must be >= 0")
        super().__init__([deepcopy(module) for _ in range(num)])
        self.num: Final[int] = num
        self.module: Final[M] = module


def repeat[M: Fn](module: M, num: int, /) -> Repeat[M]:
    r"""Repeat a module `n` times in series (`**`).

        x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

    NOTE: Equivalent to `f >> f >> ... >> f` (n times).
    """
    return Repeat(module, num)


class Parallel[M: Fn](FnSequence[M]):
    r"""Execute modules in parallel (`|`).

        x₁ ────▶ f₁(x₁)
        x₂ ────▶ f₂(x₂)
             ⋮
        xₙ ────▶ fₙ(xₙ)

    .. math:: parallel(f₁, ..., fₙ):
        X₁×…×Xₙ ⟶ Y₁×…×Yₙ
        (x₁, ..., xₙ) ⟼ (f₁(x₁), ..., fₙ(xₙ))
    """

    def __invert__(self) -> "Parallel":
        return Parallel(~module for module in self)

    # actual: tuple[*Xs] -> tuple[*Ys]
    # Modules: tuple[M[X, Y] for X, Y in zip(Xs, Ys)]
    def __call__(self, xs: tuple, /) -> tuple:
        r""".. Signature:: ``(..., n) -> [..., (..., n)]``."""
        return tuple(module(x) for x, module in zip(xs, self, strict=True))


@overload
def parallel[M: Fn, N: Fn](x: M, y: N, /) -> Parallel[M | N]: ...
@overload
def parallel[M: Fn, N: Fn](x: M, y: Sequence[N], /) -> Parallel[M | N]: ...
@overload
def parallel[M: Fn, N: Fn](x: Sequence[M], y: N, /) -> Parallel[M | N]: ...
@overload
def parallel[M: Fn, N: Fn](x: Sequence[M], y: Sequence[N], /) -> Parallel[M | N]: ...
def parallel[M: Fn, N: Fn](x: M | Sequence[M], y: N | Sequence[N], /) -> Parallel[M | N]:  # fmt: skip
    r"""Execute modules in parallel (`^`).

        x₁ ────▶ f₁(x₁)
        x₂ ────▶ f₂(x₂)
             ⋮
        xₙ ────▶ fₙ(xₙ)

    .. math:: parallel(f₁, ..., fₙ):
        X₁×…×Xₙ ⟶ Y₁×…×Yₙ
        (x₁, ..., xₙ) ⟼ (f₁(x₁), ..., fₙ(xₙ))
    """
    match x, y:
        case Fn(), Fn():
            return Parallel((x, y))
        case Fn(), [*modules]:
            return Parallel((x, *modules))
        case [[*modules], Fn()]:
            return Parallel((*modules, y))
        case [[*left_modules], [*right_modules]]:
            return Parallel((*left_modules, *right_modules))
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Replicate[M: Fn](Fn):
    r"""Apply copies of single module in parallel to multiple inputs (MIMO).

        x₁ ────▶ f(x₁)
        x₂ ────▶ f(x₂)
             ⋮
        xₙ ────▶ f(xₙ)

    .. math:: concurrent(f, n):
        X×…×X ⟶ Y×…×Y
        (x, ..., x) ⟼ (f(x), ..., f(x))
    """

    def __init__(self, module: M, num: int, /) -> None:
        if num < 0:
            raise ValueError(f"n must be non-negative, got {num}")
        super().__init__()
        self.num: Final[int] = num
        self.module: Final[M] = module

    def __call__(self, xs: tuple, /) -> tuple:
        return tuple(self.module(x) for x, _ in zip(xs, range(self.num), strict=True))


def replicate[M: Fn](module: M, num: int, /) -> Replicate[M]:
    r"""Apply copies of a single module in parallel (`//`).

        x₁ ────▶ f(x₁)
        x₂ ────▶ f(x₂)
             ⋮
        xₙ ────▶ f(xₙ)

    Note: If `num` is `None`, the module will be executed for each input.
    """
    return Replicate(module, num)


class Map[M: Fn]:
    r"""Map a module over variable length inputs."""

    def __init__(self, module: M, /) -> None:
        self.module: Final[M] = module

    def __call__(self, xs: list, /) -> list:
        return [self.module(x) for x in xs]


class Fork[M: Fn](FnSequence[M]):
    r"""Execute multiple modules with the same input (`&`).

              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │        ⋮
              └────▶ fₙ(x)

    .. math::
        Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∩…∩Xₙ，Y₁×…×Yₙ)  \\
        (f₁，…，fₙ) ⟼ fork(f₁，…，fₙ)

        fork(f₁，…，fₙ):
            X₁∩…∩Xₙ ⟶ Y₁×…×Yₙ   \\
            x ⟼ (f₁(x), ..., fₙ(x))

    Note:
        This is a "weak" meet, rather than the actual meet with respect to subtyping relationship.
        It crashes if the input is not compatible with all components.
    """

    def __init__(self, seq: Iterable[M] = (), /) -> None:
        super().__init__(seq)

    def __call__(self, x: Any, /) -> tuple:
        return tuple(module(x) for module in self)


@overload
def fork[M: Fn, N: Fn](x: M, y: N, /) -> Fork[M | N]: ...
@overload
def fork[M: Fn, N: Fn](x: M, y: Sequence[N], /) -> Fork[M | N]: ...
@overload
def fork[M: Fn, N: Fn](x: Sequence[M], y: N, /) -> Fork[M | N]: ...
@overload
def fork[M: Fn, N: Fn](x: Sequence[M], y: Sequence[N], /) -> Fork[M | N]: ...
def fork[M: Fn, N: Fn](x: M | Sequence[M], y: N | Sequence[N], /) -> Fork[M | N]:
    r"""Execute multiple modules with the same input (`&`).

              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │        ⋮
              └────▶ fₙ(x)

    Note: equivalent to diagonal(num) >> parallel(m1, m2, ..., mn)
    """
    match x, y:
        case Fn(), Fn():
            return Fork((x, y))
        case Fn(), [*modules]:
            return Fork((x, *modules))
        case [[*modules], Fn()]:
            return Fork((*modules, y))
        case [[*left_modules], [*right_modules]]:
            return Fork((*left_modules, *right_modules))
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Duplicate[M: Fn](Fn):
    r"""Duplicate a single input into multiple outputs.

              ┌────▶ f(x)
        x ────┼────▶ f(x)
              │        ⋮
              └────▶ f(x)

    .. math:: fork(f, n):
        X ⟶ Y×…×Y
        x ⟼ (f(x), ..., f(x))
    """

    def __init__(self, module: M, num: int, /) -> None:
        if num < 0:
            raise ValueError(f"Expected {num=} to be greater than 0")
        super().__init__()
        self.num: Final[int] = int(num)
        self.module: Final[M] = module

    def __call__(self, x: Any, /) -> tuple:
        return (self.module(x),) * self.num


def duplicate[M: Fn](module: M, num: int, /) -> Duplicate[M]:
    r"""Execute multiple copies of the same module with the same input (`%`).

              ┌────▶ f(x)
        x ────┼────▶ f(x)
              │        ⋮
              └────▶ f(x)

    Note: equivalent to diagonal(num) >> replicate(module, num)
    """
    return Duplicate(module, num)


class Diagonal(Duplicate[Identity]):
    r"""Duplicate a single input into multiple outputs.

              ┌────▶ x
        x ────┼────▶ x
              │      ⋮
              └────▶ x

    Note: equivalent to fork(Identity, num)
    """

    def __invert__(self) -> "Choice":
        return Choice(self.num)

    def __init__(self, num: int, /) -> None:
        super().__init__(identity, num)

    def __call__[X](self, x: X, /) -> tuple[X, ...]:
        return (x,) * self.num


def diagonal(num: int, /) -> Diagonal:
    r"""Duplicate a single input into multiple outputs.

              ┌────▶ x
        x ────┼────▶ x
              │      ⋮
              └────▶ x

    Note: equivalent to fork(Identity, num)
    """
    return Diagonal(num)


class Reduce[X = Any](Fn):
    r"""Reduce the inputs from multiple arguments.

        x₁ ────┐
        x₂ ────┼────▶ aggregate(x₁, x₂, ..., xₙ)
        ⋮      │
        xₙ ────┘

    Typical reductions are:

    - `min`, `max`, `median` for ordered data
    - `any`, `all` for boolean data
    - `sum`, `mean`, `prod`, `std`, `var`, `logsumexp` for numerical data
    - `stack`, `concat` for tensor data
    - `random.choice` for generic data
    """

    def __init__(self, reduction: Reduction[X], /) -> None:
        super().__init__()
        self.reduction: Final[Reduction[X]] = reduction

    def __call__(self, xs: Seq[X], /) -> X:
        return self.reduction(xs)


def reduce(reducer: Callable, /) -> Reduce:
    r"""Reduce the inputs from multiple arguments.

    x₁ ────┐
    x₂ ────┼────▶ aggregate(x₁, x₂, ..., xₙ)
    ⋮      │
    xₙ ────┘
    """
    return Reduce(reducer)


class Choice[X = Any](Reduce[X]):
    r"""Randomly choose one of the inputs.

    x₁ ────┐
    x₂ ────┼────▶ choice(x₁, x₂, ..., xₙ)
    ⋮      │
    xₙ ────┘
    """

    def __init__(self, num: Optional[int] = None) -> None:
        self.num: Final[int | None] = num
        super().__init__(random.choice)

    def __invert__(self) -> Diagonal:
        if self.num is None:
            raise ValueError("Cannot invert choice node with dynamic number of inputs")
        return Diagonal(self.num)


def choice(num: Optional[int] = None) -> Choice:
    r"""Randomly choose one of the inputs.

    x₁ ────┐
    x₂ ────┼────▶ choice(x₁, x₂, ..., xₙ)
    ⋮      │
    xₙ ────┘

    Note: If `num` is `None`, the number of inputs is dynamic.
    """
    return Choice(num)


class Sum(Reduce):
    r"""Sum the outputs of multiple modules (`+`).

    x₁ ────┐
    x₂ ────┼───▶ x₁ + x₂ + ... + xₙ
    ⋮      │
    xₙ ────┘
    """

    def __init__(self) -> None:
        super().__init__(sum)
