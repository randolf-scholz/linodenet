r"""Composition and reduction of callable modules.

The `FunctionalMixin` operators construct the following compositions:

| Operator | Composition    | Repetition operator | Repeated operation                  |
|----------|----------------|---------------------|-------------------------------------|
| `>>`     | Series         | `** n`              | Compose deep copies in series       |
| `^`      | Parallel       | `// n`              | Apply one module to $n$ inputs      |
| `&`      | Fork           | `% n`               | Evaluate once and repeat the output |
| `\|`     | Alias for fork |                     |                                     |

`// None` maps a module over a variable-length input list. `Diagonal` repeats
an input without copying it. `Reduce`, `Choice`, and `Sum` reduce a sequence
of inputs. Inversion with `~` is available where the callable implements it.
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

    def __invert__(self) -> Fn:
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
    def __getitem__(self, index: SupportsIndex | slice, /) -> M | Self:
        if isinstance(index, slice):
            return self.from_iterable(self._data[index])
        return self._data[index]


class FunctionalMixin(Fn, Protocol):
    r"""Mixin providing operators for callable composition.

    The operators construct series, parallel, fork, and repetition nodes.
    """

    def __invert__(self) -> FunctionalMixin:
        r"""Invert the module."""
        raise NotImplementedError

    # region series --------------------------------------------------------------------
    def __rshift__[N: Fn](self, other: N | Sequence[N], /) -> Series[Self | N]:
        r"""Execute modules in series (`>>`).

        ```text
        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
        ```
        """
        return series(self, other)

    def __rrshift__[N: Fn](self, other: N | Sequence[N], /) -> Series[Self | N]:
        r"""Execute modules in series (`>>`).

        ```text
        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
        ```
        """
        return series(other, self)

    def __pow__(self, n: int, /) -> Series[Self]:
        r"""Repeat a module `n` times (`**`).

        ```text
        x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)
        ```

        This constructs `n` deep copies of the module in series. The count must
        be nonnegative; zero repetitions return the input unchanged.
        """
        return repeat(self, n)

    # endregion series -----------------------------------------------------------------

    # region parallel ------------------------------------------------------------------
    def __xor__[N: Fn](self: Self, other: N | Sequence[N], /) -> Parallel[Self | N]:
        r"""Execute modules in parallel (`^`).

        ```text
        x₁ ───▶ f₁(x₁)
        x₂ ───▶ f₂(x₂)
            ⋮
        xₙ ───▶ fₙ(xₙ)
        ```
        """
        return parallel(self, other)

    def __rxor__[N: Fn](self, other: N | Sequence[N], /) -> Parallel[Self | N]:
        r"""Execute modules in parallel (`^`).

        ```text
        x₁ ───▶ f₁(x₁)
        x₂ ───▶ f₂(x₂)
            ⋮
        xₙ ───▶ fₙ(xₙ)
        ```
        """
        return parallel(other, self)

    @overload
    def __floordiv__(self, num: int, /) -> Replicate[Self]: ...
    @overload
    def __floordiv__(self, num: None = ..., /) -> Map[Self]: ...
    def __floordiv__(self, num: int | None = None, /) -> Replicate[Self] | Map[Self]:
        r"""Repeat a single module in parallel (`//`).

        ```text
        x₁ ─────▶ f(x₁)
        x₂ ─────▶ f(x₂)
             ⋮
        xₙ ─────▶ f(xₙ)
        ```

        Note:
            If `num` is `None`, return `Map` for a variable-length input list.
            Otherwise, return `Replicate`, which requires exactly `num` inputs.
        """
        if num is None:
            return Map(self)
        return replicate(self, num)

    # endregion parallel ---------------------------------------------------------------

    # region meet ----------------------------------------------------------------------
    def __and__[N: Fn](self, other: N | Sequence[N], /) -> Fork[Self | N]:
        r"""Execute multiple modules with the same input (`&`).

        ```text
              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │       ⋮
              └────▶ fₙ(x)
        ```

        $$\begin{aligned}
            \op{meet}(f₁,…,fₙ): X₁∩…∩Xₙ &→ Y₁×…×Yₙ \\
            x &↦ (f₁(x),…,fₙ(x))
        \end{aligned}$$

        The shared-input construction is a weak meet on function spaces.
        Here $\op{Fun}(X,Y)$ denotes the set of functions from $X$ to $Y$.

        $$\begin{aligned}
            \op{Fun}(X₁,Y₁)×…×\op{Fun}(Xₙ,Yₙ) &→ \op{Fun}(X₁∩…∩Xₙ,Y₁×…×Yₙ) \\
            (f₁,…,fₙ) &↦ \op{meet}(f₁,…,fₙ)
        \end{aligned}$$

        The input must be valid for every module.
        """
        return fork(self, other)

    def __rand__[N: Fn](self, other: N | Sequence[N], /) -> Fork[Self | N]:
        r"""Execute multiple modules with the same input (`&`).

        ```text
              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │       ⋮
              └────▶ fₙ(x)
        ```

        $$\begin{aligned}
            \op{meet}(f₁,…,fₙ): X₁∩…∩Xₙ &→ Y₁×…×Yₙ \\
            x &↦ (f₁(x),…,fₙ(x))
        \end{aligned}$$

        The shared-input construction is a weak meet on function spaces.
        Here $\op{Fun}(X,Y)$ denotes the set of functions from $X$ to $Y$.

        $$\begin{aligned}
            \op{Fun}(X₁,Y₁)×…×\op{Fun}(Xₙ,Yₙ) &→ \op{Fun}(X₁∩…∩Xₙ,Y₁×…×Yₙ) \\
            (f₁,…,fₙ) &↦ \op{meet}(f₁,…,fₙ)
        \end{aligned}$$
        """
        return fork(other, self)

    def __mod__(self, n: int, /) -> Duplicate[Self]:
        r"""Evaluate a module once and repeat its output (`%`).

        ```text
        x ──▶ f ──▶ y ──▶ (y, y, …, y)
        ```

        The tuple entries reference the same output object.
        """
        return duplicate(self, n)

    # endregion meet -------------------------------------------------------------------

    # region join ----------------------------------------------------------------------
    def __or__[N: Fn](self, other: N | Sequence[N], /) -> Fork[Self | N]:
        r"""Fork modules with the same input (`|`).

        ```text
              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │       ⋮
              └────▶ fₙ(x)
        ```

        $$\begin{aligned}
            \op{join}(f₁,…,fₙ): X₁∪…∪Xₙ &→ Y₁′×…×Yₙ′ \\
            x &↦ (f₁′(x),…,fₙ′(x))
        \end{aligned}$$

        For a mathematical join, let $Yᵢ′=Yᵢ∪\{\text{None}\}$. Extend each
        $fᵢ$ to a map $fᵢ′$ on the union of the input domains, returning `None`
        outside $Xᵢ$. The function-space construction and its evaluation are

        $$\begin{aligned}
            \op{Fun}(X₁,Y₁)×…×\op{Fun}(Xₙ,Yₙ) &→ \op{Fun}(X₁∪…∪Xₙ,Y₁′×…×Yₙ′) \\
            (f₁,…,fₙ) &↦ \op{join}(f₁,…,fₙ)
        \end{aligned}$$

        Note:
            This is an alias for `&`: every module receives the input, and the
            results are returned as a tuple. The input must be valid for every module.
            This operator does not implement the mathematical join above.
        """
        return fork(self, other)

    def __ror__[N: Fn](self, other: N | Sequence[N], /) -> Fork[Self | N]:
        r"""Fork modules with the same input (`|`).

        ```text
            ┌────▶ f₁(x)
        x ──┼────▶ f₂(x)
            │       ⋮
            └────▶ fₙ(x)
        ```

        $$\begin{aligned}
            \op{join}(f₁,…,fₙ): X₁∪…∪Xₙ &→ Y₁′×…×Yₙ′ \\
            x &↦ (f₁′(x),…,fₙ′(x))
        \end{aligned}$$


        For a mathematical join, let $Yᵢ′=Yᵢ∪\{\text{None}\}$. Extend each
        $fᵢ$ to a map $fᵢ′$ on the union of the input domains, returning `None`
        outside $Xᵢ$. The function-space construction and its evaluation are

        $$\begin{aligned}
            \op{Fun}(X₁,Y₁)×…×\op{Fun}(Xₙ,Yₙ) &→ \op{Fun}(X₁∪…∪Xₙ,Y₁′×…×Yₙ′) \\
            (f₁,…,fₙ) &↦ \op{join}(f₁,…,fₙ)
        \end{aligned}$$

        Note:
            This is an alias for `&`: every module receives the input, and the
            results are returned as a tuple. The input must be valid for every module.
            This operator does not implement the mathematical join above.
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
    r"""Wrap a callable as an instance of the `Fn` protocol."""

    def __init__(self, fn: T, /) -> None:
        r"""Store the callable to delegate calls and inversion to."""
        super().__init__()
        self.fn: Final[T] = fn

    def __invert__(self) -> WrappedFn:
        r"""Invert the wrapped function."""
        return WrappedFn(~self.fn)

    def __call__(self, arg: Any, /) -> Any:
        r"""Call the wrapped function."""
        return self.fn(arg)


class Series[M: Fn](FnSequence[M]):
    r"""Execute modules in series (`>>`).

    ```text
    x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
    ```

    $$\begin{aligned}
        \op{series}(f₁,…,fₙ): X₁ &→ Yₙ \\
        x &↦ fₙ(fₙ₋₁(…f₁(x)…))
    \end{aligned}$$

    Each module's output must be accepted by the next module. An empty series
    returns the input unchanged.
    """

    def __invert__(self) -> Series:
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

    ```text
    x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y
    ```

    $$\begin{aligned}
        \op{series}(f₁,…,fₙ): X₁ &→ Yₙ \\
        x &↦ fₙ(fₙ₋₁(…f₁(x)…))
    \end{aligned}$$

    Each module's output must be accepted by the next module. An empty series
    returns the input unchanged.
    """
    match x, y:
        case Fn(), Fn():
            return Series((x, y))  # ty: ignore[invalid-return-type]
        case Fn(), [*modules]:
            return Series((x, *modules))  # ty: ignore[invalid-argument-type]
        case [*modules], Fn():
            return Series((*modules, y))  # ty: ignore[invalid-argument-type]
        case [*left_modules], [*right_modules]:
            return Series((*left_modules, *right_modules))  # ty: ignore[invalid-argument-type]
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Repeat[M: Fn](Series[M]):
    r"""Repeat a module `num` times (`**`).

    ```text
    x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)
    ```

    This constructs `num` deep copies of the module in series. The count must
    be nonnegative; zero repetitions return the input unchanged.
    """

    def __invert__(self) -> Repeat:
        return Repeat(self.module, -self.num)

    def __init__(self, module: M, num: int, /) -> None:
        if num < 0:
            raise ValueError("num must be >= 0")
        super().__init__([deepcopy(module) for _ in range(num)])
        self.num: Final[int] = num
        self.module: Final[M] = module


def repeat[M: Fn](module: M, num: int, /) -> Repeat[M]:
    r"""Repeat a module `num` times in series (`**`).

    ```text
    x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)
    ```

    This constructs `num` deep copies of the module in series. The count must
    be nonnegative; zero repetitions return the input unchanged.
    """
    return Repeat(module, num)


class Parallel[M: Fn](FnSequence[M]):
    r"""Execute modules in parallel (`^`).

    ```text
    x₁ ────▶ f₁(x₁)
    x₂ ────▶ f₂(x₂)
         ⋮
    xₙ ────▶ fₙ(xₙ)
    ```

    $$\begin{aligned}
        \op{parallel}(f₁,…,fₙ): X₁×…×Xₙ &→ Y₁×…×Yₙ \\
        (x₁,…,xₙ) &↦ (f₁(x₁),…,fₙ(xₙ))
    \end{aligned}$$

    The input tuple must contain exactly one item per module. A length mismatch
    raises `ValueError`.
    """

    def __invert__(self) -> Parallel:
        return Parallel(~module for module in self)

    # actual: tuple[*Xs] -> tuple[*Ys]
    # Modules: tuple[M[X, Y] for X, Y in zip(Xs, Ys)]
    def __call__(self, xs: tuple, /) -> tuple:
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

    ```text
    x₁ ────▶ f₁(x₁)
    x₂ ────▶ f₂(x₂)
         ⋮
    xₙ ────▶ fₙ(xₙ)
    ```

    $$\begin{aligned}
        \op{parallel}(f₁,…,fₙ): X₁×…×Xₙ &→ Y₁×…×Yₙ \\
        (x₁,…,xₙ) &↦ (f₁(x₁),…,fₙ(xₙ))
    \end{aligned}$$

    The input tuple must contain exactly one item per module. A length mismatch
    raises `ValueError`.
    """
    match x, y:
        case Fn(), Fn():
            return Parallel((x, y))  # ty: ignore[invalid-return-type]
        case Fn(), [*modules]:
            return Parallel((x, *modules))  # ty: ignore[invalid-argument-type]
        case [*modules], Fn():
            return Parallel((*modules, y))  # ty: ignore[invalid-argument-type]
        case [*left_modules], [*right_modules]:
            return Parallel((*left_modules, *right_modules))  # ty: ignore[invalid-argument-type]
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Replicate[M: Fn](Fn):
    r"""Apply one module to each of `num` inputs.

    ```text
    x₁ ────▶ f(x₁)
    x₂ ────▶ f(x₂)
         ⋮
    xₙ ────▶ f(xₙ)
    ```

    $$\begin{aligned}
        \op{replicate}(f,n): X×…×X &→ Y×…×Y \\
        (x₁,…,xₙ) &↦ (f(x₁),…,f(xₙ))
    \end{aligned}$$

    The same module instance is reused for each input. `num` must be a nonnegative
    integer, and the input tuple must have exactly that many items. For a
    variable-length input list, use `Map`.
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
    r"""Apply one module to each of `num` inputs (`//`).

    ```text
    x₁ ────▶ f(x₁)
    x₂ ────▶ f(x₂)
         ⋮
    xₙ ────▶ f(xₙ)
    ```

    $$\begin{aligned}
        \op{replicate}(f,n): X×…×X &→ Y×…×Y \\
        (x₁,…,xₙ) &↦ (f(x₁),…,f(xₙ))
    \end{aligned}$$

    Note: If `num` is `None`, the module will be executed for each input.
    """
    return Replicate(module, num)


class Map[M: Fn]:
    r"""Map a module over variable-length inputs."""

    def __init__(self, module: M, /) -> None:
        self.module: Final[M] = module

    def __call__(self, xs: list, /) -> list:
        return [self.module(x) for x in xs]


class Fork[M: Fn](FnSequence[M]):
    r"""Execute multiple modules with the same input (`&`).

    ```text
          ┌────▶ f₁(x)
    x ────┼────▶ f₂(x)
          │        ⋮
          └────▶ fₙ(x)
    ```

    $$\begin{aligned}
        \op{fork}(f₁,…,fₙ): X₁∩…∩Xₙ &→ Y₁×…×Yₙ \\
        x &↦ (f₁(x),…,fₙ(x))
    \end{aligned}$$

    Note:
        This is a weak meet, not a meet in the subtyping order. The input must be
        valid for every module. A failing module propagates its exception;
        incompatible inputs are not replaced by `None`.
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

    ```text
          ┌────▶ f₁(x)
    x ────┼────▶ f₂(x)
          │        ⋮
          └────▶ fₙ(x)
    ```

    $$\begin{aligned}
        \op{fork}(f₁,…,fₙ): X₁∩…∩Xₙ &→ Y₁×…×Yₙ \\
        x &↦ (f₁(x),…,fₙ(x))
    \end{aligned}$$

    Note: equivalent to diagonal(num) >> parallel(m1, m2, ..., mn)
    """
    match x, y:
        case Fn(), Fn():
            return Fork((x, y))  # ty: ignore[invalid-return-type]
        case Fn(), [*modules]:
            return Fork((x, *modules))  # ty: ignore[invalid-argument-type]
        case [*modules], Fn():
            return Fork((*modules, y))  # ty: ignore[invalid-argument-type]
        case [*left_modules], [*right_modules]:
            return Fork((*left_modules, *right_modules))  # ty: ignore[invalid-argument-type]
        case _:
            raise TypeError(f"Expected Fn or Sequence[Fn], got {type(x)} and {type(y)}")


class Duplicate[M: Fn](Fn):
    r"""Evaluate a module once and repeat its output `num` times.

    ```text
    x ──▶ f ──▶ y ──▶ (y, y, …, y)
    ```

    $$\begin{aligned}
        \op{duplicate}(f,n): X &→ Y×…×Y \\
        x &↦ (f(x),…,f(x))
    \end{aligned}$$
    """

    def __init__(self, module: M, num: int, /) -> None:
        if num < 0:
            raise ValueError(f"Expected {num=} to be greater than 0")
        super().__init__()
        self.num: Final[int] = num
        self.module: Final[M] = module

    def __call__(self, x: Any, /) -> tuple:
        return (self.module(x),) * self.num


def duplicate[M: Fn](module: M, num: int, /) -> Duplicate[M]:
    r"""Evaluate a module once and repeat its output (`%`).

    ```text
    x ──▶ f ──▶ y ──▶ (y, y, …, y)
    ```

    $$\begin{aligned}
        \op{duplicate}(f,n): X &→ Y×…×Y \\
        x &↦ (f(x),…,f(x))
    \end{aligned}$$

    Note: equivalent to diagonal(num) >> replicate(module, num)
    """
    return Duplicate(module, num)


class Diagonal(Duplicate[Identity]):
    r"""Duplicate a single input into multiple outputs.

    ```text
          ┌────▶ x
    x ────┼────▶ x
          │      ⋮
          └────▶ x
    ```

    $$\begin{aligned}
        \op{diagonal}(n): X &→ X×…×X \\
        x &↦ (x,…,x)
    \end{aligned}$$

    Note: equivalent to fork(Identity, num)
    """

    def __invert__(self) -> Choice:
        return Choice(self.num)

    def __init__(self, num: int, /) -> None:
        super().__init__(identity, num)

    def __call__[X](self, x: X, /) -> tuple[X, ...]:
        return (x,) * self.num


def diagonal(num: int, /) -> Diagonal:
    r"""Duplicate a single input into multiple outputs.

    ```text
          ┌────▶ x
    x ────┼────▶ x
          │      ⋮
          └────▶ x
    ```

    $$\begin{aligned}
        \op{diagonal}(n): X &→ X×…×X \\
        x &↦ (x,…,x)
    \end{aligned}$$

    Note: equivalent to fork(Identity, num)
    """
    return Diagonal(num)


class Reduce[X = Any](Fn):
    r"""Reduce a single sequence of inputs to one output.

    ```text
    x₁ ────┐
    x₂ ────┼────▶ aggregate((x₁, x₂, …, xₙ))
    ⋮      │
    xₙ ────┘
    ```

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
    r"""Reduce a single sequence of inputs to one output.

    ```text
    x₁ ────┐
    x₂ ────┼────▶ aggregate((x₁, x₂, …, xₙ))
    ⋮      │
    xₙ ────┘
    ```
    """
    return Reduce(reducer)


class Choice[X = Any](Reduce[X]):
    r"""Randomly choose one of the inputs.

    ```text
    x₁ ────┐
    x₂ ────┼────▶ choice((x₁, x₂, …, xₙ))
    ⋮      │
    xₙ ────┘
    ```
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

    ```text
    x₁ ────┐
    x₂ ────┼────▶ choice((x₁, x₂, …, xₙ))
    ⋮      │
    xₙ ────┘
    ```

    Note: If `num` is `None`, the number of inputs is dynamic.
    """
    return Choice(num)


class Sum(Reduce):
    r"""Sum a sequence of inputs with the builtin `sum` function.

    ```text
    x₁ ────┐
    x₂ ────┼───▶ x₁ + x₂ + ... + xₙ
    ⋮      │
    xₙ ────┘
    ```
    """

    def __init__(self) -> None:
        super().__init__(sum)
