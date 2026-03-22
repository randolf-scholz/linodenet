r"""Scalar domain primitives, including intervals and scalar domain labels."""

__all__ = ["Interval", "RealDomain", "ScalarDomains"]


from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from math import isnan
from typing import Final, overload

from torch import Tensor

from .base import Domain


@dataclass(unsafe_hash=True)
class Interval(Domain):
    r"""A named tuple representing an interval."""

    lower: Final[float]
    upper: Final[float]
    lower_inclusive: Final[bool]
    upper_inclusive: Final[bool]

    @staticmethod
    def parse(arg: object, /) -> Interval | None:
        match arg:
            case Interval():
                return arg
            case str():
                try:
                    return Interval.from_string(arg)
                except ValueError:
                    return None
            case _:
                return None

    @classmethod
    def from_string(cls, s: str, /) -> Interval:
        r"""Create an Interval from a string representation.

        Examples:
            >>> Interval.from_string("[0, 1)")
            Interval(lower=0.0, upper=1.0, lower_inclusive=True, upper_inclusive=False)
            >>> Interval.from_string("(-inf, inf)")
            Interval(lower=-inf, upper=inf, lower_inclusive=False, upper_inclusive=False)
        """
        s = s.strip()

        match s[0]:
            case "[":
                lower_inclusive = True
            case "(":
                lower_inclusive = False
            case _:
                raise ValueError(f"Invalid interval string: {s}")

        match s[-1]:
            case "]":
                upper_inclusive = True
            case ")":
                upper_inclusive = False
            case _:
                raise ValueError(f"Invalid interval string: {s}")

        bounds = s[1:-1].split(",")
        if len(bounds) != 2:
            raise ValueError(f"Invalid interval string: {s}")

        lower_str, upper_str = bounds
        lower = float(lower_str.strip())
        upper = float(upper_str.strip())

        return Interval(
            lower,
            upper,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )

    @overload
    def __init__(self, s: str | Interval, /) -> None: ...
    @overload
    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> None: ...
    def __init__(
        self,
        lower: str | Interval | float,
        upper: float | None = None,
        *,
        lower_inclusive: bool | None = None,
        upper_inclusive: bool | None = None,
    ) -> None:
        if isinstance(lower, str | Interval):
            if (
                upper is not None
                or lower_inclusive is not None
                or upper_inclusive is not None
            ):
                raise TypeError("String interval constructor does not accept bounds.")
            interval = (
                lower if isinstance(lower, Interval) else Interval.from_string(lower)
            )
            self.lower = interval.lower
            self.upper = interval.upper
            self.lower_inclusive = interval.lower_inclusive
            self.upper_inclusive = interval.upper_inclusive

        else:
            if upper is None or lower_inclusive is None or upper_inclusive is None:
                raise TypeError(
                    "Expected upper and inclusivity flags for numeric bounds."
                )

            self.lower = lower  # type: ignore[misc]
            self.upper = upper  # type: ignore[misc]
            self.lower_inclusive = lower_inclusive  # type: ignore[misc]
            self.upper_inclusive = upper_inclusive  # type: ignore[misc]

    def __contains__(self, item: Tensor, /) -> Tensor:
        lower_mask = (
            (item >= self.lower) if self.lower_inclusive else (item > self.lower)
        )
        upper_mask = (
            (item <= self.upper) if self.upper_inclusive else (item < self.upper)
        )
        return lower_mask & upper_mask

    def __add__(self, other: float, /) -> Interval:
        return Interval(
            self.lower + other,
            self.upper + other,
            lower_inclusive=self.lower_inclusive,
            upper_inclusive=self.upper_inclusive,
        )

    def __sub__(self, other: float, /) -> Interval:
        return self + (-other)

    def __mul__(self, other: float, /) -> Interval:
        if other == 0:
            return Interval(
                0.0,
                0.0,
                lower_inclusive=True,
                upper_inclusive=True,
            )
        if other > 0:
            return Interval(
                self.lower * other,
                self.upper * other,
                lower_inclusive=self.lower_inclusive,
                upper_inclusive=self.upper_inclusive,
            )
        return Interval(
            self.upper * other,
            self.lower * other,
            lower_inclusive=self.upper_inclusive,
            upper_inclusive=self.lower_inclusive,
        )

    def __and__(self, other: object, /) -> Interval:
        if not isinstance(other, Interval):
            return NotImplemented

        if self.isdisjoint(other):
            return Interval(
                float("nan"),
                float("nan"),
                lower_inclusive=False,
                upper_inclusive=False,
            )

        if self.lower > other.lower:
            lower = self.lower
            lower_inclusive = self.lower_inclusive
        elif self.lower < other.lower:
            lower = other.lower
            lower_inclusive = other.lower_inclusive
        else:
            lower = self.lower
            lower_inclusive = self.lower_inclusive and other.lower_inclusive

        if self.upper < other.upper:
            upper = self.upper
            upper_inclusive = self.upper_inclusive
        elif self.upper > other.upper:
            upper = other.upper
            upper_inclusive = other.upper_inclusive
        else:
            upper = self.upper
            upper_inclusive = self.upper_inclusive and other.upper_inclusive

        return Interval(
            lower,
            upper,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )

    def isdisjoint(self, other: Interval, /) -> bool:
        r"""Return whether two intervals have empty intersection."""
        if (isnan(self.lower) and isnan(self.upper)) or (
            isnan(other.lower) and isnan(other.upper)
        ):
            return True
        if self.upper < other.lower or other.upper < self.lower:
            return True
        if self.upper > other.lower and other.upper > self.lower:
            return False
        if self.upper == other.lower:
            return not (self.upper_inclusive and other.lower_inclusive)
        if other.upper == self.lower:
            return not (other.upper_inclusive and self.lower_inclusive)
        return False

    def __eq__(self, other: object, /) -> bool:
        if (other_interval := self.parse(other)) is None:
            return NotImplemented

        if (
            isnan(self.lower)
            and isnan(self.upper)
            and isnan(other_interval.lower)
            and isnan(other_interval.upper)
        ):
            return (
                self.lower_inclusive == other_interval.lower_inclusive
                and self.upper_inclusive == other_interval.upper_inclusive
            )

        return (
            self.lower == other_interval.lower
            and self.upper == other_interval.upper
            and self.lower_inclusive == other_interval.lower_inclusive
            and self.upper_inclusive == other_interval.upper_inclusive
        )

    def __le__(self, other: object, /) -> bool:
        match other:
            case RealDomain():
                return any(self <= interval for interval in other.intervals)
            case str():
                return self <= RealDomain.from_string(other)
            case _ if (other_interval := self.parse(other)) is not None:
                lower_ok = self.lower > other_interval.lower or (
                    self.lower == other_interval.lower
                    and (other_interval.lower_inclusive or not self.lower_inclusive)
                )
                upper_ok = self.upper < other_interval.upper or (
                    self.upper == other_interval.upper
                    and (other_interval.upper_inclusive or not self.upper_inclusive)
                )
                return lower_ok and upper_ok
            case _:
                return NotImplemented

    def __lt__(self, other: object, /) -> bool:
        result = self <= other
        if result is NotImplemented:
            return NotImplemented
        return result and self != other

    def __ge__(self, other: object, /) -> bool:
        match other:
            case RealDomain():
                return all(interval <= self for interval in other.intervals)
            case str():
                return self >= RealDomain.from_string(other)
            case _ if (other_interval := self.parse(other)) is not None:
                return other_interval <= self
            case _:
                return NotImplemented

    def __or__(self, other: object, /) -> Interval | RealDomain:
        if isinstance(other, Interval) and not self.isdisjoint(other):
            lower = self.lower
            lower_inclusive = self.lower_inclusive
            if other.lower < lower or (
                other.lower == lower and other.lower_inclusive and not lower_inclusive
            ):
                lower = other.lower
                lower_inclusive = other.lower_inclusive

            upper = self.upper
            upper_inclusive = self.upper_inclusive
            if other.upper > upper or (
                other.upper == upper and other.upper_inclusive and not upper_inclusive
            ):
                upper = other.upper
                upper_inclusive = other.upper_inclusive

            return Interval(
                lower,
                upper,
                lower_inclusive=lower_inclusive,
                upper_inclusive=upper_inclusive,
            )

        if (other_union := RealDomain._coerce_union(other)) is None:
            return NotImplemented
        return RealDomain(self, *other_union.intervals)

    def __ror__(self, other: object, /) -> Interval | RealDomain:
        if isinstance(other, Interval) and not self.isdisjoint(other):
            return other | self

        if (other_union := RealDomain._coerce_union(other)) is None:
            return NotImplemented
        return RealDomain(*other_union.intervals, self)

    def __str__(self) -> str:
        lower_bracket = "[" if self.lower_inclusive else "("
        upper_bracket = "]" if self.upper_inclusive else ")"
        lower = format(self.lower, "g")
        upper = format(self.upper, "g")
        return f"{lower_bracket}{lower}, {upper}{upper_bracket}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"


class RealDomain(Domain, Collection[Interval]):
    r"""We model domains on the extended real line by a finite union of intervals."""

    intervals: Final[tuple[Interval, ...]]

    def __init__(self, *intervals: Interval | str) -> None:
        match intervals:
            case []:
                raise ValueError("Expected at least one interval.")
            case [str(spec)]:
                union = RealDomain.from_string(spec)
                intervals = union.intervals
            case _:
                intervals = self._merge_intervals(Interval(spec) for spec in intervals)

        self.intervals = intervals

    @classmethod
    def from_string(cls, s: str, /) -> RealDomain:
        r"""Create a union of intervals from a `|`-separated string."""
        parts = [part.strip() for part in s.split("|")]
        if any(not part for part in parts):
            raise ValueError(f"Invalid union of intervals string: {s}")
        return RealDomain(*(Interval.from_string(part) for part in parts))

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.intervals)

    @overload
    def __getitem__(self, index: int, /) -> Interval: ...
    @overload
    def __getitem__(self, index: slice, /) -> RealDomain: ...
    def __getitem__(self, index: int | slice, /) -> Interval | RealDomain:
        if isinstance(index, slice):
            return RealDomain(*self.intervals[index])
        return self.intervals[index]

    @staticmethod
    def _coerce_union(other: object, /) -> RealDomain | None:
        match other:
            case RealDomain():
                return other
            case Interval():
                return RealDomain(other)
            case str():
                return RealDomain.from_string(other)
            case _:
                return None

    @staticmethod
    def _merge_intervals(intervals: Iterable[Interval], /) -> tuple[Interval, ...]:
        ordered = sorted(intervals, key=lambda i: (i.lower, not i.lower_inclusive))

        merged: list[Interval] = []
        for interval in ordered:
            if not merged:
                merged.append(interval)
                continue

            current = merged[-1]
            if not RealDomain._touch_or_overlap(current, interval):
                merged.append(interval)
                continue

            merged[-1] = Interval(
                current.lower,
                interval.upper,
                lower_inclusive=current.lower_inclusive,
                upper_inclusive=interval.upper_inclusive,
            )

        return tuple(merged)

    @staticmethod
    def _touch_or_overlap(left: Interval, right: Interval, /) -> bool:
        if right.lower < left.upper:
            return True
        if right.lower > left.upper:
            return False
        return left.upper_inclusive or right.lower_inclusive

    def __contains__(self, item: Tensor, /) -> Tensor:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        mask = self.intervals[0].__contains__(item)
        for interval in self.intervals[1:]:
            mask = mask | interval.__contains__(item)
        return mask

    def __eq__(self, other: object, /) -> bool:
        if (other_union := self._coerce_union(other)) is None:
            return NotImplemented
        return hash(self) == hash(other_union)

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __add__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval + other for interval in self.intervals))

    def __sub__(self, other: float, /) -> RealDomain:
        return self + (-other)

    def __mul__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval * other for interval in self.intervals))

    def __le__(self, other: object, /) -> bool:
        if (other_union := self._coerce_union(other)) is None:
            return NotImplemented

        return all(
            any(interval <= other_interval for other_interval in other_union.intervals)
            for interval in self.intervals
        )

    def __lt__(self, other: object, /) -> bool:
        result = self <= other
        if result is NotImplemented:
            return NotImplemented
        return result and self != other

    def __or__(self, other: object, /) -> RealDomain:
        if (other_union := self._coerce_union(other)) is None:
            return NotImplemented
        return RealDomain(*self.intervals, *other_union.intervals)

    def __ror__(self, other: object, /) -> RealDomain:
        if (other_union := self._coerce_union(other)) is None:
            return NotImplemented
        return RealDomain(*other_union.intervals, *self.intervals)

    def __str__(self) -> str:
        return " | ".join(str(interval) for interval in self.intervals)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"


class ScalarDomains(Enum):
    r"""Enumeration of some scalar domains."""

    ZERO = Interval("[0, 0]")
    ONE = Interval("[1, 1]")
    POS_INF = Interval("[+inf, +inf]")
    NEG_INF = Interval("[-inf, -inf]")

    EXTENDED_LINE = Interval("[-inf, inf]")
    REAL_LINE = Interval("(-inf, inf)")
    POSITIVE_REALS = Interval("(0, inf)")
    NEGATIVE_REALS = Interval("(-inf, 0)")
    NONNEGATIVE_REALS = Interval("[0, inf)")
    NONPOSITIVE_REALS = Interval("(-inf, 0]")
    NONZERO = RealDomain.from_string("(-inf, 0) | (0, +inf)")

    UNIT_INTERVAL = Interval("[0, 1]")
    OPEN_UNIT_INTERVAL = Interval("(0, 1)")
    HALF_OPEN_UNIT_INTERVAL = Interval("[0, 1)")

    UNIT_BALL = Interval("[-1, 1]")
    OPEN_UNIT_BALL = Interval("(-1, 1)")

    @property
    def domain(self) -> Domain:
        return self.value

    def __contains__(self, item: Tensor, /) -> Tensor:
        return self.domain.__contains__(item)

    def __le__(self, other: object, /) -> bool:
        match other:
            case ScalarDomains():
                other_domain = other.domain
            case Interval() | RealDomain():
                other_domain = other
            case _:
                return NotImplemented
        return self.domain <= other_domain

    def __lt__(self, other: object, /) -> bool:
        result = self <= other
        if result is NotImplemented:
            return NotImplemented
        return result and self != other

    def __str__(self) -> str:
        return str(self.value)
