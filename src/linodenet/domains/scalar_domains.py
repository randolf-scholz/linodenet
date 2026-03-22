r"""Scalar domain primitives, including intervals and scalar domain labels."""

__all__ = ["Interval", "RealDomain", "ScalarDomains"]


from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from math import isnan, nan
from typing import ClassVar, Final, Self, overload

from torch import Tensor

from .base import Domain


@dataclass(unsafe_hash=True, init=False)
class Interval(Domain):
    r"""A named tuple representing an interval."""

    EMPTY: ClassVar[Final[Interval]] = ...  # pyright: ignore[reportAssignmentType]

    lower: Final[float]  # type: ignore[misc]
    upper: Final[float]  # type: ignore[misc]
    lower_inclusive: Final[bool]  # type: ignore[misc]
    upper_inclusive: Final[bool]  # type: ignore[misc]

    @overload
    def __new__(cls, s: str | Interval, /) -> Interval: ...
    @overload
    def __new__(
        cls,
        /,
        lower: float,
        upper: float,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Interval: ...
    def __new__(  # type: ignore[misc]  # pyright: ignore[reportInconsistentOverload]
        cls,
        lower_or_interval: str | Interval | float | None = None,
        /,
        lower: float | None = None,
        upper: float | None = None,
        *,
        lower_inclusive: bool | None = None,
        upper_inclusive: bool | None = None,
    ) -> Self | Interval:
        if isinstance(lower_or_interval, Interval | str) and (
            lower is not None
            or upper is not None
            or lower_inclusive is not None
            or upper_inclusive is not None
        ):
            raise TypeError("String interval constructor does not accept bounds.")

        match lower_or_interval:
            case str(s):
                spec = Interval._parse_string(s)
                lower = spec["lower"]
                upper = spec["upper"]
                lower_inclusive = spec["lower_inclusive"]
                upper_inclusive = spec["upper_inclusive"]

            case Interval() as interval:
                lower = interval.lower
                upper = interval.upper
                lower_inclusive = interval.lower_inclusive
                upper_inclusive = interval.upper_inclusive

            case float(value):
                if upper is None and lower is None:
                    raise TypeError("Missing upper bound")
                if upper is None:  # lower.upper given positionally
                    upper = lower
                lower = value

            case None:
                if (
                    lower is None
                    or upper is None
                    or lower_inclusive is None
                    or upper_inclusive is None
                ):
                    raise TypeError("Expected either a string/interval or bounds.")
            case _:
                raise TypeError(f"Unexpected type: {type(lower_or_interval)}")

        assert lower is not None
        assert upper is not None
        assert lower_inclusive is not None
        assert upper_inclusive is not None

        if (isnan(lower) or isnan(upper)) and Interval.EMPTY is not Ellipsis:
            return Interval.EMPTY

        self = super().__new__(cls)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower_inclusive", lower_inclusive)
        object.__setattr__(self, "upper_inclusive", upper_inclusive)
        return self

    @classmethod
    def parse(cls, arg: object, /) -> Interval | None:
        match arg:
            case Interval():
                return arg
            case str():
                try:
                    spec = Interval._parse_string(arg)
                    return Interval(**spec)
                except ValueError:
                    return None
            case _:
                return None

    @staticmethod
    def _parse_string(s: str, /) -> dict:
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

        return {
            "lower": lower,
            "upper": upper,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
        }

    def isdisjoint(self, other: Interval | str, /) -> bool:
        r"""Return whether two intervals have empty intersection."""
        other = Interval(other)
        if self.isempty() or other.isempty():
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

    def isempty(self) -> bool:
        r"""Return whether the interval represents the empty set."""
        return isnan(self.lower) or isnan(self.upper)

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

    def __eq__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union == self

        # Note: safe, because NAN always use math.nan
        #    there are many different binary representations of float(nan)
        #    but Interval.__new__ returns the same Interval.EMPTY instance for any of them
        return hash(self) == hash(other)

    def __le__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union >= self

        lower_ok = self.lower > other.lower or (
            self.lower == other.lower
            and (other.lower_inclusive or not self.lower_inclusive)
        )
        upper_ok = self.upper < other.upper or (
            self.upper == other.upper
            and (other.upper_inclusive or not self.upper_inclusive)
        )
        return lower_ok and upper_ok

    def __lt__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union > self
        return self <= other and self != other

    def __ge__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union <= self
        return other <= self

    def __gt__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union < self
        return self >= other and self != other

    def __and__(self, rhs: object, /) -> Interval | RealDomain:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union & self

        if self.isdisjoint(other):
            return Interval.EMPTY

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

    def __rand__(self, lhs: object, /) -> Interval | RealDomain:
        if (other := Interval.parse(lhs)) is None:
            if (union := RealDomain.parse(lhs)) is None:
                return NotImplemented
            return union & self
        return other & self

    def __or__(self, rhs: object, /) -> Interval | RealDomain:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
                return NotImplemented
            return union | self

        if self.isdisjoint(other):
            return RealDomain(self, other)

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

    def __ror__(self, lhs: object, /) -> Interval | RealDomain:
        if (other := Interval.parse(lhs)) is None:
            if (union := RealDomain.parse(lhs)) is None:
                return NotImplemented
            return union | self
        return other | self

    def __str__(self) -> str:
        lower_bracket = "[" if self.lower_inclusive else "("
        upper_bracket = "]" if self.upper_inclusive else ")"
        lower = format(self.lower, "g")
        upper = format(self.upper, "g")
        return f"{lower_bracket}{lower}, {upper}{upper_bracket}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"


Interval.EMPTY = Interval(  # pyright: ignore[reportAttributeAccessIssue]
    nan,
    nan,
    lower_inclusive=False,
    upper_inclusive=False,
)


class RealDomain(Domain, Collection[Interval]):
    r"""We model domains on the extended real line by a finite union of intervals."""

    intervals: Final[tuple[Interval, ...]]

    @classmethod
    def parse(cls, other: object, /) -> RealDomain | None:
        match other:
            case RealDomain():
                return other
            case Interval():
                return RealDomain(other)
            case str():
                try:
                    return RealDomain.from_string(other)
                except ValueError:
                    return None
            case _:
                return None

    @classmethod
    def from_string(cls, s: str, /) -> RealDomain:
        r"""Create a union of intervals from a `|`-separated string."""
        parts = [part.strip() for part in s.split("|")]
        if any(not part for part in parts):
            raise ValueError(f"Invalid union of intervals string: {s}")
        return RealDomain(*(Interval(part) for part in parts))

    def __init__(self, *intervals: Interval | str | RealDomain) -> None:
        if not intervals:
            raise ValueError("Expected at least one interval.")

        flat_intervals: list[Interval] = []
        for item in intervals:
            match item:
                case RealDomain() as domain:
                    flat_intervals.extend(domain.intervals)
                case str(spec):
                    parsed = RealDomain.from_string(spec)
                    flat_intervals.extend(parsed.intervals)
                case Interval() as interval:
                    flat_intervals.append(interval)
                case _:
                    raise TypeError(f"Invalid interval: {item}")

        self.intervals = self._merge_intervals(flat_intervals)

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

    def isempty(self) -> bool:
        return all(interval.isempty() for interval in self.intervals)

    @staticmethod
    def _merge_intervals(intervals: Iterable[Interval], /) -> tuple[Interval, ...]:
        if not (intervals := [i for i in intervals if not i.isempty()]):
            return (Interval("(NAN, NAN)"),)

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
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented
        return self.intervals == other_union.intervals

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __add__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval + other for interval in self.intervals))

    def __sub__(self, other: float, /) -> RealDomain:
        return self + (-other)

    def __mul__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval * other for interval in self.intervals))

    def __and__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented

        intersections = [
            left & right
            for left in self.intervals
            for right in other_union.intervals
            if not (left & right).isempty()
        ]
        if not intersections:
            return RealDomain(Interval("(NAN, NAN)"))
        return RealDomain(*intersections)

    def __rand__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented
        return other_union & self

    def __le__(self, other: object, /) -> bool:
        if (other_union := RealDomain.parse(other)) is None:
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

    def __ge__(self, other: object, /) -> bool:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented

        return all(
            any(interval <= self_interval for self_interval in self.intervals)
            for interval in other_union.intervals
        )

    def __gt__(self, other: object, /) -> bool:
        result = self >= other
        if result is NotImplemented:
            return NotImplemented
        return result and self != other

    def __or__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented
        return RealDomain(*self.intervals, *other_union.intervals)

    def __ror__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
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
    NONZERO = RealDomain("(-inf, 0) | (0, +inf)")

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
