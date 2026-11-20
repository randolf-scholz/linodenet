r"""Scalar domain primitives, including intervals and scalar domain labels."""

__all__ = [
    "Interval",
    "IntervalUnion",
    "RealDomain",
    "RealDomains",
]

import logging
from abc import abstractmethod
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import inf, isnan, nan
from typing import Any, ClassVar, Final, Self, cast, overload

import torch
from torch import Tensor

from .base import PosetEnum, ScalarDomain

__logger__ = logging.getLogger(__name__)


class RealDomain(ScalarDomain):
    r"""A floating point domain with decidable subset relationships.

    Every real domain exposes its finite interval decomposition through
    `intervals`. This representation makes subset relationships decidable:
    comparisons between real domains return a `bool`, rather than a symbolic
    or indeterminate result. Finite unions and intersections are real domains
    as well.

    Note:
        The interval decomposition follows these conventions:

        - The empty set is represented by the empty tuple `()`.
        - Nonempty domains contain nonempty, pairwise disjoint intervals sorted
          by lower bound.
        - Components are maximal: adjacent intervals are merged whenever their
          union is an interval. A shared endpoint must be excluded by both.
        - Bounds may include `-∞` and `+∞`. NaN is not a domain element;
          `Interval.EMPTY` uses NaN bounds only as an empty-interval sentinel.

        An `Interval` exposes itself as a singleton tuple, or `()` if empty.
        An `IntervalUnion` exposes its stored components, and `RealDomains`
        delegates to the underlying domain. Use `IntervalUnion(domain)` when
        a concrete union object is needed.
    """

    @property
    @abstractmethod
    def intervals(self) -> tuple[Interval, ...]:
        r"""Return the normalized interval components, or `()` for the empty set."""
        ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def __le__(self, other: RealDomain, /) -> bool: ...

    @abstractmethod
    def __and__(self, other: RealDomain, /) -> RealDomain: ...

    @abstractmethod
    def __or__(self, other: RealDomain, /) -> RealDomain: ...


@dataclass(unsafe_hash=True, init=False)
class Interval(RealDomain):
    r"""A named tuple representing an interval."""

    EMPTY: ClassVar[Final[Interval]] = cast("Any", ...)

    lower: Final[float]
    upper: Final[float]
    lower_inclusive: Final[bool]
    upper_inclusive: Final[bool]

    @overload
    def __new__(cls, s: str | Interval, /) -> Interval: ...
    @overload
    def __new__(  # pyrefly: ignore[inconsistent-overload]
        cls,
        /,
        lower: float,
        upper: float,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Interval: ...
    def __new__(  # pyright: ignore[reportInconsistentOverload]
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
                if (interval := Interval.parse(s)) is None:
                    __logger__.debug("Failed to parse interval string %r", s)
                    raise ValueError(f"Invalid interval string: {s}")
                return interval

            case Interval() as interval:
                lower = interval.lower
                upper = interval.upper
                lower_inclusive = interval.lower_inclusive
                upper_inclusive = interval.upper_inclusive

            case float(value) | int(value):
                if upper is None and lower is None:
                    raise TypeError("Missing upper bound")
                if upper is None:  # lower.upper given positionally
                    upper = lower
                lower = value

                if lower_inclusive is None:
                    raise TypeError("Missing argument: lower_inclusive (bool)")
                if upper_inclusive is None:
                    raise TypeError("Missing argument: upper_inclusive (bool)")

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
                return Interval._parse_string(arg)
            case _:
                return None

    @staticmethod
    def _parse_string(s: str, /) -> Interval | None:
        if not (s := s.strip()):
            __logger__.debug(f"Failed to parse interval string {s!r}: empty string")
            return None

        match s[0]:
            case "[":
                lower_inclusive = True
            case "(":
                lower_inclusive = False
            case _:
                __logger__.debug(
                    f"Failed to parse interval string {s!r}: invalid lower bracket"
                )
                return None

        match s[-1]:
            case "]":
                upper_inclusive = True
            case ")":
                upper_inclusive = False
            case _:
                __logger__.debug(
                    f"Failed to parse interval string {s!r}: invalid upper bracket"
                )
                return None

        match s[1:-1].split(","):
            case left, right:
                pass
            case _:
                __logger__.debug(
                    f"Failed to parse interval string {s!r}: expected two bounds"
                )
                return None

        match left.strip():
            case "-∞":
                lower = -inf
            case "∞" | "+∞":
                lower = inf
            case left:
                try:
                    lower = float(left)
                except ValueError:
                    __logger__.debug(
                        f"Failed to parse interval string {s!r}: invalid lower bound"
                    )
                    return None

        match right.strip():
            case "-∞":
                upper = -inf
            case "∞" | "+∞":
                upper = inf
            case right:
                try:
                    upper = float(right)
                except ValueError:
                    __logger__.debug(
                        f"Failed to parse interval string {s!r}: invalid upper bound"
                    )
                    return None

        return Interval(
            lower=lower,
            upper=upper,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )

    def is_disjoint(self, arg: Interval | str, /) -> bool:
        r"""Return whether two intervals have empty intersection."""
        other = Interval(arg)
        if self.is_empty() or other.is_empty():
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

    def touches(self, arg: Interval | str, /) -> bool:
        r"""Return whether two intervals meet at an included shared endpoint."""
        other = Interval(arg)
        if self.is_empty() or other.is_empty():
            return False
        return (
            self.upper == other.lower and self.upper_inclusive and other.lower_inclusive
        ) or (
            other.upper == self.lower and other.upper_inclusive and self.lower_inclusive
        )

    def is_empty(self) -> bool:
        r"""Return whether the interval represents the empty set."""
        non_empty = not (isnan(self.lower) or isnan(self.upper))
        is_empty_sentinel = self is Interval.EMPTY
        assert non_empty ^ is_empty_sentinel  # safety check.
        return is_empty_sentinel

    @property
    def intervals(self) -> tuple[Interval, ...]:
        return () if self.is_empty() else (self,)

    def check(self, item: Tensor, /) -> Tensor:
        lower_mask = (
            (item >= self.lower) if self.lower_inclusive else (item > self.lower)
        )
        upper_mask = (
            (item <= self.upper) if self.upper_inclusive else (item < self.upper)
        )
        return lower_mask & upper_mask

    def __pos__(self) -> Interval:
        return self

    def __neg__(self) -> Interval:
        return Interval(
            -self.upper,
            -self.lower,
            lower_inclusive=self.upper_inclusive,
            upper_inclusive=self.lower_inclusive,
        )

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
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union == self

        # Note: safe, because NAN always use math.nan
        #    there are many different binary representations of float(nan)
        #    but Interval.__new__ returns the same Interval.EMPTY instance for any of them
        return hash(self) == hash(other)

    def __le__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union >= self

        if self.is_empty():
            return True
        if other.is_empty():
            return False

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
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union > self
        return self <= other and self != other

    def __ge__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union <= self
        return other <= self

    def __gt__(self, rhs: object, /) -> bool:
        if (other := Interval.parse(rhs)) is None:
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union < self
        return self >= other and self != other

    @overload
    def __and__(self, rhs: Interval, /) -> Interval: ...
    @overload
    def __and__(self, rhs: RealDomain | str, /) -> Interval | IntervalUnion: ...
    def __and__(self, rhs: object, /) -> Interval | IntervalUnion:
        if (other := Interval.parse(rhs)) is None:
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union & self

        if self.is_disjoint(other):
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

    @overload
    def __rand__(self, rhs: Interval, /) -> Interval: ...
    @overload
    def __rand__(self, rhs: RealDomain | str, /) -> Interval | IntervalUnion: ...
    def __rand__(self, lhs: object, /) -> Interval | IntervalUnion:
        if (other := Interval.parse(lhs)) is None:
            if (union := IntervalUnion.parse(lhs)) is None:
                return NotImplemented
            return union & self
        return other & self

    def __or__(self, rhs: object, /) -> Interval | IntervalUnion:
        if (other := Interval.parse(rhs)) is None:
            if (union := IntervalUnion.parse(rhs)) is None:
                return NotImplemented
            return union | self

        if self.is_disjoint(other):
            return IntervalUnion(self, other)

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

    def __ror__(self, lhs: object, /) -> Interval | IntervalUnion:
        if (other := Interval.parse(lhs)) is None:
            if (union := IntervalUnion.parse(lhs)) is None:
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


# pyrefly: ignore[read-only]
Interval.EMPTY = Interval(  # pyright: ignore[reportAttributeAccessIssue]
    nan,
    nan,
    lower_inclusive=False,
    upper_inclusive=False,
)


class IntervalUnion(RealDomain, Collection[Interval]):
    r"""A normalized finite interval union following the `RealDomain` conventions.

    With no arguments, constructs the empty domain.
    """

    _intervals: Final[tuple[Interval, ...]]

    @classmethod
    def parse(cls, other: object, /) -> IntervalUnion | None:
        match other:
            case IntervalUnion():
                return other
            case RealDomain():
                return IntervalUnion(other)
            case str(string):
                intervals: list[Interval] = []
                for part in string.split("|"):
                    if (interval := Interval.parse(part)) is None:
                        __logger__.debug(
                            "Failed to parse real domain string %r: invalid interval %r",
                            string,
                            part,
                        )
                        return None
                    intervals.append(interval)
                return IntervalUnion(*intervals)
            case _:
                __logger__.debug("Failed to unknown type %r", type(other))
                return None

    def __init__(self, *intervals: RealDomain | str) -> None:
        flat_intervals: list[Interval] = []
        for item in intervals:
            match item:
                case RealDomain():
                    flat_intervals.extend(item.intervals)
                case str(spec):
                    if (parsed := self.parse(spec)) is None:
                        raise ValueError(f"Invalid union of intervals string: {spec}")
                    flat_intervals.extend(parsed)
                case _:
                    raise TypeError(f"Invalid interval: {item}")

        self._intervals = tuple(self._merge_intervals(flat_intervals))
        self._validate()

    @property
    def intervals(self) -> tuple[Interval, ...]:
        return self._intervals

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.intervals)

    @overload
    def __getitem__(self, index: int, /) -> Interval: ...
    @overload
    def __getitem__(self, index: slice, /) -> IntervalUnion: ...
    def __getitem__(self, index: int | slice, /) -> Interval | IntervalUnion:
        if isinstance(index, slice):
            return IntervalUnion(*self.intervals[index])
        return self.intervals[index]

    def is_empty(self) -> bool:
        return not self.intervals

    def is_disjoint(self, arg: RealDomain | str, /) -> bool:
        other = IntervalUnion(arg)
        if self.is_empty() or other.is_empty():
            return True

        left_intervals = iter(self)
        right_intervals = iter(other)
        left = next(left_intervals, None)
        right = next(right_intervals, None)
        while left is not None and right is not None:
            if not left.is_disjoint(right):
                return False

            if left.upper < right.upper:
                left = next(left_intervals, None)
            elif right.upper < left.upper or (
                left.upper_inclusive and not right.upper_inclusive
            ):
                right = next(right_intervals, None)
            else:
                left = next(left_intervals, None)
        return True

    def _validate(self) -> None:
        if self.is_empty():
            return

        for interval in self:
            if interval.is_empty():
                raise ValueError("Non-empty domains cannot contain empty intervals.")
            if interval.lower > interval.upper:
                raise ValueError("Intervals must satisfy lower <= upper.")

        for left, right in zip(self.intervals, self.intervals[1:], strict=False):
            if not left.is_disjoint(right):
                raise ValueError("RealDomain intervals must be pairwise disjoint.")
            if left.upper > right.lower:
                raise ValueError("RealDomain intervals must be sorted by lower bound.")

    @staticmethod
    def _merge_intervals(intervals: Iterable[Interval], /) -> Iterator[Interval]:
        if not (intervals := [i for i in intervals if not i.is_empty()]):
            return

        ordered = sorted(intervals, key=lambda i: (i.lower, not i.lower_inclusive))

        current = ordered[0]
        for interval in ordered[1:]:
            # Emit the current interval once there is a strict gap.
            if current.upper < interval.lower or (
                current.upper == interval.lower
                and not (current.upper_inclusive or interval.lower_inclusive)
            ):
                yield current
                current = interval
                continue

            # The new interval is already covered by the current hull.
            if current.upper > interval.upper:
                continue
            if current.upper < interval.upper:
                # Extend the current hull to the right.
                current = Interval(
                    current.lower,
                    interval.upper,
                    lower_inclusive=current.lower_inclusive,
                    upper_inclusive=interval.upper_inclusive,
                )
                continue

            # Equal upper bounds merge inclusivity at the shared endpoint.
            current = Interval(
                current.lower,
                current.upper,
                lower_inclusive=current.lower_inclusive,
                upper_inclusive=current.upper_inclusive or interval.upper_inclusive,
            )

        yield current

    def check(self, item: Tensor, /) -> Tensor:
        result = torch.zeros_like(item, dtype=torch.bool)
        for interval in self:
            result = result | interval.check(item)
        return result

    def __eq__(self, rhs: object, /) -> bool:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __pos__(self) -> IntervalUnion:
        return self

    def __neg__(self) -> IntervalUnion:
        return IntervalUnion(*(-interval for interval in reversed(self)))

    def __le__(self, rhs: object, /) -> bool:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented

        if self.is_empty():
            return True
        if other.is_empty():
            return False

        right_intervals = iter(other)
        right = next(right_intervals, None)
        for left in self:
            while (
                right is not None
                and right.upper <= left.lower
                and not right.touches(left)
            ):
                right = next(right_intervals, None)

            if right is None or not (left <= right):
                return False

        return True

    def __lt__(self, rhs: object, /) -> bool:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, rhs: object, /) -> bool:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented
        return other <= self

    def __gt__(self, rhs: object, /) -> bool:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented
        return self >= other and self != other

    def __add__(self, other: float, /) -> IntervalUnion:
        return IntervalUnion(*(interval + other for interval in self))

    def __sub__(self, other: float, /) -> IntervalUnion:
        return self + (-other)

    def __mul__(self, other: float, /) -> IntervalUnion:
        return IntervalUnion(*(interval * other for interval in self))

    def __and__(self, rhs: object, /) -> IntervalUnion:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented

        left_intervals = iter(self)
        right_intervals = iter(other)
        left = next(left_intervals, None)
        right = next(right_intervals, None)
        intersections: list[Interval] = []
        while left is not None and right is not None:
            intersection = left & right
            if not intersection.is_empty():
                intersections.append(intersection)

            if left.upper < right.upper:
                left = next(left_intervals, None)
            elif right.upper < left.upper or (
                left.upper_inclusive and not right.upper_inclusive
            ):
                right = next(right_intervals, None)
            else:
                left = next(left_intervals, None)

        return IntervalUnion(*intersections)

    def __rand__(self, other: object, /) -> IntervalUnion:
        if (other_union := IntervalUnion.parse(other)) is None:
            return NotImplemented
        return other_union & self

    def __or__(self, rhs: object, /) -> IntervalUnion:
        if (other := IntervalUnion.parse(rhs)) is None:
            return NotImplemented
        return IntervalUnion(*self._merge_intervals((*self, *other)))

    def __ror__(self, other: object, /) -> IntervalUnion:
        if (other_union := IntervalUnion.parse(other)) is None:
            return NotImplemented
        return other_union | self

    def __str__(self) -> str:
        return " | ".join(str(interval) for interval in self) or str(Interval.EMPTY)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"


class RealDomains(RealDomain, PosetEnum):
    r"""Enumeration of some real domains."""

    _value_: RealDomain
    ALIASES: ClassVar[Mapping[str, Self]]
    Interval: ClassVar[type[Interval]]

    ANY = Interval("[-inf, +inf]")
    EMPTY = Interval.EMPTY

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
    NONZERO = IntervalUnion("(-inf, 0) | (0, +inf)")

    UNIT_INTERVAL = Interval("[0, 1]")
    OPEN_UNIT_INTERVAL = Interval("(0, 1)")
    HALF_OPEN_UNIT_INTERVAL = Interval("[0, 1)")

    UNIT_BALL = Interval("[-1, 1]")
    OPEN_UNIT_BALL = Interval("(-1, 1)")

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            if (matched := cls.ALIASES.get(value)) is not None:
                return matched
            if (parsed := Interval.parse(value)) is not None:
                return cls(parsed)
            if (parsed := IntervalUnion.parse(value)) is not None:
                return cls(parsed)
            return None
        return None

    @property
    def domain(self) -> RealDomain:
        return self.value

    def check(self, item: Tensor, /) -> Tensor:
        return self.domain.check(item)

    def is_empty(self) -> bool:
        return self.domain.is_empty()

    @property
    def intervals(self) -> tuple[Interval, ...]:
        return self.domain.intervals

    def __le__(self, other: object, /) -> bool:
        return IntervalUnion(self).__le__(other)

    def __lt__(self, other: object, /) -> bool:
        return IntervalUnion(self).__lt__(other)

    def __ge__(self, other: object, /) -> bool:
        return IntervalUnion(self).__ge__(other)

    def __gt__(self, other: object, /) -> bool:
        return IntervalUnion(self).__gt__(other)

    def __and__(self, other: object, /) -> IntervalUnion:
        return IntervalUnion(self).__and__(other)

    def __rand__(self, other: object, /) -> IntervalUnion:
        return IntervalUnion(self).__rand__(other)

    def __or__(self, other: object, /) -> IntervalUnion:
        return IntervalUnion(self).__or__(other)

    def __ror__(self, other: object, /) -> IntervalUnion:
        return IntervalUnion(self).__ror__(other)

    def __str__(self) -> str:
        return str(self.value)


RealDomains.Interval = Interval


S = RealDomains  # temporary alias
S.ALIASES = frozendict({
    "any"                     : S.ANY,
    "empty"                   : S.EMPTY,
    "extended-line"           : S.EXTENDED_LINE,
    "half-open-unit-interval" : S.HALF_OPEN_UNIT_INTERVAL,
    "neg-inf"                 : S.NEG_INF,
    "negative-infinity"       : S.NEG_INF,
    "negative-reals"          : S.NEGATIVE_REALS,
    "nonnegative-reals"       : S.NONNEGATIVE_REALS,
    "nonpositive-reals"       : S.NONPOSITIVE_REALS,
    "nonzero"                 : S.NONZERO,
    "one"                     : S.ONE,
    "open-unit-ball"          : S.OPEN_UNIT_BALL,
    "open-unit-interval"      : S.OPEN_UNIT_INTERVAL,
    "pos-inf"                 : S.POS_INF,
    "positive-infinity"       : S.POS_INF,
    "positive-reals"          : S.POSITIVE_REALS,
    "real-line"               : S.REAL_LINE,
    "unit-ball"               : S.UNIT_BALL,
    "unit-interval"           : S.UNIT_INTERVAL,
    "zero"                    : S.ZERO,
})  # fmt: skip
del S  # remove alias
