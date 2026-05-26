r"""Scalar domain primitives, including intervals and scalar domain labels."""

__all__ = [
    "Interval",
    "RealDomain",
    "ScalarDomains",
]

import logging
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import isnan, nan
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self, cast, overload

from torch import Tensor

from .base import Indeterminate, PosetEnum, ScalarDomain

__logger__ = logging.getLogger(__name__)


@dataclass(unsafe_hash=True, init=False)
class Interval(ScalarDomain):
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
                return Interval._parse_string(arg)
            case _:
                return None

    @staticmethod
    def _parse_string(s: str, /) -> Interval | None:
        if not (s := s.strip()):
            __logger__.debug("Failed to parse interval string %r: empty string", s)
            return None

        match s[0]:
            case "[":
                lower_inclusive = True
            case "(":
                lower_inclusive = False
            case _:
                __logger__.debug(
                    "Failed to parse interval string %r: invalid lower bracket", s
                )
                return None

        match s[-1]:
            case "]":
                upper_inclusive = True
            case ")":
                upper_inclusive = False
            case _:
                __logger__.debug(
                    "Failed to parse interval string %r: invalid upper bracket", s
                )
                return None

        match s[1:-1].split(","):
            case left, right:
                pass
            case _:
                __logger__.debug(
                    "Failed to parse interval string %r: expected two bounds", s
                )
                return None

        match left.strip():
            case "-∞":
                lower = float("-inf")
            case "∞" | "+∞":
                lower = float("inf")
            case left:
                try:
                    lower = float(left)
                except ValueError:
                    __logger__.debug(
                        "Failed to parse interval string %r: invalid lower bound", s
                    )
                    return None

        match right.strip():
            case "-∞":
                upper = float("-inf")
            case "∞" | "+∞":
                upper = float("inf")
            case right:
                try:
                    upper = float(right)
                except ValueError:
                    __logger__.debug(
                        "Failed to parse interval string %r: invalid upper bound", s
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

    @overload
    def __and__(self, rhs: Interval, /) -> Interval: ...
    @overload
    def __and__(self, rhs: RealDomain | str, /) -> Interval | RealDomain: ...
    def __and__(self, rhs: object, /) -> Interval | RealDomain:
        if (other := Interval.parse(rhs)) is None:
            if (union := RealDomain.parse(rhs)) is None:
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
    def __rand__(self, rhs: RealDomain | str, /) -> Interval | RealDomain: ...
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

        if self.is_disjoint(other):
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


# pyrefly: ignore[read-only]
Interval.EMPTY = Interval(  # pyright: ignore[reportAttributeAccessIssue]
    nan,
    nan,
    lower_inclusive=False,
    upper_inclusive=False,
)


class RealDomain(ScalarDomain, Collection[Interval]):
    r"""A finite union of sorted, simplified intervals on the extended real line.

    Invariants:
        The empty domain is represented canonically as ``(Interval.EMPTY,)``.
        Non-empty domains store pairwise disjoint intervals sorted by lower bound.
        Adjacent intervals may share a boundary point only if both exclude it.
    """

    intervals: Final[tuple[Interval, ...]]

    @classmethod
    def parse(cls, other: object, /) -> RealDomain | None:
        match other:
            case RealDomain():
                return other
            case Interval():
                return RealDomain(other)
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
                return RealDomain(*intervals)
            case _:
                __logger__.debug("Failed to unknown type %r", type(other))
                return None

    def __init__(self, *intervals: Interval | str | RealDomain) -> None:
        if not intervals:
            raise ValueError("Expected at least one interval.")

        flat_intervals: list[Interval] = []
        for item in intervals:
            match item:
                case RealDomain() as domain:
                    flat_intervals.extend(domain)
                case str(spec):
                    if (parsed := self.parse(spec)) is None:
                        raise ValueError(f"Invalid union of intervals string: {spec}")
                    flat_intervals.extend(parsed)
                case Interval() as interval:
                    flat_intervals.append(interval)
                case _:
                    raise TypeError(f"Invalid interval: {item}")

        self.intervals = tuple(self._merge_intervals(flat_intervals))
        self._validate()

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

    def is_empty(self) -> bool:
        all_empty = all(interval.is_empty() for interval in self)
        uses_empty_sentinel = self.intervals == (Interval.EMPTY,)
        assert all_empty == uses_empty_sentinel
        return uses_empty_sentinel

    def is_disjoint(self, arg: RealDomain | Interval | str, /) -> bool:
        other = RealDomain(arg)
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
            yield Interval.EMPTY
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
        result = self[0].check(item)
        for interval in self[1:]:
            result = result | interval.check(item)
        return result

    def __eq__(self, rhs: object, /) -> bool:
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __pos__(self) -> RealDomain:
        return self

    def __neg__(self) -> RealDomain:
        return RealDomain(*(-interval for interval in reversed(self)))

    def __le__(self, rhs: object, /) -> bool:
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented

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
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, rhs: object, /) -> bool:
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented
        return other <= self

    def __gt__(self, rhs: object, /) -> bool:
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented
        return self >= other and self != other

    def __add__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval + other for interval in self))

    def __sub__(self, other: float, /) -> RealDomain:
        return self + (-other)

    def __mul__(self, other: float, /) -> RealDomain:
        return RealDomain(*(interval * other for interval in self))

    def __and__(self, rhs: object, /) -> RealDomain:
        if (other := RealDomain.parse(rhs)) is None:
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

        if not intersections:
            return RealDomain(Interval.EMPTY)
        return RealDomain(*intersections)

    def __rand__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented
        return other_union & self

    def __or__(self, rhs: object, /) -> RealDomain:
        if (other := RealDomain.parse(rhs)) is None:
            return NotImplemented
        return RealDomain(*self._merge_intervals((*self, *other)))

    def __ror__(self, other: object, /) -> RealDomain:
        if (other_union := RealDomain.parse(other)) is None:
            return NotImplemented
        return other_union | self

    def __str__(self) -> str:
        return " | ".join(str(interval) for interval in self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"


class ScalarDomains(ScalarDomain, PosetEnum):
    r"""Enumeration of some scalar domains."""

    ALIASES: ClassVar[Mapping[str, Self]]

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
    NONZERO = RealDomain("(-inf, 0) | (0, +inf)")

    UNIT_INTERVAL = Interval("[0, 1]")
    OPEN_UNIT_INTERVAL = Interval("(0, 1)")
    HALF_OPEN_UNIT_INTERVAL = Interval("[0, 1)")

    UNIT_BALL = Interval("[-1, 1]")
    OPEN_UNIT_BALL = Interval("(-1, 1)")

    @classmethod
    def Interval(cls, arg) -> Interval:  # noqa: N802
        return Interval(arg)

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            if (matched := cls.ALIASES.get(value)) is not None:
                return matched
            if (parsed := Interval.parse(value)) is not None:
                return cls(parsed)
            if (parsed := RealDomain.parse(value)) is not None:
                return cls(parsed)
            return None
        return None

    @property
    def domain(self) -> ScalarDomain:
        return self.value

    def check(self, item: Tensor, /) -> Tensor:
        return self.domain.check(item)

    def __le__(self, other: object, /) -> bool | Indeterminate:
        if isinstance(other, ScalarDomains):
            return self.domain <= other.domain
        return self.domain <= other

    def __lt__(self, other: object, /) -> bool | Indeterminate:
        result = self <= other
        if result is NotImplemented:
            return NotImplemented
        return result and self != other

    def __str__(self) -> str:
        return str(self.value)


S = ScalarDomains  # temporary alias
S.ALIASES = MappingProxyType({
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
