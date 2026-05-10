r"""Complex scalar domains."""

__all__ = [
    "ComplexDomain",
    # Classes
    "ClosedDisk",
    "ClosedHorizontalStrip",
    "ClosedLeftPlane",
    "ClosedLowerPlane",
    "ClosedRightPlane",
    "ClosedUnitDisk",
    "ClosedUpperPlane",
    "ClosedVerticalStrip",
    "HorizontalLine",
    "ImaginaryAxis",
    "OpenDisk",
    "OpenHorizontalStrip",
    "OpenLeftPlane",
    "OpenLowerPlane",
    "OpenRightPlane",
    "OpenUnitDisk",
    "OpenUpperPlane",
    "OpenVerticalStrip",
    "RealAxis",
    "UnitCircle",
    "VerticalLine",
]


from dataclasses import dataclass
from typing import Final

from torch import Tensor

from .base import ScalarDomain


class ComplexDomain(ScalarDomain):
    r"""Base class for complex scalar domains."""


@dataclass(frozen=True)
class OpenRightPlane(ComplexDomain):
    r"""The complex region with $Re(z) > 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.real > 0


@dataclass(frozen=True)
class OpenLeftPlane(ComplexDomain):
    r"""The complex region with $Re(z) < 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.real < 0


@dataclass(frozen=True)
class OpenUpperPlane(ComplexDomain):
    r"""The complex region with $Im(z) > 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag > 0


@dataclass(frozen=True)
class OpenLowerPlane(ComplexDomain):
    r"""The complex region with $Im(z) < 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag < 0


@dataclass(frozen=True)
class ClosedRightPlane(ComplexDomain):
    r"""The complex region with $Re(z) ≥ 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.real >= 0


@dataclass(frozen=True)
class ClosedLeftPlane(ComplexDomain):
    r"""The complex region with $Re(z) ≤ 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.real <= 0


@dataclass(frozen=True)
class ClosedUpperPlane(ComplexDomain):
    r"""The complex region with $Im(z) ≥ 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag >= 0


@dataclass(frozen=True)
class ClosedLowerPlane(ComplexDomain):
    r"""The complex region with $Im(z) ≤ 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag <= 0


@dataclass(frozen=True)
class OpenDisk(ComplexDomain):
    r"""The complex region with $|z| < r$."""

    radius: Final[float]

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("Open disks require a non-negative radius.")

    def check(self, value: Tensor, /) -> Tensor:
        return value.abs() < self.radius


@dataclass(frozen=True)
class ClosedDisk(ComplexDomain):
    r"""The complex region with $|z| ≤ r$."""

    radius: Final[float]

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("Closed disks require a non-negative radius.")

    def check(self, value: Tensor, /) -> Tensor:
        return value.abs() <= self.radius


@dataclass(frozen=True)
class OpenUnitDisk(ComplexDomain):
    r"""The complex region with $|z| < 1$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.abs() < 1


@dataclass(frozen=True)
class ClosedUnitDisk(ComplexDomain):
    r"""The complex region with $|z| ≤ 1$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.abs() <= 1


@dataclass(frozen=True)
class UnitCircle(ComplexDomain):
    r"""The complex region with $|z| = 1$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.abs() == 1


@dataclass(frozen=True)
class OpenVerticalStrip(ComplexDomain):
    r"""The complex region with $Re(z) ∈ (a,b)$."""

    lower: Final[float]
    upper: Final[float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Expected lower <= upper.")

    def check(self, value: Tensor, /) -> Tensor:
        return (self.lower < value.real) & (value.real < self.upper)


@dataclass(frozen=True)
class ClosedVerticalStrip(ComplexDomain):
    r"""The complex region with $Re(z) ∈ [a,b]$."""

    lower: Final[float]
    upper: Final[float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Expected lower <= upper.")

    def check(self, value: Tensor, /) -> Tensor:
        return (self.lower <= value.real) & (value.real <= self.upper)


@dataclass(frozen=True)
class OpenHorizontalStrip(ComplexDomain):
    r"""The complex region with $Im(z) ∈ (c, d)$."""

    lower: Final[float]
    upper: Final[float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Expected lower <= upper.")

    def check(self, value: Tensor, /) -> Tensor:
        imag = (value + 0.0j).imag
        return (self.lower < imag) & (imag < self.upper)


@dataclass(frozen=True)
class ClosedHorizontalStrip(ComplexDomain):
    r"""The complex region with $Im(z) ∈ [c,d]$."""

    lower: Final[float]
    upper: Final[float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Expected lower <= upper.")

    def check(self, value: Tensor, /) -> Tensor:
        imag = (value + 0.0j).imag
        return (self.lower <= imag) & (imag <= self.upper)


@dataclass(frozen=True)
class HorizontalLine(ComplexDomain):
    r"""The complex region with $Im(z) = c$."""

    value: Final[float]

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag == self.value


@dataclass(frozen=True)
class VerticalLine(ComplexDomain):
    r"""The complex region with $Re(z) = c$."""

    value: Final[float]

    def check(self, value: Tensor, /) -> Tensor:
        return value.real == self.value


@dataclass(frozen=True)
class ImaginaryAxis(ComplexDomain):
    r"""The complex region with $Re(z) = 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.real == 0


@dataclass(frozen=True)
class RealAxis(ComplexDomain):
    r"""The complex region with $Im(z) = 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return (value + 0.0j).imag == 0
