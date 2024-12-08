r"""Abstract classes for distributions."""

__all__ = [
    "DistributionProto",
    "ConditionalDistributionProto",
    "JointDistributionProto",
]

from typing import Protocol


class DistributionProto(Protocol):
    r"""A protocol for distributions."""


class ConditionalDistributionProto(Protocol):
    r"""A protocol for conditional distributions."""


class JointDistributionProto(Protocol):
    r"""A protocol for joint distributions."""
