r"""Distributions base class."""

__all__ = [
    # ABCs & Protocols
    "Distribution",
    "Marginalizable",
    # Classes
    "DistributionBase",
    "DistributionList",
    "DistributionDict",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Size, Tensor, jit, nn

from linodenet.containers import ModuleMapping, ModuleSequence


@runtime_checkable
class Distribution[X](Protocol):
    r"""A protocol for distributions, compatible with `torch.distributions.Distribution`."""

    # @property
    # def batch_shape(self) -> tuple[int, ...]: ...
    # @property
    # def event_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def sample(self, num: int = 1, /) -> X: ...
    @abstractmethod
    def log_prob(self, value: X, /) -> Tensor: ...


class DistributionList[X, D: DistributionBase](ModuleSequence[D]):
    r"""A list of distributions, similar to `nn.ModuleList`."""

    samples: Tensor
    log_probs: Tensor

    def __init__(self, modules: Iterable[D], /) -> None:
        super().__init__(modules)
        self.register_buffer("samples", torch.empty(()))
        self.register_buffer("log_probs", torch.empty(()))

    @abstractmethod
    def sample(self, num: int = 1, /) -> X: ...
    @abstractmethod
    def log_prob(self, value: X, /) -> Tensor: ...


class DistributionDict[X, D: DistributionBase](ModuleMapping[D]):
    r"""A dictionary of distributions, similar to `nn.ModuleDict`."""

    samples: Tensor
    log_probs: Tensor

    def __init__(self, modules: Mapping[str, D], /) -> None:
        super().__init__(modules)
        self.register_buffer("samples", torch.empty(()))
        self.register_buffer("log_probs", torch.empty(()))

    @abstractmethod
    def sample(self, num: int = 1, /) -> X: ...
    @abstractmethod
    def log_prob(self, value: X, /) -> Tensor: ...


class Marginalizable[X](Distribution[X], Protocol):
    r"""A protocol for marginalizable distributions."""

    @abstractmethod
    def marginalize(self, x: X, /, *, dims: tuple[int, ...]) -> Distribution[X]: ...


class DistributionBase(nn.Module):
    r"""Base class for distributions."""

    samples: Tensor
    log_probs: Tensor
    event_shape: Final[tuple[int, ...]]
    r"""CONST: The shape of a single sample."""
    batch_shape: Final[tuple[int, ...]]
    r"""CONST: builtin batch shape (for multi-head distributions)."""

    def __init__(
        self,
        *,
        event_shape: Sequence[int],
        batch_shape: Sequence[int],
    ) -> None:
        super().__init__()
        self.event_shape = Size(event_shape)
        self.batch_shape = Size(batch_shape)
        self.register_buffer("samples", torch.empty(()))
        self.register_buffer("log_probs", torch.empty(()))

    @abstractmethod
    def sample(self, num: int = 1, /) -> Tensor: ...
    @abstractmethod
    def log_prob(self, value: Tensor, /) -> Tensor: ...

    @jit.export
    def sample_and_log_prob(self, num: int = 1, /) -> tuple[Tensor, Tensor]:
        samples = self.sample(num)
        log_probs = self.log_prob(samples)
        self.samples = samples
        self.log_probs = log_probs
        return samples, log_probs
