r"""Test inheritance from protocol types."""

from abc import abstractmethod
from typing import Final, Protocol

import torch
from torch import Tensor, nn


def test_inherit_protocol_init() -> None:

    class Filter(Protocol):
        input_size: Final[int]  # type: ignore[misc]
        hidden_size: Final[int]  # type: ignore[misc]

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)

    class FilterClass(Filter):
        pass

    instance = FilterClass(3, 4)
    assert instance.input_size == 3
    assert instance.hidden_size == 4


def test_inherit_protocol_torch() -> None:

    class Filter(Protocol):
        input_size: Final[int]  # type: ignore[misc]
        hidden_size: Final[int]  # type: ignore[misc]

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)

    class FilterModule(Filter, nn.Module):
        @abstractmethod
        def forward(self, y: Tensor, x: Tensor) -> Tensor: ...

    class ConcreteFilter(FilterModule):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__(input_size, hidden_size)
            self.encoder = nn.Linear(input_size, hidden_size)

        def forward(self, y: Tensor, x: Tensor) -> Tensor:
            return self.encoder(y) + x

    m = 3
    n = 4
    batch_size = 5
    instance = ConcreteFilter(m, n)
    assert instance.input_size == m
    assert instance.hidden_size == n
    y = torch.randn(batch_size, m)
    x = torch.randn(batch_size, n)
    r = instance(y, x)
    assert r.shape == (batch_size, n)
