r"""Test inheritance from protocol types."""

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from tempfile import TemporaryFile
from typing import Final, Protocol, cast

import pytest
import torch
from torch import Tensor, jit, nn


class Filter(Protocol):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)


def test_inherit_protocol_init() -> None:
    class FilterClass(Filter):
        pass

    instance = FilterClass(3, 4)
    assert instance.input_size == 3
    assert instance.hidden_size == 4


def test_inherit_protocol_torch() -> None:
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

    # test torch.jit.script
    scripted_instance = jit.script(instance)
    with TemporaryFile() as f:
        torch.jit.save(scripted_instance, f)
        f.seek(0)
        loaded_instance = torch.jit.load(f)

    assert loaded_instance.input_size == m
    assert loaded_instance.hidden_size == n
    assert loaded_instance(y, x).shape == (batch_size, n)


def test_multiple_inheritance_fails_with_bad_order() -> None:
    class FilterModule(Filter, nn.Module):  # <- only works with this order!
        pass

    # pyrefly: ignore[inconsistent-inheritance]
    class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
        pass

    with pytest.raises(TypeError, match="Cannot create a consistent method resolution"):

        class _(ModuleSequence[nn.Module], FilterModule): ...  # type: ignore  # noqa: PGH003


def test_multiple_inheritance() -> None:
    class FilterModule(nn.Module, Filter):  # <- only works with this order!
        def __init__(self, input_size: int, hidden_size: int) -> None:
            nn.Module.__init__(self)
            Filter.__init__(self, input_size, hidden_size)

        @abstractmethod
        def forward(self, y: Tensor, x: Tensor) -> Tensor: ...

    # pyrefly: ignore[inconsistent-inheritance]
    class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
        pass

    # NOTE: init only works with this order!
    class FilterSequence(FilterModule, ModuleSequence[nn.Module]):
        def __init__(
            self,
            modules: Iterable[Filter] = (),
            *,
            input_size: int,
            hidden_size: int,
        ) -> None:
            nn.ModuleList.__init__(self, cast("Iterable[nn.Module]", modules))
            # ⚠️ need to call Filter.__init__, not FilterModule.__init__ ⚠️
            # otherwise nn.Module.__init__ gets called twice!
            Filter.__init__(self, input_size, hidden_size)

        def forward(self, y: Tensor, x: Tensor) -> Tensor:
            for module in self:
                x = module(y, x)
            return x

    m = 3
    n = 4
    f = FilterSequence([nn.RNNCell(3, 4)], input_size=m, hidden_size=n)
    assert isinstance(f[0], nn.Module)
