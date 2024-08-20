r"""Test JIT compatibility with subclassing."""

from typing import Final, Optional

import pytest
import torch
from torch import Tensor, jit, nn


class Foo(nn.Module):
    r"""Dummy module."""

    input_size: Final[int]
    r"""The input size of the module."""
    hidden_size: Final[int]
    r"""The hidden size of the module."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class WithDecoder(Foo):
    r"""Direct subclass of Foo."""

    decoder: Optional[nn.Module]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        decoder: nn.Module,
    ) -> None:
        super().__init__(input_size, hidden_size)
        self.register_module("decoder", decoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(super().forward(x))


class InheritedDecoder(WithDecoder):
    r"""Direct subclass of WithDecoder."""


def test_with_decoder():
    decoder = nn.Linear(4, 4)
    model = WithDecoder(3, 4, decoder=decoder)
    assert isinstance(model.decoder, nn.Linear)
    x = torch.randn(3, 3)
    y = model(x)
    assert y.shape == (3, 4)

    # test with scripted
    with pytest.raises(
        AssertionError,
        match="Unsupported annotation typing.Optional[torch.nn.modules.module.Module]*.",
    ):
        scripted_model = jit.script(model)

    assert isinstance(scripted_model, jit.ScriptModule)
    y = scripted_model(x)
    assert y.shape == (3, 4)

    # # serialize and deserialize
    # with TemporaryDirectory() as folder:
    #     path = f"{folder}/model.pt"
    #     jit.save(scripted_model, path)
    #     deserialized_model = jit.load(path)
    #     y = deserialized_model(x)
    #     assert y.shape == (3, 4)
    #     assert deserialized_model.input_size == 3


def test_subclass_with_decoder():
    decoder = nn.Linear(4, 4)
    model = InheritedDecoder(3, 4, decoder=decoder)
    print(model.decoder)
    assert isinstance(model.decoder, nn.Linear)
    x = torch.randn(3, 3)
    y = model(x)
    assert y.shape == (3, 4)

    # test with scripted
    with pytest.raises(
        AssertionError,
        match="Unsupported annotation typing.Optional[torch.nn.modules.module.Module]*.",
    ):
        jit.script(model)

    # assert isinstance(scripted_model, jit.ScriptModule)
    # y = scripted_model(x)
    # assert y.shape == (3, 4)

    # # serialize and deserialize
    # with TemporaryDirectory() as folder:
    #     path = f"{folder}/model.pt"
    #     jit.save(scripted_model, path)
    #     deserialized_model = jit.load(path)
    #     y = deserialized_model(x)
    #     assert y.shape == (3, 4)
    #     assert deserialized_model.input_size == 3
