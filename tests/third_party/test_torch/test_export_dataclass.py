from dataclasses import dataclass
from tempfile import TemporaryFile

import torch
from torch import Tensor, nn


@dataclass
class Value:
    x: Tensor

    def __post_init__(self) -> None:
        self.x = torch.nan_to_num(self.x, nan=0.0)


def test_intermediate_dataclass() -> None:

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            value = Value(x=x)
            return value.x

    model = Model()
    inputs = torch.tensor([1.0, float("nan"), -2.0])
    exported = torch.export.export(model, args=(inputs,))

    with TemporaryFile() as file:
        torch.export.save(exported, file)
        deserialized = torch.export.load(file).module()

    expected = model(inputs)
    actual = deserialized(inputs)
    assert torch.equal(expected, torch.tensor([1.0, 0.0, -2.0]))
    assert torch.equal(actual, expected)
