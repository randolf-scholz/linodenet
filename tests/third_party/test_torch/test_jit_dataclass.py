r"""Test JIT scriptability and serialization of a dataclass module in PyTorch."""

from dataclasses import dataclass
from tempfile import TemporaryFile

import torch
from torch import Tensor, jit, nn


@dataclass
class MyModule(nn.Module):
    # NOTE: still need to manually do __init__
    weight: Tensor

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


def test_jit_scriptable() -> None:
    # Create an instance of the dataclass module
    module = MyModule(weight=torch.tensor(2.0))

    # Check if the module is JIT-scriptable
    scripted_module = jit.script(module)
    assert scripted_module is not None, "Module is not JIT-scriptable"

    # Verify the scripted module produces the correct output
    input_tensor = torch.tensor(3.0)
    output = scripted_module(input_tensor)
    assert torch.allclose(output, torch.tensor(6.0)), "Scripted module output mismatch"


def test_serializable() -> None:
    # Create an instance of the dataclass module
    module = MyModule(weight=torch.tensor(2.0))

    # Script the module
    scripted_module = jit.script(module)

    # Serialize and deserialize the scripted module
    with TemporaryFile() as file:
        torch.jit.save(scripted_module, file)
        file.seek(0)
        loaded_module = torch.jit.load(file)

    # Verify the deserialized module produces the same output
    input_tensor = torch.tensor(3.0)
    original_output = scripted_module(input_tensor)
    loaded_output = loaded_module(input_tensor)
    assert torch.allclose(original_output, loaded_output), (
        "Serialized module output mismatch"
    )
