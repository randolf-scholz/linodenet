r"""Test JIT scriptability and serialization of a dataclass module in PyTorch."""

from dataclasses import dataclass

import torch
from torch import Tensor, jit, nn


@dataclass
class MyModule(nn.Module):
    weight: Tensor

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
    torch.jit.save(scripted_module, "temp_module.pt")
    loaded_module = torch.jit.load("temp_module.pt")

    # Verify the deserialized module produces the same output
    input_tensor = torch.tensor(3.0)
    original_output = scripted_module(input_tensor)
    loaded_output = loaded_module(input_tensor)
    assert torch.allclose(original_output, loaded_output), (
        "Serialized module output mismatch"
    )
