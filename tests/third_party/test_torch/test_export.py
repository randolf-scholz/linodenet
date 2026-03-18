r"""Test whether export is compatible with parametrizations."""

from tempfile import TemporaryFile

import torch
from torch import Tensor, nn
from torch.export import Dim, export


def test_minimal() -> None:
    with TemporaryFile() as file:
        model = torch.nn.Linear(3, 3)
        exported_model = export(model, args=(torch.randn(3),))
        # "BufferedRandom" cannot be assigned to type "str | PathLike[Unknown] | BytesIO"
        torch.export.save(exported_model, file)


def test_export() -> None:
    # SEE: https://pytorch.org/docs/stable/export.html#limitations-of-torch-export
    class M(nn.Module):
        input_size_a: int = 8
        input_size_b: int = 16
        hidden_size: int = 16

        def __init__(self) -> None:
            super().__init__()
            m = self.hidden_size
            n1 = self.input_size_a
            n2 = self.input_size_b

            self.branch1 = nn.Sequential(nn.Linear(n1, m), nn.ReLU())
            self.branch2 = nn.Sequential(nn.Linear(n2, m), nn.ReLU())
            self.buffer = torch.ones(m)

        def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return out1 + self.buffer, out2

    B = 32
    N1 = M.input_size_a
    N2 = M.input_size_b

    example_args = (torch.randn(B, N1), torch.randn(B, N2))

    # Create a dynamic batch size
    batch = Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, dynamic_shapes=dynamic_shapes
    )
    print(exported_program)

    # make a backward pass
    B = 24
    args = (torch.randn(B, N1), torch.randn(B, N2))
    m_exported = exported_program.module()
    output = m_exported(*args)
    output[0].mean().backward()

    # test serialize and deserialize
    with TemporaryFile() as file:
        torch.export.save(exported_program, file)
        deserialized = torch.export.load(file)
    # test deserialized program
    args = (torch.randn(B, N1), torch.randn(B, N2))
    m_deserialized = deserialized.module()
    output = m_deserialized(*args)
    output[0].mean().backward()


def test_exported_trainable() -> None:
    module = torch.nn.Linear(3, 3)
    exported = export(module, args=(torch.randn(2, 3),))

    with TemporaryFile() as file:
        torch.export.save(exported, file)
        deserialized = torch.export.load(file).module()

    optim = torch.optim.SGD(deserialized.parameters(), lr=0.01)
    arg = torch.randn(2, 3)
    output = deserialized(arg)
    loss = output.norm()
    loss.backward()
    optim.step()
    new_output = deserialized(arg)
    new_loss = new_output.norm()
    assert torch.any(new_output != output)
    assert new_loss < loss


def test_export_with_property() -> None:
    # SEE: https://pytorch.org/docs/stable/export.html#limitations-of-torch-export
    class M(nn.Module):
        input_size_a: int = 8
        input_size_b: int = 16
        hidden_size: int = 16
        parametrized_weight: Tensor

        def __init__(self) -> None:
            super().__init__()
            m = self.hidden_size
            n1 = self.input_size_a
            n2 = self.input_size_b

            self.branch1 = nn.Sequential(nn.Linear(n1, m), nn.ReLU())
            self.branch2 = nn.Sequential(nn.Linear(n2, m), nn.ReLU())
            self.buffer = torch.ones(m)
            self.param = nn.Parameter(torch.randn(m, m))
            self.register_buffer("parametrized_weight", torch.empty(m, m))
            # initialize buffer
            assert torch.allclose(self.symmetric, self.symmetric.T)
            assert self.parametrized_weight.shape == (m, m)

        @property
        def symmetric(self) -> Tensor:
            new = self.param + self.param.T
            self.parametrized_weight.copy_(new.detach())
            return self.param + self.param.T

        def forward(self, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
            out1 = self.branch1(x1)
            _ = self.branch2(x2)
            return out1 @ self.symmetric + self.buffer, self.symmetric

    B = 32
    N1 = M.input_size_a
    N2 = M.input_size_b
    example_args = (torch.randn(B, N1), torch.randn(B, N2))

    # Create a dynamic batch size
    batch = Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, dynamic_shapes=dynamic_shapes
    )
    print(exported_program)

    # make a backward pass
    B = 24
    args = (torch.randn(B, N1), torch.randn(B, N2))
    m_exported = exported_program.module()
    output = m_exported(*args)
    output[0].mean().backward()

    # test serialize and deserialize
    with TemporaryFile() as file:
        torch.export.save(exported_program, file)
        deserialized = torch.export.load(file).module()

    # test deserialized program
    args = (torch.randn(B, N1), torch.randn(B, N2))
    output = deserialized(*args)
    output[0].mean().backward()
    assert torch.allclose(output[1], output[1].T)
