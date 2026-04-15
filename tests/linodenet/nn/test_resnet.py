import pytest
import torch
from torch import nn

from linodenet.nn import ResNet
from linodenet.nn.rezero import ReZero
from tests.testing import TestSuite


@pytest.mark.parametrize("use_rezero", [False, True], ids=["plain", "rezero"])
def test_instantiation(use_rezero: bool) -> None:
    input_size = 8
    num_blocks = 3
    layers_per_block = 3
    latent_size = 16

    model = ResNet(
        input_size,
        num_blocks=num_blocks,
        layers_per_block=layers_per_block,
        latent_size=latent_size,
        activation="ReLU",
        use_rezero=use_rezero,
    )

    assert model.input_size == input_size
    assert model.num_blocks == num_blocks
    assert model.layers_per_block == layers_per_block
    assert model.latent_size == latent_size
    assert model.use_rezero is use_rezero
    assert len(model) == num_blocks
    expected_type = ReZero if use_rezero else nn.Sequential
    assert all(isinstance(block, expected_type) for block in model)


class TestResNet(TestSuite):
    def test_rezero_initializes_to_identity(self) -> None:
        model = ResNet(
            4,
            num_blocks=2,
            layers_per_block=2,
            latent_size=6,
            use_rezero=True,
        )
        x = torch.randn(5, 4)

        y = model(x)

        self.assert_close(y, x, atol=1e-6, rtol=1e-6)
