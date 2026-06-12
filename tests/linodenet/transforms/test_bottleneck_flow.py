import pytest
import torch

from linodenet.mappings.transforms import BottleneckFlow, ResidualBottleneck
from tests.testing import DEVICES

from .base import TestTransform


@pytest.mark.parametrize("dtype", [torch.float32], ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_rezero", [False, True], ids=["plain", "rezero"])
class TestBottleneckFlow(TestTransform):
    SEED = 0
    BATCH_SIZE = 32
    INPUT_SIZE = 16
    HIDDEN_SIZE = 4
    NUM_BLOCKS = 3
    LAYERS_PER_BLOCK = 2
    VALUE_TOL = {
        torch.float32: (1e-5, 1e-5),
    }

    @staticmethod
    def make_model(
        input_size: int,
        /,
        *,
        hidden_size: int,
        num_blocks: int,
        layers_per_block: int,
        use_rezero: bool,
        device: str,
        dtype: torch.dtype,
    ) -> BottleneckFlow:
        return BottleneckFlow(
            input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            use_rezero=use_rezero,
        ).to(device=device, dtype=dtype)

    def test_instantiation(
        self,
        dtype: torch.dtype,
        device: str,
        use_rezero: bool,
    ) -> None:
        model = self.make_model(
            self.INPUT_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_blocks=self.NUM_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            use_rezero=use_rezero,
            device=device,
            dtype=dtype,
        )

        assert model.input_size == self.INPUT_SIZE
        assert model.hidden_size == self.HIDDEN_SIZE
        assert model.num_blocks == self.NUM_BLOCKS
        assert model.layers_per_block == self.LAYERS_PER_BLOCK
        assert model.use_rezero is use_rezero
        assert len(model) == self.NUM_BLOCKS
        assert all(isinstance(block, ResidualBottleneck) for block in model)

    @pytest.mark.parametrize("hidden_size", [1, 4, 8], ids="hidden_size={}".format)
    @pytest.mark.parametrize("layers_per_block", [1, 2], ids="layers={}".format)
    def test_invertible_at_initialization(
        self,
        hidden_size: int,
        layers_per_block: int,
        dtype: torch.dtype,
        device: str,
        use_rezero: bool,
    ) -> None:
        torch.manual_seed(self.SEED)
        atol, rtol = self.VALUE_TOL[dtype]
        model = self.make_model(
            self.INPUT_SIZE,
            hidden_size=hidden_size,
            num_blocks=self.NUM_BLOCKS,
            layers_per_block=layers_per_block,
            use_rezero=use_rezero,
            device=device,
            dtype=dtype,
        )
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        y = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        self.assert_invertible(
            model,
            x,
            y,
            atol=atol,
            rtol=rtol,
            logdet_atol=atol,
            logdet_rtol=rtol,
        )
