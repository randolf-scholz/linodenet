import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from linodenet.mappings import ContractiveNew, ContractiveTransform, TransformBase
from linodenet.mappings.linear import LinearContraction
from tests.testing import DEVICES, SEEDS_10, TestCase


class TestContractiveFlow(TestCase):
    VALUE_ATOL = 1e-3
    VALUE_RTOL = 1e-3
    BATCH_SIZE = 32
    PERF_SEED = 0
    PERF_INPUT_SIZE = 256
    PERF_ROUNDS = 10
    PERF_WARMUP_ROUNDS = 1

    @pytest.mark.parametrize(
        "flow_cls",
        [ContractiveTransform, ContractiveNew],
        ids=["loop", "while_loop"],
    )
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64], ids="input_size={}".format)
    def test_invertibility(
        self,
        flow_cls: type[TransformBase],
        device: str,
        seed: int,
        input_size: int,
    ) -> None:
        r"""Check forward/inverse round trips; does not test logabsdet (not implemented yet)."""
        torch.manual_seed(seed)
        layer = LinearContraction(input_size, input_size, bias=True).to(device)
        flow = flow_cls(layer)

        x = torch.randn(self.BATCH_SIZE, input_size, device=device)
        y = flow.encode(x)
        xhat = flow.decode(y)

        assert y.shape == x.shape
        assert xhat.shape == x.shape
        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

        y = torch.randn(self.BATCH_SIZE, input_size, device=device)
        x = flow.decode(y)
        yhat = flow.encode(x)

        assert x.shape == y.shape
        assert yhat.shape == y.shape
        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

    @pytest.mark.parametrize(
        "flow_cls",
        [ContractiveTransform, ContractiveNew],
        ids=["loop", "while_loop"],
    )
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    def test_decode_performance(
        self,
        benchmark: BenchmarkFixture,
        flow_cls: type[TransformBase],
        device: str,
    ) -> None:
        r"""Benchmark the compiled inverse pass on a representative large input."""
        benchmark.group = (
            f"contractive_decode/{device}/seed={self.PERF_SEED}/"
            f"input_size={self.PERF_INPUT_SIZE}"
        )
        torch.manual_seed(self.PERF_SEED)
        layer = LinearContraction(
            self.PERF_INPUT_SIZE,
            self.PERF_INPUT_SIZE,
            bias=True,
        ).to(device)
        flow = flow_cls(layer)
        compiled_decode = torch.compile(
            flow.decode,
            fullgraph=flow_cls is ContractiveNew,
        )

        # trigger compile
        y_demo = torch.randn(self.BATCH_SIZE, self.PERF_INPUT_SIZE, device=device)
        compiled_decode(y_demo)

        def setup() -> tuple[tuple, dict]:
            y = torch.randn(self.BATCH_SIZE, self.PERF_INPUT_SIZE, device=device)
            return (y,), {}

        benchmark.pedantic(
            compiled_decode,
            setup=setup,
            rounds=self.PERF_ROUNDS,
            warmup_rounds=self.PERF_WARMUP_ROUNDS,
        )
