import pytest
import torch

from linodenet.mappings import LinearContraction, ResidualBottleneck, ReZeroBottleneck
from linodenet.nn.parametrize import update_parametrizations
from tests.testing import DEVICES, DTYPES, SEEDS_5, TestSuite

from .test_transform import TestTransform


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestResidualBottleneck(TestTransform):
    VALUE_TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-6, 1e-6),
    }
    LOGDET_TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-6, 1e-6),
    }
    BATCH_SIZE = 32
    FINITE_DIFF_STEP = {
        torch.float32: 1e-3,
        torch.float64: 1e-5,
    }
    FINITE_DIFF_TOL = {
        torch.float32: (2e-2, 2e-2),
        torch.float64: (2e-4, 2e-4),
    }

    def make_flow(
        self,
        input_size: int,
        hidden_size: int,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> ResidualBottleneck:
        flow = ResidualBottleneck(
            input_size=input_size,
            hidden_size=hidden_size,
            bottleneck=LinearContraction(
                hidden_size,
                hidden_size,
                bias=True,
                c=0.5,
                device=device,
                dtype=dtype,
            ),
            activation="Tanh",
            maxiter=128,
            atol=1e-8 if dtype is torch.float64 else 1e-6,
            rtol=1e-8 if dtype is torch.float64 else 1e-6,
            device=device,
            dtype=dtype,
        )
        update_parametrizations(flow)
        return flow

    @pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16], ids="input_size={}".format)
    @pytest.mark.parametrize("hidden_size", [1, 2, 4], ids="hidden_size={}".format)
    def test_roundtrip_and_logdet_cancellation(
        self,
        seed: int,
        input_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if hidden_size > input_size:
            pytest.skip("hidden_size must not exceed input_size")

        torch.manual_seed(seed)
        atol, rtol = self.VALUE_TOL[dtype]
        logdet_atol, logdet_rtol = self.LOGDET_TOL[dtype]
        flow = self.make_flow(
            input_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        x = torch.randn(self.BATCH_SIZE, input_size, device=device, dtype=dtype)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

        assert y.shape == x.shape
        assert xhat.shape == x.shape
        assert forward_logabsdet.shape == x.shape[:-1]
        assert inverse_logabsdet.shape == x.shape[:-1]
        self.assert_close(xhat, x, atol=atol, rtol=rtol)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=logdet_atol,
            rtol=logdet_rtol,
        )

    def test_logabsdet_matches_dense_jacobian(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(0)
        atol, rtol = self.LOGDET_TOL[dtype]
        flow = self.make_flow(5, 2, device=device, dtype=dtype)
        x = torch.randn(5, device=device, dtype=dtype, requires_grad=True)

        y, logabsdet = flow.encode_and_logabsdet(x)
        jacobian = torch.autograd.functional.jacobian(flow.encode, x)
        _, expected_logabsdet = torch.linalg.slogdet(jacobian)

        assert y.shape == x.shape
        assert logabsdet.shape == x.shape[:-1]
        self.assert_close(logabsdet, expected_logabsdet, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
    def test_logabsdet_matches_finite_difference_volume_change(
        self,
        seed: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(seed)
        atol, rtol = self.FINITE_DIFF_TOL[dtype]
        flow = self.make_flow(5, 2, device=device, dtype=dtype)
        x = torch.randn(3, 5, device=device, dtype=dtype)
        self.assert_logabsdet_matches_finite_difference_volume_change(
            flow,
            x,
            step=self.FINITE_DIFF_STEP[dtype],
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestReZeroBottleneck(TestSuite):
    def test_initially_identity(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        flow = ReZeroBottleneck(
            input_size=5,
            hidden_size=2,
            bottleneck=LinearContraction(
                2,
                2,
                bias=True,
                c=0.5,
                device=device,
                dtype=dtype,
            ),
            activation="Tanh",
            use_bias=False,
            maxiter=128,
            device=device,
            dtype=dtype,
        )
        update_parametrizations(flow)
        x = torch.randn(16, 5, device=device, dtype=dtype)

        y, logabsdet = flow.encode_and_logabsdet(x)

        self.assert_close(y, x)
        self.assert_close(logabsdet, torch.zeros_like(logabsdet))
