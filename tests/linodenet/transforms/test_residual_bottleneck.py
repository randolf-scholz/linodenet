import pytest
import torch
from torch import nn

from linodenet.mappings import LinearContraction, ResidualBottleneck
from linodenet.nn.parametrize import update_parametrizations
from tests.testing import DEVICES, SEEDS_5

from .test_transform import TestTransform


@pytest.mark.parametrize("dtype", [torch.float32], ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestResidualBottleneck(TestTransform):
    VALUE_TOL = {
        torch.float32: (1e-4, 1e-4),
    }
    LOGDET_TOL = {
        torch.float32: (1e-4, 1e-4),
    }
    BATCH_SIZE = 32
    FINITE_DIFF_STEP = {
        torch.float32: 1e-3,
    }
    FINITE_DIFF_TOL = {
        torch.float32: (2e-2, 2e-2),
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
            bottleneck=nn.Sequential(
                nn.ELU(),
                LinearContraction(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    c=0.5,
                    device=device,
                    dtype=dtype,
                ),
            ),
            gate="identity",
            maxiter=128,
            atol=1e-6,
            rtol=1e-6,
            device=device,
            dtype=dtype,
        )
        update_parametrizations(flow)
        return flow

    def test_identity_gate_ignores_scalar_map_and_has_no_scalar(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        r"""Non-ReZero bottlenecks should warn about unused scalar maps."""
        with pytest.warns(
            UserWarning,
            match="Ignoring scalar_map because gate is not 'rezero'.",
        ):
            flow = ResidualBottleneck(
                input_size=5,
                hidden_size=2,
                bottleneck=nn.Identity(),
                gate="identity",
                scalar_map=nn.Identity(),
                device=device,
                dtype=dtype,
            )

        assert isinstance(flow.gate, nn.Identity)

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
        y = torch.randn(self.BATCH_SIZE, input_size, device=device, dtype=dtype)
        self.assert_invertible(
            flow,
            x,
            y,
            atol=atol,
            rtol=rtol,
            logdet_atol=logdet_atol,
            logdet_rtol=logdet_rtol,
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
