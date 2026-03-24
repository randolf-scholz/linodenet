import pytest
import torch
from torch import Tensor

from linodenet_special.trace_estimation import (
    ExactEstimator,
    HutchinsonEstimator,
    HutchPlusPlusEstimator,
    SamplerKind,
    XTraceEstimator,
    xtrace_estimator_corrected,
)
from tests.testing import DEVICES


def scaled_map(scale: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return scale * x

    return op


def linear_map(matrix: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return torch.einsum("...ij, ...j -> ...i", matrix, x)

    return op


class OnesSampler:
    def sample(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        return torch.ones((*shape, num), dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestExactEstimator:
    def test_exact_estimate_op_only(self, device: str) -> None:
        matrix = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.5, 0.0], [0.0, -0.5]],
            ],
            device=device,
        )
        estimator = ExactEstimator().to(device=device)

        estimate = estimator.estimate(linear_map(matrix), None, shape=matrix.shape[:-1])

        expected = torch.einsum("...ii -> ...", matrix)
        torch.testing.assert_close(estimate, expected)

    def test_exact_estimate_powers_adj_only(self, device: str) -> None:
        matrix = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.5, 0.0], [0.0, -0.5]],
            ],
            device=device,
        )
        estimator = ExactEstimator().to(device=device)

        estimates = list(
            estimator.estimate_powers(
                None,
                linear_map(matrix.mT),
                3,
                shape=matrix.shape[:-1],
            )
        )

        expected = [
            torch.einsum("...ii -> ...", torch.linalg.matrix_power(matrix, power))
            for power in range(1, 4)
        ]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestHutchinsonEstimator:
    NUM_SAMPLES = 1024

    def test_hutchinson_sampler_from_string(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4, sampler="sign").to(device=device)

        samples = estimator._make_samples(shape=(3, 5))

        assert isinstance(estimator.sampler, type(SamplerKind.SIGN.make()))
        assert samples.shape == (3, 5, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchinson_sampler_from_enum(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4, sampler=SamplerKind.SIGN).to(
            device=device
        )

        samples = estimator._make_samples(shape=(2, 3))

        assert samples.shape == (2, 3, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchinson_sampler_from_custom_instance(self, device: str) -> None:
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=8, sampler=OnesSampler()).to(
            device=device
        )

        estimate = estimator.estimate(scaled_map(scale), None, shape=tuple(scale.shape))

        expected = scale.squeeze(-1)
        torch.testing.assert_close(estimate, expected)

    def test_hutchinson_sampler_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="is not a valid SamplerKind"):
            HutchinsonEstimator(num_samples=4, sampler="unknown").to(device=device)

    def test_hutchinson_estimate_op_only(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=self.NUM_SAMPLES).to(device=device)

        estimate = estimator.estimate(scaled_map(scale), None, shape=tuple(scale.shape))

        expected = scale.squeeze(-1)
        torch.testing.assert_close(estimate, expected, atol=0.08, rtol=0.0)

    def test_hutchinson_estimate_powers_adj_only(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=self.NUM_SAMPLES).to(device=device)

        estimates = list(
            estimator.estimate_powers(
                None, scaled_map(scale), 4, shape=tuple(scale.shape)
            )
        )

        expected = [scale.squeeze(-1).pow(power) for power in range(1, 5)]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth, atol=0.08, rtol=0.0)

    def test_hutchinson_estimate_powers_two_sided(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=self.NUM_SAMPLES).to(device=device)

        estimates = list(
            estimator.estimate_powers(
                lambda x: scale * x,
                lambda x: scale * x,
                4,
                shape=tuple(scale.shape),
            )
        )

        expected = [scale.squeeze(-1).pow(power) for power in range(1, 5)]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth, atol=0.08, rtol=0.0)

    def test_hutchinson_estimator_requires_operator(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4).to(device=device)

        with pytest.raises(ValueError, match="at least one of op or adj_op"):
            next(estimator.estimate_powers(None, None, 1, shape=(2, 1)))


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestHutchPlusPlusEstimator:
    NUM_SAMPLES = 32
    BATCH_SIZE = 2
    INPUT_SIZE = 100

    def test_hutchplusplus_sampler_from_string(self, device: str) -> None:
        estimator = HutchPlusPlusEstimator(num_samples=6, sampler="sign").to(
            device=device
        )

        samples = estimator._make_samples(4, shape=(2, 3))

        assert isinstance(estimator.sampler, type(SamplerKind.SIGN.make()))
        assert samples.shape == (2, 3, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchplusplus_sampler_from_custom_instance(self, device: str) -> None:
        estimator = HutchPlusPlusEstimator(num_samples=6, sampler=OnesSampler()).to(
            device=device
        )

        samples = estimator._make_samples(4, shape=(2, 3))

        assert samples.shape == (2, 3, 4)
        torch.testing.assert_close(samples, torch.ones_like(samples))

    def test_hutchplusplus_sampler_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="is not a valid SamplerKind"):
            HutchPlusPlusEstimator(num_samples=6, sampler="unknown").to(device=device)

    def test_hutchplusplus_estimate_op_only(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = HutchPlusPlusEstimator(num_samples=self.NUM_SAMPLES).to(
            device=device
        )

        estimate = estimator.estimate(lambda x: scale * x, None, shape=scale.shape)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=1e-2, rtol=0.0)

    def test_hutchplusplus_estimate_powers_adj_only(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = HutchPlusPlusEstimator(num_samples=self.NUM_SAMPLES).to(
            device=device
        )

        estimate = estimator.estimate(None, lambda x: scale * x, shape=scale.shape)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=1e-2, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestXTraceEstimator:
    NUM_SAMPLES = 3
    BATCH_SIZE = 2
    INPUT_SIZE = 8

    def test_xtrace_sampler_from_enum(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=SamplerKind.SIGN).to(
            device=device
        )

        samples = estimator._make_samples(3, shape=(2, 5))

        assert samples.shape == (2, 5, 3)
        assert torch.all((samples == -1) | (samples == +1))

    def test_xtrace_sampler_from_custom_instance(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=OnesSampler()).to(
            device=device
        )

        samples = estimator._make_samples(3, shape=(2, 5))

        assert samples.shape == (2, 5, 3)
        torch.testing.assert_close(samples, torch.ones_like(samples))

    def test_xtrace_sampler_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="is not a valid SamplerKind"):
            XTraceEstimator(num_samples=4, sampler="unknown").to(device=device)

    def test_xtrace_estimate_op_only(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = XTraceEstimator(num_samples=self.NUM_SAMPLES).to(device=device)

        estimate = estimator.estimate(lambda x: scale * x, None, shape=scale.shape)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=1e-2, rtol=0.0)

    def test_xtrace_corrected(self, device: str) -> None:

        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)

        samples = torch.randn(
            self.BATCH_SIZE, self.NUM_SAMPLES, self.INPUT_SIZE, device=device
        )
        estimate = xtrace_estimator_corrected(
            torch.func.vmap(lambda x: scale * x, -2, -2), samples
        )

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=1e-6, rtol=0.0)
