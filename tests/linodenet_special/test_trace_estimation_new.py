import pytest
import torch
from torch import Tensor

from linodenet_special.trace_estimation import ExactEstimator, HutchinsonEstimator
from tests.testing import DEVICES


def scaled_map(scale: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return scale * x

    return op


def linear_map(matrix: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return torch.einsum("...ij, ...j -> ...i", matrix, x)

    return op


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
        estimator = HutchinsonEstimator(num_samples=4)

        with pytest.raises(ValueError, match="at least one of op or adj_op"):
            next(estimator.estimate_powers(None, None, 1, shape=(2, 1)))
