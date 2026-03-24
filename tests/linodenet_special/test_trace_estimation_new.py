from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor
from torch.func import vmap

from linodenet_special.trace_estimation import (
    ExactEstimator,
    HutchinsonEstimator,
    HutchPlusPlusEstimator,
    SamplerKind,
    XTraceEstimator,
    xtrace_estimator_corrected,
)
from tests.testing import DEVICES, PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def scaled_map(scale: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return scale * x

    return op


def linear_map(matrix: Tensor):
    def op(x: Tensor, /) -> Tensor:
        return torch.einsum("...ij, ...j -> ...i", matrix, x)

    return op


class OnesSampler:
    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        return torch.ones((*shape, num), dtype=dtype, device=device)


class FixedSampler:
    def __init__(self, samples: Tensor, /) -> None:
        self.samples = samples

    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        assert self.samples.shape == (*shape, num)
        return self.samples.to(device=device, dtype=dtype)


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

        samples = estimator.sampler((3, 5), 4, dtype=torch.float32, device=device)

        assert isinstance(estimator.sampler, type(SamplerKind.SIGN.make()))
        assert samples.shape == (3, 5, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchinson_sampler_from_enum(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4, sampler=SamplerKind.SIGN).to(
            device=device
        )

        samples = estimator.sampler((2, 3), 4, dtype=torch.float32, device=device)

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
    NUM_SAMPLES = 80
    BATCH_SIZE = 2
    INPUT_SIZE = 100

    def test_hutchplusplus_sampler_from_string(self, device: str) -> None:
        estimator = HutchPlusPlusEstimator(num_samples=6, sampler="sign").to(
            device=device
        )

        samples = estimator.sampler((2, 3), 4, dtype=torch.float32, device=device)

        assert samples.shape == (2, 3, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchplusplus_sampler_from_custom_instance(self, device: str) -> None:
        estimator = HutchPlusPlusEstimator(num_samples=6, sampler=OnesSampler()).to(
            device=device
        )

        samples = estimator.sampler((2, 3), 4, dtype=torch.float32, device=device)

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
        estimator = HutchPlusPlusEstimator(
            num_samples=self.NUM_SAMPLES, sampler="sphere"
        ).to(device=device)

        estimate = estimator.estimate(None, lambda x: scale * x, shape=scale.shape)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestXTraceEstimator:
    NUM_SAMPLES = 5
    BATCH_SIZE = 2
    INPUT_SIZE = 8

    def make_test(self, device: str) -> tuple[Callable[[Tensor], Tensor], Tensor]:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE).to(device=device)
        return lambda x: scale * x, scale.sum(-1)

    def test_xtrace_sampler_from_enum(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=SamplerKind.SIGN).to(
            device=device
        )

        samples = estimator.sampler((2, 5), 3, dtype=torch.float32, device=device)

        assert samples.shape == (2, 5, 3)
        assert torch.all((samples == -1) | (samples == +1))

    def test_xtrace_sphere_sampler_normalizes_columns(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=SamplerKind.SPHERE).to(
            device=device
        )

        samples = estimator.sampler((2, 5), 3, dtype=torch.float64, device=device)

        assert samples.shape == (2, 5, 3)
        expected_norm = torch.full((2, 3), 5.0**0.5, dtype=samples.dtype, device=device)
        torch.testing.assert_close(
            torch.linalg.vector_norm(samples, dim=-2), expected_norm
        )

    def test_xtrace_sampler_from_custom_instance(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=OnesSampler()).to(
            device=device
        )

        samples = estimator.sampler((2, 5), 3, dtype=torch.float32, device=device)

        assert samples.shape == (2, 5, 3)
        torch.testing.assert_close(samples, torch.ones_like(samples))

    def test_xtrace_sampler_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="is not a valid SamplerKind"):
            XTraceEstimator(num_samples=4, sampler="unknown").to(device=device)

    def test_xtrace_estimate_op_only(self, device: str) -> None:
        fn, expected = self.make_test(device=device)
        shape = (self.BATCH_SIZE, self.INPUT_SIZE)

        estimator = XTraceEstimator(
            num_samples=self.NUM_SAMPLES,
            sampler="sphere",
            renormalize=True,
        ).to(device=device)

        estimate = estimator.estimate(fn, None, shape=shape)

        torch.testing.assert_close(estimate, expected, atol=1e-3, rtol=1e-3)

    def test_xtrace_corrected(self, device: str) -> None:
        fn, expected = self.make_test(device=device)

        samples = torch.randn(
            self.BATCH_SIZE, self.NUM_SAMPLES, self.INPUT_SIZE, device=device
        )
        estimate = xtrace_estimator_corrected(vmap(fn, -2, -2), samples)

        torch.testing.assert_close(estimate, expected, atol=1e-3, rtol=1e-3)

    def test_xtrace_relative_error_plot(self, device: str) -> None:
        mpl.use("Agg")
        torch.manual_seed(0)

        batch_size = 128
        input_size = 256
        dtype = torch.float32
        num_samples_grid = (1, 2, 4, 8, 16, 32, 64, 128, 256)
        result_dir = RESULT_DIR
        result_dir.mkdir(exist_ok=True)

        scale = 0.5 + torch.rand(batch_size, input_size, device=device, dtype=dtype)
        fn = lambda x: scale * x
        expected = scale.sum(-1)

        corrected_errors: list[Tensor] = []
        estimator_errors: list[Tensor] = []
        base_sampler = SamplerKind.SPHERE.make()

        for num_samples in num_samples_grid:
            probe_columns = base_sampler(
                (batch_size, input_size),
                num_samples,
                dtype=dtype,
                device=device,
            )
            rowwise_samples = probe_columns.mT

            corrected = xtrace_estimator_corrected(vmap(fn, -2, -2), rowwise_samples)
            estimator = XTraceEstimator(
                num_samples=num_samples,
                sampler=FixedSampler(probe_columns),
                renormalize=True,
            ).to(device=device, dtype=dtype)
            estimate = estimator.estimate(fn, None, shape=(batch_size, input_size))

            denom = expected.abs().clamp_min(torch.finfo(dtype).eps)
            corrected_errors.append(((corrected - expected).abs() / denom).mean())
            estimator_errors.append(((estimate - expected).abs() / denom).mean())

        corrected_curve = torch.stack(corrected_errors).cpu()
        estimator_curve = torch.stack(estimator_errors).cpu()

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot(num_samples_grid, corrected_curve, marker="o", label="corrected")
        ax.plot(num_samples_grid, estimator_curve, marker="s", label="module")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_samples")
        ax.set_ylabel("mean relative error")
        ax.set_title(
            f"XTrace relative error ({device}, batch={batch_size}, input={input_size})"
        )
        ax.legend()

        out = result_dir / f"xtrace_relative_error_comparison_{device}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        assert out.exists()
        assert torch.isfinite(corrected_curve).all()
        assert torch.isfinite(estimator_curve).all()
