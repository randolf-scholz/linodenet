from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import ortho_group
from torch import Tensor
from torch.func import vmap

from linodenet_special.trace_estimation import (
    BaseEstimator,
    ExactEstimator,
    HutchinsonEstimator,
    HutchPlusPlusEstimator,
    Sampler,
    XTraceEstimator,
    logabsdet_series,
    xtrace_estimator_matlab,
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


class SequenceSampler:
    def __init__(self, samples: list[Tensor], /) -> None:
        self.samples = samples
        self.index = 0

    def __call__(
        self,
        shape: tuple[int, ...],
        num: int,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        sample = self.samples[self.index]
        self.index += 1
        assert sample.shape == (*shape, num)
        return sample.to(device=device, dtype=dtype)


class AnalyticEstimator(BaseEstimator):
    def forward(
        self,
        op: Callable[[Tensor], Tensor],
        x: Tensor,
        /,
    ) -> Tensor:
        eye = torch.eye(
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        ).expand(*x.shape[:-1], x.shape[-1], x.shape[-1])
        matrix = vmap(op, in_dims=-1, out_dims=-1)(eye)
        return torch.einsum("...ii -> ...", matrix)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestExactEstimator:
    @pytest.mark.parametrize("mode", ["forward", "adjoint"])
    def test_exact_estimate(self, device: str, mode: str) -> None:
        matrix = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.5, 0.0], [0.0, -0.5]],
            ],
            device=device,
        )
        x = torch.zeros(matrix.shape[:-1], device=device)
        estimator = ExactEstimator(mode=mode).to(device=device)

        estimate = estimator(linear_map(matrix), x)

        expected = torch.einsum("...ii -> ...", matrix)
        torch.testing.assert_close(estimate, expected)

    @pytest.mark.parametrize("mode", ["forward", "adjoint"])
    def test_exact_estimate_powers(self, device: str, mode: str) -> None:
        matrix = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.5, 0.0], [0.0, -0.5]],
            ],
            device=device,
        )
        x = torch.zeros(matrix.shape[:-1], device=device)
        estimator = ExactEstimator(mode=mode).to(device=device)

        estimates = list(estimator.estimate_powers(linear_map(matrix), x, 3))

        expected = [
            torch.einsum("...ii -> ...", torch.linalg.matrix_power(matrix, power))
            for power in range(1, 4)
        ]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth)

    @pytest.mark.parametrize("mode", ["forward", "adjoint"])
    def test_exact_estimate_logabsdet_matches_closed_form(
        self, device: str, mode: str
    ) -> None:
        matrix = torch.tensor(
            [
                [[0.25, 0.0], [0.0, -0.125]],
                [[-0.5, 0.0], [0.0, 0.75]],
            ],
            device=device,
        )
        x = torch.zeros(matrix.shape[:-1], device=device)
        estimator = ExactEstimator(mode=mode).to(device=device)

        estimate = estimator.estimate_logabsdet(linear_map(matrix), x)

        eigenvalues = torch.linalg.eigvals(matrix)
        expected = torch.log(torch.abs(1 + eigenvalues)).sum(dim=-1)
        torch.testing.assert_close(estimate, expected)

    def test_exact_mode_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="mode must be 'forward' or 'adjoint'"):
            ExactEstimator(mode="symmetric").to(device=device)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestBaseEstimator:
    def test_estimate_powers_defaults_to_repeated_estimate(self, device: str) -> None:
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = AnalyticEstimator().to(device=device)

        estimates = list(estimator.estimate_powers(scaled_map(scale), scale, 4))

        expected = [scale.squeeze(-1).pow(power) for power in range(1, 5)]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth)

    def test_estimate_logabsdet_uses_power_series(self, device: str) -> None:
        scale = torch.tensor([[0.125], [-0.2], [0.3]], device=device)
        estimator = AnalyticEstimator().to(device=device)

        estimate = logabsdet_series(estimator, scaled_map(scale), scale, 6)

        expected = sum(
            ((-1) ** (power + 1) / power) * scale.squeeze(-1).pow(power)
            for power in range(1, 7)
        )
        torch.testing.assert_close(estimate, expected)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestHutchinsonEstimator:
    NUM_SAMPLES = 1024

    def test_hutchinson_sampler_from_string(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4, sampler="sign").to(device=device)

        samples = estimator.sampler((3, 5), 4, dtype=torch.float32, device=device)

        assert isinstance(estimator.sampler, type(Sampler.new(Sampler.SIGN)))
        assert samples.shape == (3, 5, 4)
        assert torch.all((samples == -1) | (samples == +1))

    def test_hutchinson_sampler_from_enum(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4, sampler=Sampler.SIGN).to(
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

        estimate = estimator(scaled_map(scale), scale)

        expected = scale.squeeze(-1)
        torch.testing.assert_close(estimate, expected)

    def test_hutchinson_sampler_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(ValueError, match="is not a valid SamplerKind"):
            HutchinsonEstimator(num_samples=4, sampler="unknown").to(device=device)

    def test_hutchinson_mode_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(
            ValueError, match="mode must be 'forward', 'adjoint', or 'symmetric'"
        ):
            HutchinsonEstimator(num_samples=4, mode="unknown").to(device=device)

    @pytest.mark.parametrize("mode", ["forward", "adjoint", "symmetric"])
    def test_hutchinson_estimate(self, device: str, mode: str) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=self.NUM_SAMPLES, mode=mode).to(
            device=device
        )

        estimate = estimator(scaled_map(scale), scale)

        expected = scale.squeeze(-1)
        torch.testing.assert_close(estimate, expected, atol=0.08, rtol=0.0)

    @pytest.mark.parametrize("mode", ["forward", "adjoint", "symmetric"])
    def test_hutchinson_estimate_powers(self, device: str, mode: str) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([[0.25], [-0.5], [0.75]], device=device)
        estimator = HutchinsonEstimator(num_samples=self.NUM_SAMPLES, mode=mode).to(
            device=device
        )

        estimates = list(estimator.estimate_powers(scaled_map(scale), scale, 4))

        expected = [scale.squeeze(-1).pow(power) for power in range(1, 5)]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth, atol=0.08, rtol=0.0)

    def test_hutchinson_estimator_requires_vector_input(self, device: str) -> None:
        estimator = HutchinsonEstimator(num_samples=4).to(device=device)

        with pytest.raises(ValueError, match="x must be at least one-dimensional"):
            next(
                estimator.estimate_powers(
                    lambda x: x, torch.tensor(1.0, device=device), 1
                )
            )


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

    def test_hutchplusplus_mode_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(
            ValueError,
            match="mode must be 'forward', 'adjoint', or 'symmetric'",
        ):
            HutchPlusPlusEstimator(num_samples=6, mode="unknown").to(device=device)

    @pytest.mark.parametrize("mode", ["forward", "symmetric"])
    def test_hutchplusplus_estimate(self, device: str, mode: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = HutchPlusPlusEstimator(
            num_samples=self.NUM_SAMPLES,
            mode=mode,
        ).to(device=device)

        estimate = estimator(lambda x: scale * x, scale)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=0.15, rtol=0.0)

    def test_hutchplusplus_estimate_adjoint(self, device: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = HutchPlusPlusEstimator(
            num_samples=self.NUM_SAMPLES,
            sampler="sphere",
            mode="adjoint",
        ).to(device=device)

        estimate = estimator(lambda x: scale * x, scale)

        expected = scale.sum(-1)
        torch.testing.assert_close(estimate, expected, atol=0.15, rtol=0.0)

    @pytest.mark.parametrize("mode", ["forward", "adjoint", "symmetric"])
    def test_hutchplusplus_estimate_powers(self, device: str, mode: str) -> None:
        torch.manual_seed(0)
        scale = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = HutchPlusPlusEstimator(
            num_samples=self.NUM_SAMPLES,
            mode=mode,
        ).to(device=device)

        estimates = list(estimator.estimate_powers(lambda x: scale * x, scale, 3))

        expected = [scale.pow(power).sum(dim=-1) for power in range(1, 4)]
        for estimate, truth in zip(estimates, expected, strict=True):
            torch.testing.assert_close(estimate, truth, atol=0.15, rtol=0.0)


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
        estimator = XTraceEstimator(num_samples=4, sampler=Sampler.SIGN).to(
            device=device
        )

        samples = estimator.sampler((2, 5), 3, dtype=torch.float32, device=device)

        assert samples.shape == (2, 5, 3)
        assert torch.all((samples == -1) | (samples == +1))

    def test_xtrace_sphere_sampler_normalizes_columns(self, device: str) -> None:
        estimator = XTraceEstimator(num_samples=4, sampler=Sampler.SPHERE).to(
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

    def test_xtrace_mode_rejects_unknown_string(self, device: str) -> None:
        with pytest.raises(
            ValueError,
            match="mode must be 'forward', 'adjoint', or 'symmetric'",
        ):
            XTraceEstimator(num_samples=4, mode="unknown").to(device=device)

    def test_xtrace_estimate_op_only(self, device: str) -> None:
        fn, expected = self.make_test(device=device)
        x = torch.zeros(self.BATCH_SIZE, self.INPUT_SIZE, device=device)

        estimator = XTraceEstimator(
            num_samples=self.NUM_SAMPLES,
            sampler="sphere",
            renormalize=True,
        ).to(device=device)

        estimate = estimator(fn, x)

        torch.testing.assert_close(estimate, expected, atol=4.0, rtol=0.0)

    def test_xtrace_estimate_naive(self, device: str) -> None:
        fn, expected = self.make_test(device=device)
        x = torch.zeros(self.BATCH_SIZE, self.INPUT_SIZE, device=device)

        estimator = XTraceEstimator(
            num_samples=self.NUM_SAMPLES,
            sampler="sphere",
            renormalize=True,
        ).to(device=device)

        estimate = estimator.estimate_naive(fn, x)

        torch.testing.assert_close(estimate, expected, atol=4.0, rtol=0.0)

    @pytest.mark.parametrize("mode", ["adjoint", "symmetric"])
    def test_xtrace_nonforward_modes_not_implemented(
        self, device: str, mode: str
    ) -> None:
        fn, _ = self.make_test(device=device)
        x = torch.zeros(self.BATCH_SIZE, self.INPUT_SIZE, device=device)
        estimator = XTraceEstimator(num_samples=self.NUM_SAMPLES, mode=mode).to(
            device=device
        )

        with pytest.raises(
            NotImplementedError,
            match=f"XTraceEstimator only supports mode='forward', got '{mode}'",
        ):
            estimator(fn, x)

    def test_xtrace_corrected(self, device: str) -> None:
        fn, expected = self.make_test(device=device)

        samples = torch.randn(
            self.BATCH_SIZE, self.NUM_SAMPLES, self.INPUT_SIZE, device=device
        )
        estimate = xtrace_estimator_matlab(vmap(fn, -2, -2), samples)

        torch.testing.assert_close(estimate, expected, atol=0.4, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestVisualization:
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    DTYPE = torch.float32
    NUM_MATVECS_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    def compute_curves(
        self,
        fn: Callable[[Tensor], Tensor],
        expected: Tensor,
        *,
        device: str,
    ) -> dict[str, Tensor]:
        mpl.use("Agg")
        torch.manual_seed(0)

        batch_size = self.BATCH_SIZE
        input_size = self.INPUT_SIZE
        dtype = self.DTYPE
        denom = expected.abs().clamp_min(torch.finfo(dtype).eps)
        x = torch.zeros(batch_size, input_size, device=device, dtype=dtype)

        base_sampler = Sampler.new(Sampler.ORTH)
        full_probe_columns = base_sampler(
            (batch_size, input_size),
            input_size,
            dtype=dtype,
            device=device,
        )
        hutch_full_probe_columns = base_sampler(
            (batch_size, input_size),
            input_size,
            dtype=dtype,
            device=device,
        )
        hpp_full_probe_columns = base_sampler(
            (batch_size, input_size),
            input_size,
            dtype=dtype,
            device=device,
        )
        hpp_full_residual_columns = base_sampler(
            (batch_size, input_size),
            input_size,
            dtype=dtype,
            device=device,
        )

        curves: dict[str, list[Tensor]] = {
            "xtrace": [],
            "hutch": [],
            "hutch++": [],
        }

        for num_matvecs in self.NUM_MATVECS_GRID:
            xtrace_num_samples = num_matvecs // 2
            if xtrace_num_samples == 0:
                xtrace = torch.full((), torch.nan, device=device, dtype=dtype)
            else:
                probe_columns = full_probe_columns[..., :xtrace_num_samples]
                xtrace = XTraceEstimator(
                    num_matvecs=num_matvecs,
                    sampler=FixedSampler(probe_columns),
                    renormalize=True,
                ).to(device=device, dtype=dtype)(fn, x)

            hutch_columns = hutch_full_probe_columns[..., :num_matvecs]
            hutch = HutchinsonEstimator(
                num_matvecs=num_matvecs,
                sampler=FixedSampler(hutch_columns),
            ).to(device=device, dtype=dtype)(fn, x)

            hpp_num_samples = num_matvecs // 3
            if hpp_num_samples == 0:
                hutchpp = torch.full((), torch.nan, device=device, dtype=dtype)
            else:
                hpp_samples = hpp_full_probe_columns[..., :hpp_num_samples]
                hpp_residuals = hpp_full_residual_columns[..., :hpp_num_samples]
                hutchpp_sampler = SequenceSampler([hpp_samples, hpp_residuals])
                hutchpp = HutchPlusPlusEstimator(
                    num_matvecs=num_matvecs,
                    sampler=hutchpp_sampler,
                ).to(device=device, dtype=dtype)(fn, x)
            curves["xtrace"].append(((xtrace - expected).abs() / denom).mean())
            curves["hutch"].append(((hutch - expected).abs() / denom).mean())
            curves["hutch++"].append(((hutchpp - expected).abs() / denom).mean())

        return {name: torch.stack(values).cpu() for name, values in curves.items()}

    def assert_and_plot_curves(
        self,
        curves: dict[str, Tensor],
        *,
        device: str,
        title: str,
        stem: str,
    ) -> None:
        result_dir = RESULT_DIR
        result_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        markers = {
            "xtrace": "s",
            "hutch": "^",
            "hutch++": "D",
        }
        for name, curve in curves.items():
            finite = torch.isfinite(curve)
            ax.plot(
                np.asarray(self.NUM_MATVECS_GRID)[finite.numpy()],
                curve[finite],
                marker=markers[name],
                label=name,
            )
            assert finite.any()

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("num_matvecs")
        ax.set_ylabel("mean relative error")
        ax.set_title(title)
        ax.legend()

        out = result_dir / f"{stem}_{device}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        assert out.exists()

    @torch.no_grad()
    def test_diagonal(self, device: str) -> None:
        torch.manual_seed(0)
        scale = 0.5 + torch.rand(
            self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=self.DTYPE
        )
        curves = self.compute_curves(lambda x: scale * x, scale.sum(-1), device=device)
        self.assert_and_plot_curves(
            curves,
            device=device,
            title=(
                f"Diagonal trace estimation "
                f"({device}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_diagonal",
        )

    @torch.no_grad()
    def test_gaussian(self, device: str) -> None:
        torch.manual_seed(0)
        matrix = torch.randn(
            self.BATCH_SIZE,
            self.INPUT_SIZE,
            self.INPUT_SIZE,
            device=device,
            dtype=self.DTYPE,
        ) / (self.INPUT_SIZE**0.5)
        curves = self.compute_curves(
            lambda x: torch.einsum("...ij, ...j -> ...i", matrix, x),
            expected=torch.einsum("...ii -> ...", matrix),
            device=device,
        )
        self.assert_and_plot_curves(
            curves,
            device=device,
            title=(
                f"Gaussian trace estimation "
                f"({device}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_gaussian",
        )

    @torch.no_grad()
    def test_linear_spectrum(self, device: str) -> None:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        u_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        v_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        u = torch.from_numpy(u_numpy).to(device=device, dtype=self.DTYPE)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=self.DTYPE)
        spectrum = torch.linspace(
            0, 2, self.INPUT_SIZE, device=device, dtype=self.DTYPE
        ).expand(self.BATCH_SIZE, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", u, spectrum, v)
        curves = self.compute_curves(
            lambda x: torch.einsum("...ij, ...j -> ...i", matrix, x),
            expected=torch.einsum("...ii -> ...", matrix),
            device=device,
        )
        self.assert_and_plot_curves(
            curves,
            device=device,
            title=(
                f"Linear-spectrum trace estimation "
                f"({device}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_linear_spectrum",
        )

    @torch.no_grad()
    def test_exponential_spectrum(self, device: str) -> None:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        u_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        v_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        u = torch.from_numpy(u_numpy).to(device=device, dtype=self.DTYPE)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=self.DTYPE)
        spectrum = (
            1.25
            ** torch.arange(
                -(self.INPUT_SIZE // 2),
                (self.INPUT_SIZE + 1) // 2,
                device=device,
                dtype=self.DTYPE,
            )
        ).expand(self.BATCH_SIZE, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", u, spectrum, v)
        curves = self.compute_curves(
            lambda x: torch.einsum("...ij, ...j -> ...i", matrix, x),
            expected=torch.einsum("...ii -> ...", matrix),
            device=device,
        )
        self.assert_and_plot_curves(
            curves,
            device=device,
            title=(
                f"Exponential-spectrum trace estimation "
                f"({device}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_exponential_spectrum",
        )

    @torch.no_grad()
    def test_low_rank(self, device: str) -> None:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        u_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        v_numpy = ortho_group(self.INPUT_SIZE).rvs(
            size=self.BATCH_SIZE, random_state=rng
        )
        u = torch.from_numpy(u_numpy).to(device=device, dtype=self.DTYPE)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=self.DTYPE)
        rank = self.INPUT_SIZE // 16
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=self.DTYPE),
                torch.zeros(self.INPUT_SIZE - rank, device=device, dtype=self.DTYPE),
            ]
        ).expand(self.BATCH_SIZE, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", u, spectrum, v)
        curves = self.compute_curves(
            lambda x: torch.einsum("...ij, ...j -> ...i", matrix, x),
            expected=torch.einsum("...ii -> ...", matrix),
            device=device,
        )
        self.assert_and_plot_curves(
            curves,
            device=device,
            title=(
                f"Low-rank trace estimation "
                f"({device}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_low_rank",
        )
