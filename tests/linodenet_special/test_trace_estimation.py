from collections.abc import Callable
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import Tensor

from linodenet_special.trace_estimation import (
    Samplers,
    hutch_pp_estimator,
    hutchinson_estimator,
    xtrace_estimator,
)
from tests.testing import DEVICES, DTYPES, PROJECT, TestSuite

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def linear_map(matrix: Tensor, /) -> Callable[[Tensor], Tensor]:
    def op(x: Tensor, /) -> Tensor:
        return torch.einsum("...ij, ...j -> ...i", matrix, x)

    return op


class TestTraceEstimator(TestSuite):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    DTYPE = torch.float32
    SEED = 0

    def make_diagonal(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        diagonal = 0.5 + torch.rand(batch_size, input_size, device=device, dtype=dtype)
        matrix = torch.diag_embed(diagonal)
        trace = diagonal.sum(dim=-1)
        return matrix, trace

    def make_gaussian(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = (
            torch.randn(
                batch_size,
                input_size,
                input_size,
                device=device,
                dtype=dtype,
            )
            / input_size**0.5
        )
        trace = torch.einsum("...ii -> ...", matrix)
        return matrix, trace

    def make_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        matrix = (matrix + matrix.mT) / (2 * input_size**0.5)
        trace = torch.einsum("...ii -> ...", matrix)
        return matrix, trace

    def make_skew_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        matrix = (matrix - matrix.mT) / (2 * input_size**0.5)
        trace = torch.zeros(batch_size, device=device, dtype=dtype)
        return matrix, trace

    def make_linear_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            seed=seed,
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        spectrum = torch.linspace(0, 2, input_size, device=device, dtype=dtype)
        spectrum = spectrum.expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def make_exponential_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            seed=seed,
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        exponents = torch.arange(
            -(input_size // 2),
            (input_size + 1) // 2,
            device=device,
            dtype=dtype,
        )
        spectrum = (1.25**exponents).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def make_low_rank(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED if seed is None else seed)
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            seed=seed,
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        rank = input_size // 16
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=dtype),
                torch.zeros(input_size - rank, device=device, dtype=dtype),
            ]
        ).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def _make_orthogonal_batch(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        torch.manual_seed(self.SEED if seed is None else seed)
        gaussian = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        q, _ = torch.linalg.qr(gaussian)
        return q


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
class TestTraceCorrectness(TestTraceEstimator):
    pass


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
class TestPowersCorrectness(TestTraceEstimator):
    pass


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
class TestLogAbsDetCorrectness(TestTraceEstimator):
    pass


class TestVisualizations(TestTraceEstimator):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    DTYPE = torch.float32
    DEVICE = "cpu"
    SAMPLER = "sphere"
    NUM_MATVECS_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    METHODS = {
        "xtrace": xtrace_estimator,
        "hutch": hutchinson_estimator,
        "hutch++": hutch_pp_estimator,
    }

    def compute_curves(
        self,
        matrix: Tensor,
        expected: Tensor,
        /,
    ) -> dict[str, Tensor]:
        mpl.use("Agg")
        torch.manual_seed(self.SEED)

        batch_size = self.BATCH_SIZE
        input_size = self.INPUT_SIZE
        dtype = self.DTYPE
        device = self.DEVICE
        denom = expected.abs().clamp_min(torch.finfo(dtype).eps)
        x = torch.zeros(batch_size, input_size, device=device, dtype=dtype)
        op = linear_map(matrix)

        curves: dict[str, list[Tensor]] = {name: [] for name in self.METHODS}

        for num_matvecs in self.NUM_MATVECS_GRID:
            for name, method in self.METHODS.items():
                sampler = Samplers.new(self.SAMPLER)
                torch.manual_seed(self.SEED)
                try:
                    estimate = method(op, x, num_matvecs, sampler=sampler)
                except ValueError:
                    estimate = torch.full((), torch.nan, device=device, dtype=dtype)
                curves[name].append(((estimate - expected).abs() / denom).mean())

        return {name: torch.stack(values).cpu() for name, values in curves.items()}

    def assert_and_plot_curves(
        self,
        curves: dict[str, Tensor],
        /,
        *,
        title: str,
        stem: str,
    ) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
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
        ax.legend(loc="lower left")
        fig.text(
            0.01,
            0.01,
            datetime.now().replace(tzinfo=None).isoformat(timespec="seconds"),
            ha="left",
            va="bottom",
            fontsize=8,
            color="gray",
        )

        out = RESULT_DIR / f"{stem}_{self.DEVICE}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        assert out.exists()

    @torch.no_grad()
    def test_diagonal(self) -> None:
        matrix, expected = self.make_diagonal(dtype=self.DTYPE, device=self.DEVICE)
        curves = self.compute_curves(matrix, expected)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Diagonal trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_diagonal",
        )

    @torch.no_grad()
    def test_gaussian(self) -> None:
        matrix, expected = self.make_gaussian(dtype=self.DTYPE, device=self.DEVICE)
        curves = self.compute_curves(matrix, expected)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Gaussian trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_gaussian",
        )

    @torch.no_grad()
    def test_linear_spectrum(self) -> None:
        matrix, expected = self.make_linear_spectrum(
            dtype=self.DTYPE, device=self.DEVICE
        )
        curves = self.compute_curves(matrix, expected)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Linear-spectrum trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_linear_spectrum",
        )

    @torch.no_grad()
    def test_exponential_spectrum(self) -> None:
        matrix, expected = self.make_exponential_spectrum(
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        curves = self.compute_curves(matrix, expected)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Exponential-spectrum trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_exponential_spectrum",
        )

    @torch.no_grad()
    def test_low_rank(self) -> None:
        matrix, expected = self.make_low_rank(dtype=self.DTYPE, device=self.DEVICE)
        curves = self.compute_curves(matrix, expected)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Low-rank trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_low_rank",
        )
