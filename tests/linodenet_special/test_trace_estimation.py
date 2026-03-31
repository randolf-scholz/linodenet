import itertools
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from linodenet_special.trace_estimation import (
    HutchinsonEstimator,
    LogAbsDetEstimators,
    LogabsdetSeriesEstimator,
    TraceEstimators,
)
from tests.testing import DEVICES, PROJECT, TestSuite

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@dataclass(frozen=True)
class TraceCase:
    r"""Test matrix with known spectral data."""

    matrix: Tensor
    spectrum: Tensor

    @cached_property
    def trace(self) -> Tensor:
        trace = self.spectrum.sum(dim=-1)
        return trace.real if trace.is_complex() else trace

    @cached_property
    def logabsdet(self) -> Tensor:
        return torch.log(torch.abs(1 + self.spectrum)).sum(dim=-1)

    def powers(self, k: int, /) -> Iterator[Tensor]:
        for degree in range(1, k + 1):
            trace = self.spectrum.pow(degree).sum(dim=-1)
            yield trace.real if trace.is_complex() else trace


class TestTraceEstimator(TestSuite):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    SEED = 0

    def _make_generator(
        self,
        /,
        *,
        seed: int | None,
        device: str | torch.device,
    ) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.SEED if seed is None else seed)
        return generator

    def make_diagonal(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        spectrum = 0.5 + torch.rand(
            batch_size,
            input_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        matrix = torch.diag_embed(spectrum)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_ldu(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        spectrum = (
            torch.randn(
                batch_size,
                input_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / input_size**0.5
        )
        lower = (
            torch.randn(
                batch_size,
                input_size,
                input_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / input_size**0.5
        )
        lower = torch.tril(lower, diagonal=-1) + torch.eye(
            input_size,
            device=device,
            dtype=dtype,
        )
        upper = (
            torch.randn(
                batch_size,
                input_size,
                input_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / input_size**0.5
        )
        upper = torch.triu(upper, diagonal=1) + torch.eye(
            input_size,
            device=device,
            dtype=dtype,
        )
        scale = torch.exp(
            0.1
            * torch.randn(
                batch_size,
                input_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        basis = lower @ torch.diag_embed(scale) @ upper
        diagonal = torch.diag_embed(spectrum)
        matrix = torch.linalg.solve(
            basis.mT,
            torch.einsum("...ij, ...jk -> ...ik", basis, diagonal).mT,
        ).mT
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        spectrum = (
            torch.randn(
                batch_size,
                input_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / input_size**0.5
        )
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_skew_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        num_blocks = input_size // 2
        frequencies = 0.5 + torch.rand(
            batch_size,
            num_blocks,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        canonical = torch.zeros(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        indices = torch.arange(num_blocks, device=device)
        canonical[..., 2 * indices, 2 * indices + 1] = frequencies
        canonical[..., 2 * indices + 1, 2 * indices] = -frequencies
        matrix = torch.einsum("...ik, ...kl, ...jl -> ...ij", q, canonical, q)

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        spectrum = torch.zeros(
            batch_size, input_size, device=device, dtype=complex_dtype
        )
        spectrum[..., 2 * indices] = 1j * frequencies.to(dtype=complex_dtype)
        spectrum[..., 2 * indices + 1] = -1j * frequencies.to(dtype=complex_dtype)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_linear_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        spectrum = torch.linspace(0, 2, input_size, device=device, dtype=dtype)
        spectrum = spectrum.expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_exponential_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        exponents = torch.arange(
            -(input_size // 2),
            (input_size + 1) // 2,
            device=device,
            dtype=dtype,
        )
        spectrum = (1.25**exponents).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_low_rank(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        rank = math.ceil(math.cbrt(input_size))
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=dtype),
                torch.zeros(input_size - rank, device=device, dtype=dtype),
            ]
        ).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(matrix=matrix, spectrum=spectrum)

    def make_contraction(
        self,
        test_case: TraceCase,
        /,
        *,
        c: float = 0.97,
    ) -> TraceCase:
        max_spectral_radius = test_case.spectrum.abs().amax(dim=-1, keepdim=True)
        one = torch.ones_like(max_spectral_radius)
        eps = torch.finfo(test_case.matrix.dtype).eps
        scale = torch.minimum(one, c / max_spectral_radius.clamp_min(eps))
        return TraceCase(
            matrix=test_case.matrix * scale.unsqueeze(-1),
            spectrum=test_case.spectrum * scale,
        )

    def make_low_rank_contraction(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        c: float = 0.97,
    ) -> TraceCase:
        test_case = self.make_low_rank(
            seed=seed,
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        return self.make_contraction(test_case, c=c)

    def _make_orthogonal_batch(
        self,
        /,
        *,
        batch_size: int,
        input_size: int,
        dtype: torch.dtype | None,
        device: str | torch.device,
        generator: torch.Generator,
    ) -> Tensor:
        gaussian = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        q, _ = torch.linalg.qr(gaussian)
        return q


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestTraceCorrectness(TestTraceEstimator):
    pass


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestPowersCorrectness(TestTraceEstimator):
    pass


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestLogAbsDetCorrectness(TestTraceEstimator):
    BATCH_SIZE = 4
    PROBLEM_SIZE = 128
    NUM_MATVECS = 16
    NUM_TERMS = 8
    SEED = 0
    TOLERANCES: dict[str, float] = {
        LogAbsDetEstimators.EXACT: 1e-5,
        LogAbsDetEstimators.HUTCH: 1e-1,
        LogAbsDetEstimators.HUTCH_PP: 5e-2,
    }

    def assert_logabsdet_close(
        self, name: str, test_case: TraceCase, /, *, device: str
    ) -> None:
        torch.manual_seed(self.SEED)

        estimator = LogAbsDetEstimators.new(
            name,
            num_matvecs=self.NUM_MATVECS,
            num_terms=self.NUM_TERMS,
            sampler="sphere",
            mode="symmetric",
        ).to(device=device)

        x = torch.zeros(
            self.BATCH_SIZE,
            self.PROBLEM_SIZE,
            device=device,
        )
        A = test_case.matrix
        output = estimator(lambda z: torch.einsum("...ji, ...j -> ...i", A, z), x)
        estimate = output[1] if isinstance(output, tuple) else output

        expected = test_case.logabsdet
        mean_relative_error = ((estimate - expected) / expected).abs().mean()
        atol = self.TOLERANCES[name]
        self.assert_upper_bounded(mean_relative_error, 0.0, atol=atol, rtol=0.0)

    @pytest.mark.parametrize("name", LogAbsDetEstimators, ids=str)
    def test_low_rank_contraction(self, name: str, device: str) -> None:
        test_case = self.make_low_rank_contraction(
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_logabsdet_close(name, test_case, device=device)

    @pytest.mark.parametrize("name", LogAbsDetEstimators, ids=str)
    def test_dense_contraction(self, name: str, device: str) -> None:
        test_case = self.make_contraction(
            self.make_symmetric(input_size=self.PROBLEM_SIZE, device=device)
        )
        self.assert_logabsdet_close(name, test_case, device=device)


def test_trace_estimator_accepts_hutchinson_alias() -> None:
    estimator = TraceEstimators.new(
        "hutchinson",
        num_matvecs=4,
        mode="reverse",
        sampler="sphere",
    )
    assert isinstance(estimator, HutchinsonEstimator)


def test_logabsdet_estimator_accepts_hutchinson_alias() -> None:
    estimator = LogAbsDetEstimators.new(
        "hutchinson",
        num_matvecs=4,
        num_terms=3,
        sampler="sphere",
        mode="reverse",
    )
    assert isinstance(estimator, LogabsdetSeriesEstimator)
    assert isinstance(estimator.estimator, HutchinsonEstimator)


class TestVisualizations(TestTraceEstimator):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    DEVICE = "cpu"
    NUM_MATVECS_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    METHODS: dict[str, tuple[str, str, str, dict]] = {
        "hutch": ("hutch", "reverse", "sphere", {}),
        "hutch++": ("hutch++", "reverse", "sphere", {}),
        "xtrace": ("xtrace", "reverse", "sphere", {}),
        # "hutch(gauss)": ("hutch", "forward", "gaussian", {}),
        # "hutch++(gauss)": ("hutch++", "forward", "gaussian", {}),
        # "xtrace(gauss)": ("xtrace", "forward", "gaussian", {}),
        # "hutch(reverse)": ("hutch", "reverse", "orth", {}),
        # "hutch(symmetric)": ("hutch", "symmetric", "orth", {}),
        # "hutch++(reverse)": ("hutch++", "reverse", "orth", {}),
        # "hutch++(symmetric)": ("hutch++", "symmetric", "orth", {}),
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
        device = self.DEVICE
        eps = torch.finfo(expected.dtype).eps
        denom = expected.abs().clamp_min(eps)
        x = torch.zeros(batch_size, input_size, device=device)

        curves: dict[str, list[Tensor]] = {test_id: [] for test_id in self.METHODS}

        batched_mm = torch.compile(
            lambda z: torch.einsum("...ji, ...j -> ...i", matrix, z)
        )

        for test_id, (name, mode, sampler, kwargs) in self.METHODS.items():
            for num_matvecs in self.NUM_MATVECS_GRID:
                torch.manual_seed(self.SEED)
                try:
                    estimator = TraceEstimators.new(
                        name,
                        num_matvecs=num_matvecs,
                        mode=mode,
                        sampler=sampler,
                        **kwargs,
                    ).to(device=device)
                    estimate = estimator(batched_mm, x)
                except ValueError:
                    estimate = torch.full((), torch.nan, device=device)
                except NotImplementedError:
                    estimate = torch.full((), torch.nan, device=device)
                curves[test_id].append(((estimate - expected).abs() / denom).mean())

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
        colors = {"hutch": "C0", "hutch++": "C1", "xtrace": "C2"}
        markers = {"hutch": "^", "hutch++": "D", "xtrace": "s"}
        linestyles: defaultdict[str, itertools.cycle[str]] = defaultdict(
            lambda: itertools.cycle(["-", "--", ":", "-."])
        )
        for test_id, curve in curves.items():
            (
                name,
                *_,
            ) = self.METHODS[test_id]
            ax.plot(
                self.NUM_MATVECS_GRID,
                curve,
                color=colors[name],
                marker=markers[name],
                linestyle=next(linestyles[name]),
                label=test_id,
            )

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
        test_case = self.make_diagonal(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Diagonal trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_diagonal",
        )

    @torch.no_grad()
    def test_ldu(self) -> None:
        test_case = self.make_ldu(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"LDU trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_ldu",
        )

    @torch.no_grad()
    def test_linear_spectrum(self) -> None:
        test_case = self.make_linear_spectrum(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
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
        test_case = self.make_exponential_spectrum(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
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
        test_case = self.make_low_rank(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Low-rank trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_low_rank",
        )
