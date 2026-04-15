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
from torch.linalg import matrix_norm

from linodenet_special.trace_estimation import (
    ExactTrace,
    HutchinsonEstimator,
    LogAbsDetEstimators as L,
    LogabsdetSeriesEstimator,
    TraceEstimators,
)
from tests.testing import DEVICES, PREFER_GPU, PROJECT, TestSuite

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]
type Tolerance = float


def test_trace_estimator_accepts_hutchinson_alias() -> None:
    estimator = TraceEstimators.new(
        "hutchinson",
        num_matvecs=4,
        mode="reverse",
        sampler="sphere",
    )
    assert isinstance(estimator, HutchinsonEstimator)


def test_logabsdet_estimator_accepts_hutchinson_alias() -> None:
    estimator = L.new(
        "hutchinson",
        num_matvecs=4,
        num_terms=3,
        sampler="sphere",
        mode="reverse",
    )
    assert isinstance(estimator, LogabsdetSeriesEstimator)
    assert isinstance(estimator.estimator, HutchinsonEstimator)


@dataclass(frozen=True)
class TraceCase:
    r"""Test matrix with known spectral data."""

    name: str
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
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        spectrum = 0.5 + torch.rand(
            batch_size,
            input_size,
            device=device,
            dtype=work_dtype,
            generator=generator,
        )
        matrix = torch.diag_embed(spectrum)
        return TraceCase(
            name="diagonal",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_normal(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64
        scale = 1 / math.sqrt(input_size)
        spectrum = scale * torch.randn(
            batch_size,
            input_size,
            device=device,
            dtype=work_dtype,
            generator=generator,
        )
        basis = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", basis, spectrum, basis)
        return TraceCase(
            name="normal",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64
        scale = 1 / math.sqrt(input_size)

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        spectrum = scale * torch.randn(
            batch_size,
            input_size,
            device=device,
            dtype=work_dtype,
            generator=generator,
        )
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(
            name="symmetric",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_skew_symmetric(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        num_blocks = input_size // 2
        frequencies = 0.5 + torch.rand(
            batch_size,
            num_blocks,
            device=device,
            dtype=work_dtype,
            generator=generator,
        )
        canonical = torch.zeros(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=work_dtype,
        )
        indices = torch.arange(num_blocks, device=device)
        canonical[..., 2 * indices, 2 * indices + 1] = frequencies
        canonical[..., 2 * indices + 1, 2 * indices] = -frequencies
        matrix = torch.einsum("...ik, ...kl, ...jl -> ...ij", q, canonical, q)

        spectrum = torch.zeros(
            batch_size, input_size, device=device, dtype=torch.complex128
        )
        spectrum[..., 2 * indices] = 1j * frequencies.to(dtype=torch.complex128)
        spectrum[..., 2 * indices + 1] = -1j * frequencies.to(dtype=torch.complex128)
        return TraceCase(
            name="skew_symmetric",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(
                dtype=torch.complex64 if dtype == torch.float32 else torch.complex128
            ),
        )

    def make_linear_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        spectrum = torch.linspace(0, 2, input_size, device=device, dtype=work_dtype)
        spectrum = spectrum.expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(
            name="linear_spectrum",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_exponential_spectrum(
        self,
        /,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        exponents = torch.arange(
            -(input_size // 2),
            (input_size + 1) // 2,
            device=device,
            dtype=work_dtype,
        )
        spectrum = (1.25**exponents).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(
            name="exponential_spectrum",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_decaying_contraction(
        self,
        /,
        *,
        q: float,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        if not 0.0 < q < 1.0:
            raise ValueError(f"q must satisfy 0 < q < 1, got {q!r}")

        basis = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        magnitudes = q ** (
            1 + torch.arange(input_size, device=device, dtype=work_dtype)
        )
        signs = (
            2
            * torch.randint(
                0,
                2,
                (batch_size, input_size),
                device=device,
                generator=generator,
            )
            - 1
        )
        spectrum = signs.to(dtype=work_dtype) * magnitudes
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", basis, spectrum, basis)
        return TraceCase(
            name="decaying_spectrum_contraction",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_low_rank(
        self,
        /,
        *,
        rank: int,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)
        work_dtype = torch.float64

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=work_dtype,
            device=device,
            generator=generator,
        )
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=work_dtype),
                torch.zeros(input_size - rank, device=device, dtype=work_dtype),
            ]
        ).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(
            name="low_rank",
            matrix=matrix.to(dtype=dtype),
            spectrum=spectrum.to(dtype=dtype),
        )

    def make_contraction(
        self,
        test_case: TraceCase,
        /,
        *,
        c: float,
    ) -> TraceCase:
        max_spectral_radius = test_case.spectrum.abs().amax(dim=-1, keepdim=True)
        one = torch.ones_like(max_spectral_radius)
        eps = torch.finfo(test_case.matrix.dtype).eps
        scale = torch.minimum(one, c / max_spectral_radius.clamp_min(eps))
        return TraceCase(
            name=f"{test_case.name}_contraction",
            matrix=test_case.matrix * scale.unsqueeze(-1),
            spectrum=test_case.spectrum * scale,
        )

    def make_low_rank_contraction(
        self,
        /,
        *,
        rank: int,
        c: float,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        test_case = self.make_low_rank(
            rank=rank,
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
            dtype=torch.float64,
            generator=generator,
        )
        q, _ = torch.linalg.qr(gaussian)
        return q.to(dtype=dtype)

    def assert_trace_close(
        self,
        name: str,
        test_case: TraceCase,
        /,
        *,
        eta: Tolerance,
        num_matvecs: int,
        device: str,
        debug: bool = False,
    ) -> None:
        torch.manual_seed(self.SEED)

        estimator = TraceEstimators.new(
            name,
            num_matvecs=num_matvecs,
            mode="reverse",
            sampler="sphere",
        ).to(device=device)

        x = torch.zeros(
            test_case.matrix.shape[0],
            test_case.matrix.shape[-1],
            device=device,
            dtype=test_case.matrix.dtype,
        )
        matrix = test_case.matrix
        estimate = estimator(
            lambda z: torch.einsum("...ji, ...j -> ...i", matrix, z),
            x,
        )

        expected = test_case.trace
        errors = estimate - expected
        nuc_norm = matrix_norm(matrix, ord="nuc").mean()
        eps = torch.finfo(errors.dtype).eps
        rmse = errors.square().mean().sqrt()
        calibrated_eta = rmse / nuc_norm.clamp_min(eps)
        mean_relative_error = (errors.abs() / expected.abs().clamp_min(eps)).mean()

        if debug:
            print(
                f"{name=:8s} "
                f"rmse={rmse.item():.4e} "
                f"eta={calibrated_eta.item():.4e} "
                f"‖A‖⁎={nuc_norm.item():.4e} "
                f"mean_relative_error={mean_relative_error.item():.4e}"
            )
            return

        self.assert_magnitude_bounded(rmse, nuc_norm, scale=eta)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestExactTrace(TestTraceEstimator):
    BATCH_SIZE = 4
    INPUT_SIZE = 32
    MAX_POWER = 4

    PROBLEM_RANK = 8
    DECAY_Q = 0.95

    def test_exact_trace_matches_known_spectrum(self, device: str) -> None:
        test_case = self.make_normal(
            batch_size=self.BATCH_SIZE,
            input_size=self.INPUT_SIZE,
            device=device,
        )
        estimator = ExactTrace().to(device=device)
        x = torch.zeros(
            self.BATCH_SIZE,
            self.INPUT_SIZE,
            device=device,
            dtype=test_case.matrix.dtype,
        )

        estimate = estimator(
            lambda z: torch.einsum("...ji, ...j -> ...i", test_case.matrix, z),
            x,
        )

        self.assert_close(estimate, test_case.trace, atol=1e-6, rtol=1e-6)

    def test_exact_trace_powers_match_known_spectrum(self, device: str) -> None:
        test_case = self.make_skew_symmetric(
            batch_size=self.BATCH_SIZE,
            input_size=self.INPUT_SIZE,
            dtype=torch.float32,
            device=device,
        )
        estimator = ExactTrace().to(device=device)
        x = torch.zeros(
            self.BATCH_SIZE,
            self.INPUT_SIZE,
            device=device,
            dtype=test_case.matrix.dtype,
        )

        estimates = estimator.powers(
            lambda z: torch.einsum("...ji, ...j -> ...i", test_case.matrix, z),
            x,
            self.MAX_POWER,
        )
        expected = test_case.powers(self.MAX_POWER)

        for estimate, truth in zip(estimates, expected, strict=True):
            self.assert_close(estimate, truth, atol=1e-4, rtol=1e-5)

    def test_exact_trace_logabsdet_matches_closed_form(self, device: str) -> None:
        test_case = self.make_low_rank_contraction(
            rank=self.PROBLEM_RANK,
            c=self.DECAY_Q,
            batch_size=self.BATCH_SIZE,
            input_size=self.INPUT_SIZE,
            device=device,
        )
        estimator = ExactTrace().to(device=device)
        x = torch.zeros(
            self.BATCH_SIZE,
            self.INPUT_SIZE,
            device=device,
            dtype=test_case.matrix.dtype,
        )

        value, estimate = estimator.logabsdet(
            lambda z: torch.einsum("...ji, ...j -> ...i", test_case.matrix, z),
            x,
        )

        self.assert_close(value, torch.zeros_like(x), atol=1e-6, rtol=1e-6)
        self.assert_close(estimate, test_case.logabsdet, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestTraceCorrectness(TestTraceEstimator):
    BATCH_SIZE = 32
    PROBLEM_SIZE = 128
    PROBLEM_RANK = 6
    NUM_MATVECS = 16
    SEED = 0
    ETAS: dict[tuple[str, str], Tolerance] = {
        ("diagonal", TraceEstimators.EXACT): 1e-7,
        ("diagonal", TraceEstimators.HUTCH): 1e-1,
        ("diagonal", TraceEstimators.HUTCH_PP): 1e-1,
        ("diagonal", TraceEstimators.XTRACE): 1e-1,
        ("normal", TraceEstimators.EXACT): 1e-7,
        ("normal", TraceEstimators.HUTCH): 1e-1,
        ("normal", TraceEstimators.HUTCH_PP): 1e-1,
        ("normal", TraceEstimators.XTRACE): 1e-1,
        ("low_rank", TraceEstimators.EXACT): 1e-7,
        ("low_rank", TraceEstimators.HUTCH): 2e-1,
        ("low_rank", TraceEstimators.HUTCH_PP): 2e-1,
        ("low_rank", TraceEstimators.XTRACE): 1e-6,
    }

    @pytest.mark.parametrize("name", TraceEstimators, ids=str)
    def test_diagonal(self, name: str, device: str) -> None:
        test_case = self.make_diagonal(
            batch_size=self.BATCH_SIZE,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_trace_close(
            name,
            test_case,
            eta=self.ETAS["diagonal", name],
            num_matvecs=self.NUM_MATVECS,
            device=device,
        )

    @pytest.mark.parametrize("name", TraceEstimators, ids=str)
    def test_normal(self, name: str, device: str) -> None:
        test_case = self.make_normal(
            batch_size=self.BATCH_SIZE,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_trace_close(
            name,
            test_case,
            eta=self.ETAS["normal", name],
            num_matvecs=self.NUM_MATVECS,
            device=device,
        )

    @pytest.mark.parametrize("name", TraceEstimators, ids=str)
    def test_low_rank(self, name: str, device: str) -> None:
        test_case = self.make_low_rank(
            rank=self.PROBLEM_RANK,
            batch_size=self.BATCH_SIZE,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_trace_close(
            name,
            test_case,
            eta=self.ETAS["low_rank", name],
            num_matvecs=self.NUM_MATVECS,
            device=device,
        )

    @pytest.mark.parametrize("name", TraceEstimators, ids=str)
    def test_calibration(self, name: str, device: str) -> None:
        print()
        for label, test_case in (
            (
                "diagonal",
                self.make_diagonal(
                    batch_size=self.BATCH_SIZE,
                    input_size=self.PROBLEM_SIZE,
                    device=device,
                ),
            ),
            (
                "normal",
                self.make_normal(
                    batch_size=self.BATCH_SIZE,
                    input_size=self.PROBLEM_SIZE,
                    device=device,
                ),
            ),
            (
                "low_rank",
                self.make_low_rank(
                    rank=self.PROBLEM_RANK,
                    batch_size=self.BATCH_SIZE,
                    input_size=self.PROBLEM_SIZE,
                    device=device,
                ),
            ),
        ):
            print(f"{device=}, {label=}")
            self.assert_trace_close(
                name,
                test_case,
                eta=self.ETAS[label, name],
                num_matvecs=self.NUM_MATVECS,
                device=device,
                debug=True,
            )


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestPowersCorrectness(TestTraceEstimator):
    pass


@pytest.mark.parametrize("device", PREFER_GPU, ids=str)
class TestLogAbsDetCorrectness(TestTraceEstimator):
    BATCH_SIZE = 32
    PROBLEM_SIZE = 256
    PROBLEM_RANK = 8
    DECAY_Q = 0.95
    NUM_MATVECS = 16
    NUM_TERMS = 10
    SEED = 0

    ETAS: dict[tuple[str, str], Tolerance] = {
        ("low_rank", L.EXACT): 1e-6,
        ("low_rank", L.HUTCH): 1e-1,
        ("low_rank", L.HUTCH_PP): 1e-1,
        ("flat_spectrum", L.EXACT): 1e-6,
        ("flat_spectrum", L.HUTCH): 5e-2,
        ("flat_spectrum", L.HUTCH_PP): 5e-2,
        ("decaying_spectrum", L.EXACT): 1e-6,
        ("decaying_spectrum", L.HUTCH): 1e-1,
        ("decaying_spectrum", L.HUTCH_PP): 2e-1,
    }

    def assert_logabsdet_close(
        self,
        method: str,
        test_case: TraceCase,
        /,
        *,
        eta: Tolerance,
        device: str,
        debug: bool = False,
    ) -> None:
        torch.manual_seed(self.SEED)

        estimator = L.new(
            method,
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
        errors = estimate - expected
        nuc_norm = matrix_norm(A, ord="nuc").mean()
        eps = torch.finfo(errors.dtype).eps
        rmse = errors.square().mean().sqrt()
        calibrated_eta = rmse / nuc_norm.clamp_min(eps)

        if debug:
            # Calibration target: choose η so rmse ≤ η‖A‖⁎.
            eta_value = calibrated_eta.item()
            scale = 10 ** math.floor(math.log10(abs(eta_value)))
            bound = math.ceil(eta_value / scale) * scale
            print(
                f"{test_case.name:32s} {method=:8s}  "
                f"rmse={rmse.item():.1e}  "
                f"eta={eta_value:.1e} (<{bound:.0e}) "
                f"‖A‖⁎={nuc_norm.item():.1e}"
            )
            return

        self.assert_upper_bounded(rmse, eta * nuc_norm)

    @pytest.mark.parametrize("name", L, ids=str)
    def test_low_rank_contraction(self, name: str, device: str) -> None:
        test_case = self.make_low_rank_contraction(
            rank=self.PROBLEM_RANK,
            c=self.DECAY_Q,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_logabsdet_close(
            name,
            test_case,
            eta=self.ETAS["low_rank", name],
            device=device,
        )

    @pytest.mark.parametrize("name", L, ids=str)
    def test_flat_contraction(self, name: str, device: str) -> None:
        test_case = self.make_contraction(
            self.make_normal(input_size=self.PROBLEM_SIZE, device=device),
            c=self.DECAY_Q,
        )
        self.assert_logabsdet_close(
            name,
            test_case,
            eta=self.ETAS["flat_spectrum", name],
            device=device,
        )

    @pytest.mark.parametrize("name", L, ids=str)
    def test_decaying_contraction(self, name: str, device: str) -> None:
        test_case = self.make_decaying_contraction(
            q=self.DECAY_Q,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_logabsdet_close(
            name,
            test_case,
            eta=self.ETAS["decaying_spectrum", name],
            device=device,
        )

    @pytest.mark.parametrize("method", L)
    def test_calibration(self, method: str, device: str) -> None:
        print()
        for label, test_case in (
            (
                "low_rank",
                self.make_low_rank_contraction(
                    rank=self.PROBLEM_RANK,
                    c=self.DECAY_Q,
                    input_size=self.PROBLEM_SIZE,
                    device=device,
                ),
            ),
            (
                "flat_spectrum",
                self.make_contraction(
                    self.make_normal(input_size=self.PROBLEM_SIZE, device=device),
                    c=self.DECAY_Q,
                ),
            ),
            (
                "decaying_spectrum",
                self.make_decaying_contraction(
                    q=self.DECAY_Q,
                    input_size=self.PROBLEM_SIZE,
                    device=device,
                ),
            ),
        ):
            self.assert_logabsdet_close(
                method,
                test_case,
                eta=self.ETAS[label, method],
                device=device,
                debug=True,
            )


class TestVisualizations(TestTraceEstimator):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    LOW_RANK = 7
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
    def test_normal(self) -> None:
        test_case = self.make_normal(device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Normal trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_normal",
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
        test_case = self.make_low_rank(rank=self.LOW_RANK, device=self.DEVICE)
        curves = self.compute_curves(test_case.matrix, test_case.trace)
        self.assert_and_plot_curves(
            curves,
            title=(
                f"Low-rank trace estimation "
                f"({self.DEVICE}, batch={self.BATCH_SIZE}, input={self.INPUT_SIZE})"
            ),
            stem="trace_estimation_low_rank",
        )
