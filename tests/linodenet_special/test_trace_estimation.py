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
        return TraceCase(name="diagonal", matrix=matrix, spectrum=spectrum)

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
        return TraceCase(name="ldu", matrix=matrix, spectrum=spectrum)

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
        return TraceCase(name="symmetric", matrix=matrix, spectrum=spectrum)

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
        return TraceCase(name="skew_symmetric", matrix=matrix, spectrum=spectrum)

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
        return TraceCase(name="linear_spectrum", matrix=matrix, spectrum=spectrum)

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
        return TraceCase(name="exponential_spectrum", matrix=matrix, spectrum=spectrum)

    def make_decaying_contraction(
        self,
        /,
        *,
        q: float,
        seed: int | None = None,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> TraceCase:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        generator = self._make_generator(seed=seed, device=device)

        if not 0.0 < q < 1.0:
            raise ValueError(f"q must satisfy 0 < q < 1, got {q!r}")

        basis = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        exponents = torch.arange(1, input_size + 1, device=device, dtype=torch.float32)
        magnitudes = torch.as_tensor(q, device=device, dtype=torch.float32).pow(
            exponents
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
        spectrum = signs.to(dtype=torch.float32) * magnitudes
        spectrum = spectrum.to(device=device, dtype=dtype)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", basis, spectrum, basis)
        return TraceCase(
            name="decaying_spectrum_contraction",
            matrix=matrix,
            spectrum=spectrum,
        )

    def make_low_rank(
        self,
        /,
        *,
        rank: int,
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
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=dtype),
                torch.zeros(input_size - rank, device=device, dtype=dtype),
            ]
        ).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        return TraceCase(name="low_rank", matrix=matrix, spectrum=spectrum)

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
        dtype: torch.dtype | None = None,
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
            dtype=dtype,
            generator=generator,
        )
        q, _ = torch.linalg.qr(gaussian)
        return q

    def assert_trace_close(
        self,
        name: str,
        test_case: TraceCase,
        /,
        *,
        atol: float,
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
        norms = matrix_norm(matrix, ord="nuc")
        eps = torch.finfo(norms.dtype).eps
        scaled_rmse = (errors / norms.clamp_min(eps)).square().mean().sqrt()
        mean_relative_error = (errors.abs() / expected.abs().clamp_min(eps)).mean()

        if debug:
            print(
                f"{name=:8s} "
                f"scaled_rmse={scaled_rmse.item():.4e} "
                f"mean_relative_error={mean_relative_error.item():.4e}"
            )
            return

        self.assert_upper_bounded(scaled_rmse, 0.0, atol=atol, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestExactTrace(TestTraceEstimator):
    BATCH_SIZE = 4
    INPUT_SIZE = 32
    MAX_POWER = 4

    PROBLEM_RANK = 8
    DECAY_Q = 0.95

    def test_exact_trace_matches_known_spectrum(self, device: str) -> None:
        test_case = self.make_ldu(
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

        self.assert_close(estimate, test_case.trace)

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

        self.assert_close(value, torch.zeros_like(x))
        self.assert_close(estimate, test_case.logabsdet)


@pytest.mark.parametrize("device", DEVICES, ids=str)
class TestTraceCorrectness(TestTraceEstimator):
    BATCH_SIZE = 32
    PROBLEM_SIZE = 128
    PROBLEM_RANK = 6
    NUM_MATVECS = 16
    SEED = 0
    TOLERANCES: dict[tuple[str, str], float] = {
        ("diagonal", TraceEstimators.EXACT): 1e-5,
        ("diagonal", TraceEstimators.HUTCH): 2e-2,
        ("diagonal", TraceEstimators.HUTCH_PP): 3e-2,
        ("diagonal", TraceEstimators.XTRACE): 2e-2,
        ("ldu", TraceEstimators.EXACT): 1e-5,
        ("ldu", TraceEstimators.HUTCH): 6e-2,
        ("ldu", TraceEstimators.HUTCH_PP): 7e-2,
        ("ldu", TraceEstimators.XTRACE): 5e-2,
        ("low_rank", TraceEstimators.EXACT): 1e-5,
        ("low_rank", TraceEstimators.HUTCH): 2.5e-1,
        ("low_rank", TraceEstimators.HUTCH_PP): 1e-1,
        ("low_rank", TraceEstimators.XTRACE): 1e-4,
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
            atol=self.TOLERANCES["diagonal", name],
            num_matvecs=self.NUM_MATVECS,
            device=device,
        )

    @pytest.mark.parametrize("name", TraceEstimators, ids=str)
    def test_ldu(self, name: str, device: str) -> None:
        test_case = self.make_ldu(
            batch_size=self.BATCH_SIZE,
            input_size=self.PROBLEM_SIZE,
            device=device,
        )
        self.assert_trace_close(
            name,
            test_case,
            atol=self.TOLERANCES["ldu", name],
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
            atol=self.TOLERANCES["low_rank", name],
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
                "ldu",
                self.make_ldu(
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
                atol=self.TOLERANCES[label, name],
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

    TOLERANCES: dict[tuple[str, str], float] = {
        ("low_rank", L.EXACT): 3e-7,
        ("low_rank", L.HUTCH): 2e-1,
        ("low_rank", L.HUTCH_PP): 2e-1,
        ("flat_spectrum", L.EXACT): 7e-6,
        ("flat_spectrum", L.HUTCH): 6e0,
        ("flat_spectrum", L.HUTCH_PP): 2e1,
        ("decaying_spectrum", L.EXACT): 4e-7,
        ("decaying_spectrum", L.HUTCH): 3e-1,
        ("decaying_spectrum", L.HUTCH_PP): 4e-1,
    }

    def assert_logabsdet_close(
        self,
        method: str,
        test_case: TraceCase,
        /,
        *,
        atol: float,
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
        rmse = ((estimate - expected) / expected).abs().mean()

        if debug:
            # scaling: see XTrace paper.
            # Var[tr] = E[|tr - tr(A)|²] ≤ η‖A‖⁎² ⇝ scaled_rmse ≤ η
            # reminder: ‖A‖⁎ = nuclear norm = sum of singular values.
            nuc_norms = matrix_norm(A, ord="nuc")
            scale = 10 ** math.floor(math.log10(abs(rmse)))
            bound = math.ceil(rmse / scale) * scale
            print(
                f"{test_case.name:32s} {method=:8s}  "
                f"{rmse=:.1e} (<{bound:.0e}) "
                f"‖A‖⁎={nuc_norms.mean():.1e}"
            )
            return

        self.assert_upper_bounded(rmse, 0.0, atol=atol, rtol=0.0)

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
            atol=self.TOLERANCES["low_rank", name],
            device=device,
        )

    @pytest.mark.parametrize("name", L, ids=str)
    def test_flat_contraction(self, name: str, device: str) -> None:
        test_case = self.make_contraction(
            self.make_ldu(input_size=self.PROBLEM_SIZE, device=device),
            c=self.DECAY_Q,
        )
        self.assert_logabsdet_close(
            name,
            test_case,
            atol=self.TOLERANCES["flat_spectrum", name],
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
            atol=self.TOLERANCES["decaying_spectrum", name],
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
                    self.make_ldu(input_size=self.PROBLEM_SIZE, device=device),
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
                atol=self.TOLERANCES[label, method],
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
