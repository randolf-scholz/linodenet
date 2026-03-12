r"""Test the spectral norm implementation."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from linodenet_special import (
    SpectralNorm,
    spectral_norm,
    spectral_norm_native,
    spectral_norm_riemann,
)
from tests.utils import timer

from .fixtures import (
    DEVICES,
    SEEDS,
    make_test_case_diagonal,
    make_test_case_quasi_gaussian,
    make_test_case_rank_one,
    make_test_case_repeated_singular_values,
)


def scaled_norm(x: Tensor) -> Tensor:
    r"""Scaled norm of a tensor."""
    return x.pow(2).mean().sqrt()


def inner(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute the inner product."""
    return torch.einsum("..., ... ->", x, y)


def compute_spectral_norm_impl(
    impl: Callable[..., Tensor], shape: tuple[int, int], **kwargs: Any
) -> tuple[Tensor, Tensor]:
    r"""Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n)

    # outer gradients
    g_s = torch.randn(())

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    s_native = spectral_norm_native(A_native)

    # Native Backward
    r_native = inner(g_s, s_native)
    r_native.backward()

    # Native Gradient
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    s_custom = impl(A_custom, **kwargs)

    # Custom Backward
    r_custom = inner(g_s, s_custom)
    r_custom.backward()

    # Custom Gradient
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # Compute the errors
    err_value = (s_custom - s_native).norm() / s_native.norm()
    err_grads = (g_custom - g_native).norm() / g_native.norm()

    return err_value, err_grads


class BasicTest:
    SHAPES = [
        # m > n
        (8, 4),
        (32, 16),
        (128, 64),
        (512, 256),
        # m < n
        (4, 8),
        (16, 32),
        (64, 128),
        (256, 512),
        # m == n
        (8, 8),
        (32, 32),
        (128, 128),
        (512, 512),
    ]

    @pytest.mark.skip(reason="Expensive and covered by other tests.")
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    def test_spectral_norm(self, device: str, shape: tuple[int, int]) -> None:
        r"""Test the spectral norm implementation."""
        m, n = shape
        A0 = torch.randn(m, n, device=device)
        cond = torch.linalg.cond(A0)

        # Native Forward
        A_native = nn.Parameter(A0.clone())
        with timer() as time:
            s_native = spectral_norm_native(A_native)
        time_val_native = time.elapsed_time

        # Native Backward
        with timer() as time:
            s_native.backward()
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()
        time_grad_native = time.elapsed_time

        # Custom Forward
        A_custom = nn.Parameter(A0.clone())
        with timer() as time:
            s_custom = spectral_norm(A_custom)
        time_val_custom = time.elapsed_time

        # Custom Backward
        with timer() as time:
            s_custom.backward()
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()
        time_grad_custom = time.elapsed_time

        err_value = torch.norm(s_custom - s_native) / torch.norm(s_native)
        err_grads = torch.norm(g_custom - g_native) / torch.norm(g_native)
        print(f"{shape=}  {device=}  {cond=}")
        print(f"ERRORS: value: {err_value:.4%}, grad: {err_grads:.4%}")
        print(
            f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}"
        )
        print(
            f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}"
        )

        assert s_custom > 0, "Singular value is non-positive."
        assert err_value < 1e-4, "Large error in spectral norm value"
        assert err_grads < 1e-2, "Large error in spectral norm gradient"
        assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"

    @pytest.mark.parametrize("value_tol", [1e-5])
    @pytest.mark.parametrize("grads_tol", [1e-3])
    def test_spectral_norm_grad(self, value_tol: float, grads_tol: float) -> None:
        r"""Test the spectral norm."""
        M, N, NUM_RUNS, SEED = 16, 16, 100, 0
        torch.manual_seed(SEED)
        err_vals = []
        err_grad = []
        for _ in range(NUM_RUNS):
            err_value, err_grads = compute_spectral_norm_impl(spectral_norm, (M, N))
            err_vals.append(err_value.item())
            err_grad.append(err_grads.item())
        avgerr_vals = sum(err_vals) / len(err_vals)
        avgerr_grad = sum(err_grad) / len(err_grad)
        print(f"Average Error:: {avgerr_vals:.3e}, grad: {avgerr_grad:.3e}")
        assert avgerr_vals < value_tol, (
            f"Value error too large! {avgerr_vals:.3e} > {value_tol=}"
        )
        assert avgerr_grad < grads_tol, (
            f"Grads error too large! {avgerr_grad:.3e} > {grads_tol=}"
        )
        print("All tests passed.")


class TestCorrectness:
    SHAPES: list[tuple[int, int]] = [
        # scalar
        (1, 1),
        # square matrices
        (2, 2),
        (4, 4),
        (16, 16),
        (64, 64),
        (128, 128),
        # rectangular matrices
        (16, 64),
        (128, 64),
        (64, 16),
        (64, 128),
        # rank-1 matrices
        (1, 2),
        (1, 4),
        (1, 16),
        (1, 64),
        (1, 128),
        (2, 1),
        (4, 1),
        (16, 1),
        (64, 1),
        (128, 1),
    ]
    DIMS = [2, 4, 16, 64, 256]
    ATOL = 1e-3
    RTOL = 1e-4

    SPECTRAL_NORMS: dict[str, SpectralNorm] = {
        "custom": spectral_norm,
        "native": spectral_norm_native,
        "riemann": spectral_norm_riemann,
    }

    @pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
    @pytest.mark.parametrize(
        ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL}-rtol={RTOL}")]
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", SPECTRAL_NORMS)
    def test_rank_one(
        self,
        method: str,
        *,
        shape: tuple[int, int],
        seed: int,
        device: str | torch.device,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Test the accuracy of the gradient for rank one matrices.

        The analytical gradient is ∂‖A‖₂/∂A = uvᵀ, where u and v are the singular vectors.
        In particular, for rank one matrices A=uvᵀ, the gradient is ∂‖A‖₂/∂A = uvᵀ/(‖u‖⋅‖v‖).
        """
        impl = self.SPECTRAL_NORMS[method]
        torch.manual_seed(seed)

        case = make_test_case_rank_one(
            shape, dtype=torch.float, device=device, seed=seed
        )
        A = case.value

        # analytical result
        analytical_value = case.spectral_norm
        analytical_grad = case.spectral_norm_gradient

        # check forward pass
        sigma = impl(A)
        assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

        # backward pass
        sigma.backward()

        # check backward pass
        assert A.grad is not None
        assert scaled_norm(A.grad - analytical_grad) < (
            atol + rtol * scaled_norm(analytical_grad)
        ), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
        )
        assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
        )

    @pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
    @pytest.mark.parametrize(
        ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", SPECTRAL_NORMS)
    def test_diagonal(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Checks that the singular triplet method works for diagonal matrices.

        NOTE: builtin SVD seems to have auto-detection for diagonal matrices...
        """
        impl = self.SPECTRAL_NORMS[method]
        torch.manual_seed(seed)

        case = make_test_case_diagonal(
            shape, dtype=torch.float, device=device, seed=seed
        )
        A = case.value

        # analytical result
        analytical_value = case.spectral_norm
        analytical_grad = case.spectral_norm_gradient

        # check forward pass
        sigma = impl(A)
        assert (sigma - analytical_value).norm() < atol + rtol * analytical_value.norm()
        assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

        # backward pass
        sigma.backward()

        # check backward pass
        assert A.grad is not None
        assert scaled_norm(A.grad - analytical_grad) < (
            atol + rtol * scaled_norm(analytical_grad)
        ), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1]:.3e}"
        )
        assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1]:.3e}"
        )

    @pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
    @pytest.mark.parametrize(
        ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"{ATOL=},{RTOL=}")]
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", SPECTRAL_NORMS)
    def test_analytical(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
        atol: float,
        rtol: float,
    ) -> None:
        r"""We test the analytical result for random matrices.

        We randomly sample U, S and V.
        """
        impl = self.SPECTRAL_NORMS[method]
        case = make_test_case_quasi_gaussian(
            shape, dtype=torch.float, device=device, seed=seed
        )
        A = case.value

        # analytical result
        analytical_value = case.spectral_norm
        analytical_grad = case.spectral_norm_gradient

        # check forward pass
        sigma = impl(A)
        assert (sigma - analytical_value).norm() < atol + rtol * analytical_value.norm()
        assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

        # backward pass
        sigma.backward()

        # check backward pass
        assert A.grad is not None
        assert scaled_norm(A.grad - analytical_grad) < (
            atol + rtol * scaled_norm(analytical_grad)
        ), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1:]}"
        )
        assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1:]}"
        )

    @pytest.mark.xfail(reason="Algorithms are unstable for repeated singular values.")
    @pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
    @pytest.mark.parametrize(
        ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", SPECTRAL_NORMS)
    def test_repeated_singular_values(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Tests algorithm against orthogonal matrix.

        Note:
            For repeated singular values, the gradient is no longer well-defined,
            but (I think), any sum  $∑_{j≤i} uᵢvᵢᵀ$ is a subgradient.
            In particular, if A is already orthogonal, then $∂‖A‖₂/∂A = A$.
            Is, in some sense, the largest subgradient.
        """
        impl = self.SPECTRAL_NORMS[method]
        case = make_test_case_repeated_singular_values(
            shape, dtype=torch.float, device=device, seed=seed
        )
        A = case.value

        # analytical result
        analytical_value = case.spectral_norm
        analytical_grad = case.spectral_norm_gradient

        # check forward pass
        sigma = impl(A)
        assert (sigma - analytical_value).norm() < atol + rtol * analytical_value.norm()
        assert torch.allclose(sigma, analytical_value, atol=atol, rtol=rtol)

        # backward pass
        sigma.backward()

        # check backward pass
        assert A.grad is not None
        assert scaled_norm(A.grad - analytical_grad) < (
            atol + rtol * scaled_norm(analytical_grad)
        ), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1]:.3e}"
        )
        assert torch.allclose(A.grad, analytical_grad, atol=atol, rtol=rtol), (
            f"Max element-wise error: {(A.grad - analytical_grad).abs().max():.3e}"
            f"  ‖A‖₂={analytical_value:.3e}"
            f"  κ(A)={torch.linalg.cond(A):.3e}"
            f"  δ(A)={case.S.sort().values.diff()[-1]:.3e}"
        )


class TestPerformance:
    SPECTRAL_NORMS = {
        "custom": spectral_norm,
        "native": spectral_norm_native,
    }

    ATOL = 1e-5
    RTOL = 1e-3
    SHAPES = [
        (64, 64),
        (256, 256),
        (512, 512),
    ]
    ROUNDS = {
        16: 1024,
        32: 512,
        64: 512,
        128: 256,
        256: 256,
        512: 64,
        1024: 64,
    }

    @staticmethod
    def get_param(
        shape: tuple[int, int],
        *,
        device: str | torch.device,
        generator: torch.Generator,
    ) -> nn.Parameter:
        r"""Get a random parameter of shape (m, n)."""
        n = shape[-1]
        A = torch.randn(shape, device=device, generator=generator) / torch.sqrt(
            torch.tensor(n)
        )
        return nn.Parameter(A)

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SPECTRAL_NORMS)
    @pytest.mark.benchmark(group="spectral_norm_forward")
    def test_spectral_norm_forward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm forward pass."""
        benchmark.group = f"spectral_norm_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SPECTRAL_NORMS[name]
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.get_param(shape, device=device, generator=generator)

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native = spectral_norm_native(A_native)

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom = impl(A_custom)

        # check correctness
        residual = s_custom - s_native
        assert residual.norm() < self.ATOL + self.RTOL * s_native.norm(), (
            "Large error in spectral norm value!"
        )

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            param = self.get_param(shape, device=device, generator=generator)
            return (param,), {}

        with torch.no_grad():
            benchmark.pedantic(
                impl,
                setup=setup,
                rounds=self.ROUNDS[shape[0]],
                warmup_rounds=self.ROUNDS[shape[0]] // 4,
            )

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SPECTRAL_NORMS)
    @pytest.mark.benchmark(group="spectral_norm_backward")
    def test_spectral_norm_backward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm backward pass."""
        benchmark.group = f"spectral_norm_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SPECTRAL_NORMS[name]

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.get_param(shape, device=device, generator=generator)
        g_s = torch.randn((), device=device, generator=generator)

        def backward(s: Tensor, /) -> None:
            loss = g_s * s
            loss.backward()

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native = spectral_norm_native(A_native)
        backward(s_native)
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom = impl(A_custom)
        backward(s_custom)
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()

        # check correctness
        residual = g_custom - g_native
        assert residual.norm() < self.ATOL + self.RTOL * g_native.norm(), (
            "Large error in spectral norm gradient!"
        )

        # perform benchmark
        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            param = self.get_param(shape, device=device, generator=generator)
            output = impl(param)
            return (output,), {}

        benchmark.pedantic(
            backward,
            setup=setup,
            rounds=self.ROUNDS[shape[0]],
            warmup_rounds=self.ROUNDS[shape[0]] // 4,
        )

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SPECTRAL_NORMS)
    @pytest.mark.benchmark(group="spectral_norm")
    def test_spectral_norm(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm forward+backward."""
        benchmark.group = f"spectral_norm_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SPECTRAL_NORMS[name]

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.get_param(shape, device=device, generator=generator)
        g_s = torch.randn((), device=device, generator=generator)

        def backward(sigma: Tensor, /) -> None:
            loss = g_s * sigma
            loss.backward()

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native = spectral_norm_native(A_native)
        backward(s_native)
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom = impl(A_custom)
        backward(s_custom)
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()

        # check correctness
        residual = g_custom - g_native
        assert residual.norm() < self.ATOL + self.RTOL * g_native.norm(), (
            "Large error in spectral norm gradient!"
        )

        def func() -> None:
            param = self.get_param(shape, device=device, generator=generator)
            sigma = impl(param)
            loss = g_s * sigma
            loss.backward()

        benchmark.pedantic(
            func,
            rounds=self.ROUNDS[shape[0]],
            warmup_rounds=self.ROUNDS[shape[0]] // 4,
        )
