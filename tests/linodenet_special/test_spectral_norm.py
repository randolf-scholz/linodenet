r"""Test the spectral norm implementation."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from linodenet_special import spectral_norm, spectral_norm_native
from linodenet_special.compiled import spectral_norm as spectral_norm_cpp
from linodenet_special.fallbacks import spectral_norm as spectral_norm_py
from linodenet_special.fallbacks.spectral_norm import (
    State,
    _body_fn as body_fn,
    _cond_fn as cond_fn,
    _spectral_norm_forward_impl,
)
from tests.testing import DEVICES, SEEDS_5, TestCase, timer
from tests.testing.examples import ExampleWithKnownSVD


def test_compile_torch_while() -> None:
    A = torch.randn(8, 4)
    u = torch.randn(8)
    v = torch.randn(4)
    grad_u = A.mv(v)
    grad_v = A.mT.mv(u)
    maxiter = torch.as_tensor(10, device=A.device, dtype=torch.int32)
    atol = torch.as_tensor(1e-6, device=A.device, dtype=A.dtype)
    rtol = torch.as_tensor(1e-6, device=A.device, dtype=A.dtype)

    # test with plain python
    state = State(maxiter, u, v, grad_u, grad_v, A, atol, rtol)
    assert cond_fn(state)
    while cond_fn(state):
        state = body_fn(state)
    assert not cond_fn(state)

    # test with torch.compiled cond_fn, body_fn
    compiled_body_fn = torch.compile(body_fn)
    compiled_cond_fn = torch.compile(cond_fn)
    state = State(maxiter, u, v, grad_u, grad_v, A, atol, rtol)
    assert compiled_cond_fn(state)
    while compiled_cond_fn(state):
        state = compiled_body_fn(state)
    assert not compiled_cond_fn(state)

    # test with torch.while_loop
    state = State(maxiter, u, v, grad_u, grad_v, A, atol, rtol)
    assert cond_fn(state)
    state = torch.while_loop(
        cond_fn=cond_fn,
        body_fn=body_fn,
        carried_inputs=(state,),
    )
    assert not cond_fn(state)

    @torch.compile
    def loop_body(st: State) -> State:
        return torch.while_loop(
            cond_fn=cond_fn,
            body_fn=body_fn,
            carried_inputs=(st,),
        )

    state = State(maxiter, u, v, grad_u, grad_v, A, atol, rtol)
    assert cond_fn(state)
    state = loop_body(state)
    assert not cond_fn(state)

    # check forward_impl
    _spectral_norm_forward_impl(A, u, v, 10, atol, rtol)
    compiled_impl = torch.compile(_spectral_norm_forward_impl)
    compiled_impl(A, u, v, 10, atol, rtol)

    # check the entire spectral norm implementation
    spectral_norm_py(A, maxiter=10)
    compiled_spectral_norm = torch.compile(spectral_norm_py)
    compiled_spectral_norm(A, maxiter=10)


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


CORRECTNESS_SPECTRAL_NORMS = {
    "py+compiled": torch.compile(spectral_norm_py),
    "cpp+compile": torch.compile(spectral_norm_cpp),
    "py": spectral_norm_py,
    "cpp": spectral_norm_cpp,
    "native": spectral_norm_native,
}
CORRECTNESS_SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (4, 4),
    (16, 16),
    (64, 64),
    (128, 128),
    (16, 64),
    (128, 64),
    (64, 16),
    (64, 128),
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
PERFORMANCE_SPECTRAL_NORMS = {
    "py+compiled": torch.compile(spectral_norm_py),
    "cpp+compile": torch.compile(spectral_norm_cpp),
    "svd+compile": torch.compile(spectral_norm_native),
}
PERFORMANCE_SHAPES = [(64, 64), (128, 64), (64, 128)]


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


@pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", CORRECTNESS_SPECTRAL_NORMS)
class TestCorrectness(TestCase):
    SPECTRAL_NORMS = CORRECTNESS_SPECTRAL_NORMS
    SHAPES = CORRECTNESS_SHAPES
    ATOL = 1e-3
    RTOL = 1e-5

    def check_forward_pass(
        self,
        case: ExampleWithKnownSVD,
        sigma: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Check that the spectral norm matches the analytical value."""
        sigma_ref = case.spectral_norm
        assert sigma.isfinite().all()
        assert sigma_ref.isfinite().all()
        assert sigma.shape == sigma_ref.shape
        assert (sigma > 0).all()
        assert (sigma_ref > 0).all()
        self.assert_close(sigma, sigma_ref, atol=atol, rtol=rtol)

    def check_backward_pass(
        self,
        case: ExampleWithKnownSVD,
        sigma: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Check that the backward pass returns a valid spectral-norm subgradient."""
        A = case.value
        sigma.backward()
        assert A.grad is not None
        assert A.grad.isfinite().all()

        grad = A.grad
        analytical_value = case.spectral_norm
        analytical_grad = case.spectral_norm_gradient
        # For the spectral norm, subgradients admit the dual characterization
        #    ∂‖A‖₂ = {G : ‖G‖_* ≤ 1 and ⟨G, A⟩ = ‖A‖₂},
        # where ‖·‖_* is the nuclear norm dual to ‖·‖₂. This avoids checking the
        # global subgradient inequality against all perturbations X explicitly.
        grad_inner = inner(grad, A)
        self.assert_close(grad_inner, analytical_value, atol=atol, rtol=rtol)
        grad_nuclear_norm = torch.linalg.matrix_norm(grad, ord="nuc")
        self.assert_upper_bounded(grad_nuclear_norm, 1.0, atol=atol, rtol=rtol)

        if case.S.shape[-1] >= 2:
            spectral_gap = case.S[..., 0] - case.S[..., 1]
            if spectral_gap <= atol + rtol * analytical_value.abs():
                return

        self.assert_close(grad, analytical_grad, atol=atol, rtol=rtol)

    @pytest.mark.flaky(reruns=3)
    def test_rank_one(
        self,
        method: str,
        *,
        shape: tuple[int, int],
        seed: int,
        device: str | torch.device,
    ) -> None:
        r"""Test the accuracy of the gradient for rank one matrices.

        The analytical gradient is ∂‖A‖₂/∂A = uvᵀ, where u and v are the singular vectors.
        In particular, for rank one matrices A=uvᵀ, the gradient is ∂‖A‖₂/∂A = uvᵀ/(‖u‖⋅‖v‖).
        """
        impl = self.SPECTRAL_NORMS[method]
        torch.manual_seed(seed)

        case = ExampleWithKnownSVD.rank_one(
            shape, dtype=torch.float, device=device, seed=seed
        )
        sigma = impl(case.value)
        self.check_forward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)
        self.check_backward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.flaky(reruns=3)
    def test_diagonal(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        r"""Checks that the singular triplet method works for diagonal matrices.

        NOTE: builtin SVD seems to have auto-detection for diagonal matrices...
        """
        impl = self.SPECTRAL_NORMS[method]
        torch.manual_seed(seed)

        case = ExampleWithKnownSVD.diagonal(
            shape, dtype=torch.float, device=device, seed=seed
        )
        sigma = impl(case.value)
        self.check_forward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)
        self.check_backward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.flaky(reruns=3)
    def test_quasi_gaussian(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        r"""We test the analytical result for random matrices.

        We randomly sample U, S and V.
        """
        impl = self.SPECTRAL_NORMS[method]
        case = ExampleWithKnownSVD.quasi_gaussian(
            shape, dtype=torch.float, device=device, seed=seed
        )
        sigma = impl(case.value)
        self.check_forward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)
        self.check_backward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)

    def test_repeated_singular_values(
        self,
        method: str,
        *,
        device: str | torch.device,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        r"""Tests algorithm against an orthogonal matrix.

        Note:
            For repeated singular values, the gradient is no longer unique.
            We therefore only test that the returned gradient is a valid
            subgradient of the spectral norm.
        """
        impl = self.SPECTRAL_NORMS[method]
        case = ExampleWithKnownSVD.repeated_singular_values(
            shape, dtype=torch.float, device=device, seed=seed
        )
        sigma = impl(case.value)
        self.check_forward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)
        self.check_backward_pass(case, sigma, atol=self.ATOL, rtol=self.RTOL)


@pytest.mark.parametrize("shape", PERFORMANCE_SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", PERFORMANCE_SPECTRAL_NORMS)
class TestPerformance(TestCase):
    SPECTRAL_NORMS = PERFORMANCE_SPECTRAL_NORMS
    SHAPES = PERFORMANCE_SHAPES
    ROUNDS = 64
    WARMUP_ROUNDS = 16

    def make_test_case(
        self,
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

    def test_spectral_norm_forward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm forward pass."""
        benchmark.group = f"spectral_norm_forward/{device}/{shape[0]}x{shape[1]}"
        impl = self.SPECTRAL_NORMS[name]
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            torch.set_float32_matmul_precision("high")
            param = self.make_test_case(shape, device=device, generator=generator)
            return (param,), {}

        with torch.no_grad():
            benchmark.pedantic(
                impl,
                setup=setup,
                rounds=self.ROUNDS,
                warmup_rounds=self.WARMUP_ROUNDS,
            )

    def test_spectral_norm_backward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm backward pass."""
        benchmark.group = f"spectral_norm_backward/{device}/{shape[0]}x{shape[1]}"
        impl = self.SPECTRAL_NORMS[name]

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        g_s = torch.randn((), device=device, generator=generator)

        def backward(s: Tensor, /) -> None:
            loss = g_s * s
            loss.backward()
            torch.cuda.synchronize()

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            torch.set_float32_matmul_precision("high")
            param = self.make_test_case(shape, device=device, generator=generator)
            output = impl(param)
            return (output,), {}

        benchmark.pedantic(
            backward,
            setup=setup,
            rounds=self.ROUNDS,
            warmup_rounds=self.WARMUP_ROUNDS,
        )
