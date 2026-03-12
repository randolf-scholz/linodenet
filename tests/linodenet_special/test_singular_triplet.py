from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy
import torch
from numpy.random import default_rng
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

import linodenet_special
from linodenet_special import singular_triplet, singular_triplet_native
from tests.utils import timer

from .fixtures import DEVICES, SEEDS


def random_rank_one_matrix(m: int, n: int) -> np.ndarray:
    rng = default_rng(seed=0)
    ustar = rng.standard_normal(m)
    vstar = rng.standard_normal(n)
    return np.outer(ustar, vstar)


def inner(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute the inner product."""
    return torch.einsum("..., ... ->", x, y)


def compute_singular_triplet_impl(
    impl: Callable[..., tuple[Tensor, Tensor, Tensor]],
    shape: tuple[int, int],
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    r"""Test the spectral norm implementation."""
    m, n = shape
    A0 = torch.randn(m, n)

    # outer gradients
    g_s = torch.randn(())
    g_u = torch.randn(m)
    g_v = torch.randn(n)

    # Native Forward
    A_native = nn.Parameter(A0.clone())
    s_native, u_native, v_native = singular_triplet_native(A_native)

    # Native Backward
    r_native = inner(g_s, s_native) + inner(g_u, u_native) + inner(g_v, v_native)
    r_native.backward()

    # Native Gradient
    assert A_native.grad is not None
    g_native = A_native.grad.clone().detach()

    # Custom Forward
    A_custom = nn.Parameter(A0.clone())
    s_custom, u_custom, v_custom = impl(A_custom, **kwargs)

    # Custom Backward
    r_custom = inner(g_s, s_custom) + inner(g_u, u_custom) + inner(g_v, v_custom)
    r_custom.backward()

    # Custom Gradient
    assert A_custom.grad is not None
    g_custom = A_custom.grad.clone().detach()

    # Compute the errors
    err_value = (s_custom - s_native).norm() / s_native.norm()
    err_grads = (g_custom - g_native).norm() / g_native.norm()

    return err_value, err_grads


class TestBasic:
    RANK_ONE_SHAPES: list[tuple[int, int]] = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 16),
        (1, 64),
        (1, 256),
        (2, 1),
        (4, 1),
        (16, 1),
        (64, 1),
        (256, 1),
    ]
    SHAPES: list[tuple[int, int]] = [
        # square matrices
        (2, 2),
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        # rectangular matrices
        (16, 64),
        (256, 64),
    ]
    DIMS = [2, 4, 16, 64, 256]
    ATOL = 1e-3
    RTOL = 1e-5

    SVD_METHODS: dict[str, Callable] = {
        "numpy_svd": np.linalg.svd,
        "scipy_svd": scipy.linalg.svd,
        "torch_svd": torch.linalg.svd,
        "linodenet_svd": singular_triplet,
    }

    @pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{seed=}")
    @pytest.mark.parametrize(
        ("atol", "rtol"), [pytest.param(ATOL, RTOL, id=f"atol={ATOL},rtol={RTOL}")]
    )
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape=}")
    @pytest.mark.parametrize("method", SVD_METHODS)
    def test_svd_rank_one(
        self,
        method: str,
        *,
        shape: tuple[int, int],
        seed: int,
        atol: float,
        rtol: float,
    ) -> None:
        r"""Checks that the singular triplet method works for rank one matrices."""
        torch.manual_seed(seed)
        m, n = shape
        matrix = random_rank_one_matrix(m, n)

        match self.SVD_METHODS[method]:
            case scipy.linalg.svd:
                A = matrix
                U, S, Vh = scipy.linalg.svd(A)
                # cols of U = LSV, rows of Vh: RSV
                u, s, v = U[:, 0], S[0], Vh[0, :]
                assert np.allclose(s * np.outer(u, v), A, atol=atol, rtol=rtol)
            case np.linalg.svd:
                A = matrix
                U, S, Vh = np.linalg.svd(A)
                # cols of U = LSV, rows of Vh: RSV
                u, s, v = U[:, 0], S[0], Vh[0, :]
                assert np.allclose(s * np.outer(u, v), A, atol=atol, rtol=rtol)
            case torch.linalg.svd:
                B = torch.from_numpy(matrix)
                U, S, Vh = torch.linalg.svd(B)
                # cols of U = LSV, rows of Vh: RSV
                u, s, v = U[:, 0], S[0], Vh[0, :]
                assert torch.allclose(s * torch.outer(u, v), B, atol=atol, rtol=rtol)
            case linodenet_special.singular_triplet:
                B = torch.from_numpy(matrix)
                s, u, v = singular_triplet(B)
                assert torch.allclose(s * torch.outer(u, v), B, atol=atol, rtol=rtol)
            case _:
                raise ValueError(f"Unknown method: {method}")

    @pytest.mark.xfail(reason="Matrices badly conditioned.")
    @pytest.mark.parametrize("value_tol", [1e-5])
    @pytest.mark.parametrize("grads_tol", [1e-3])
    def test_grad(self, value_tol: float, grads_tol: float) -> None:
        r"""Test the singular triplet."""
        err_vals = []
        err_grad = []
        torch.manual_seed(0)
        for _ in range(100):
            m, n = 32, 32
            err_value, err_grads = compute_singular_triplet_impl(
                singular_triplet, (m, n)
            )
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

    @pytest.mark.skip(reason="Expensive and covered by other tests.")
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda x: f"{x[0]}x{x[1]}")
    def test_value_and_grad(self, device: str, shape: tuple[int, int]) -> None:
        r"""Test the singular triplet implementation."""
        m, n = shape
        A0 = torch.randn(m, n, device=device)
        xi = torch.randn(1, device=device)
        phi = torch.randn(m, device=device)
        psi = torch.randn(n, device=device)
        cond = torch.linalg.cond(A0)

        # Native Forward
        A_native = nn.Parameter(A0.clone())
        with timer() as time:
            s_native, u_native, v_native = singular_triplet_native(A_native)
        time_val_native = time.elapsed_time

        # Native Backward
        with timer() as time:
            r_native = xi * s_native + phi.dot(u_native) + psi.dot(v_native)
            r_native.backward()
            assert A_native.grad is not None
            g_native = A_native.grad.clone().detach()
        time_grad_native = time.elapsed_time

        # Custom Forward
        A_custom = nn.Parameter(A0.clone())
        with timer() as time:
            s_custom, u_custom, v_custom = singular_triplet(A_custom)
        time_val_custom = time.elapsed_time

        # Adjust signs so they match
        if (u_custom - u_native).norm() > (u_custom + u_native).norm():
            u_custom = -1 * u_custom
            v_custom = -1 * v_custom

        # Custom Backward
        with timer() as time:
            # NOTE: We flip signs if appropriate to match the native implementation
            r_custom = xi * s_custom + phi.dot(u_custom) + psi.dot(v_custom)
            r_custom.backward()
            assert A_custom.grad is not None
            g_custom = A_custom.grad.clone().detach()
        time_grad_custom = time.elapsed_time

        err_s = (s_custom - s_native).norm() / s_native.norm()
        err_u = (u_custom - u_native).norm() / u_native.norm()
        err_v = (v_custom - v_native).norm() / v_native.norm()
        err_grads = (g_custom - g_native).norm() / g_native.norm()

        print(f"{shape=}  {device=}  {cond=}")
        print(
            f"ERRORS: value: {err_s:.4%}/{err_u:.4%}/{err_v:.4%}, grad: {err_grads:.4%}"
        )
        print(
            f"TIME (native): value: {time_val_native:.4f}, grad: {time_grad_native:.4f}"
        )
        print(
            f"TIME (custom): value: {time_val_custom:.4f}, grad: {time_grad_custom:.4f}"
        )
        print(
            (g_native - xi * torch.outer(u_native, v_native)).norm() / g_native.norm()
        )
        print(
            (g_custom - xi * torch.outer(u_custom, v_custom)).norm() / g_custom.norm()
        )

        assert s_custom > 0, "Singular value is non-positive."
        assert err_s < 1e-4, "Large error in spectral norm value"
        assert err_u < 1e-3, "Large error in left singular vector"
        assert err_v < 1e-3, "Large error in right singular vector"
        assert err_grads < 1e-2, "Large error in spectral norm gradient"
        assert time_grad_custom < 1.2 * time_grad_native, "Custom backward is too slow"


class TestPerformance:
    SINGULAR_TRIPLETS = {
        "custom": singular_triplet,
        "native": singular_triplet_native,
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
    def make_test_case(
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

    @pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SINGULAR_TRIPLETS)
    @pytest.mark.benchmark(group="singular_triplet_forward")
    def test_forward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test the spectral norm implementation."""
        benchmark.group = f"singular_triplet_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SINGULAR_TRIPLETS[name]

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.make_test_case(shape, device=device, generator=generator)

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native, u_native, v_native = singular_triplet_native(A_native)
        dyadic_native = torch.outer(u_native, v_native)

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom, u_custom, v_custom = impl(A_custom)
        dyadic_custom = torch.outer(u_custom, v_custom)

        # check correctness
        assert (s_custom - s_native).norm() < self.ATOL + self.RTOL * s_native.norm(), (
            "Large error in singular value!"
        )
        assert (
            dyadic_custom - dyadic_native
        ).norm() < self.ATOL + self.RTOL * dyadic_native.norm(), (
            "Large error in singular vectors!"
        )

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            param = self.make_test_case(shape, device=device, generator=generator)
            return (param,), {}

        # perform benchmark
        with torch.no_grad():
            benchmark.pedantic(
                impl,
                setup=setup,
                rounds=self.ROUNDS[shape[0]],
                warmup_rounds=self.ROUNDS[shape[0]] // 4,
            )

    @pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SINGULAR_TRIPLETS)
    @pytest.mark.benchmark(group="singular_triplet_backward")
    def test_plain_backward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test simplified backward when only singular value used."""
        benchmark.group = f"singular_triplet_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SINGULAR_TRIPLETS[name]

        torch.manual_seed(0)
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.make_test_case(shape, device=device, generator=generator)
        g_s = torch.randn((), device=device, generator=generator)

        def backward(s: Tensor, /) -> None:
            loss = g_s * s
            loss.backward()

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native, _, _ = singular_triplet_native(A_native)
        backward(s_native)
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom, _, _ = impl(A_custom)
        backward(s_custom)
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()

        # check correctness
        assert (g_custom - g_native).norm() < self.ATOL + self.RTOL * g_native.norm(), (
            "Large error in spectral norm gradient!"
        )

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            param = self.make_test_case(shape, device=device, generator=generator)
            s, _, _ = impl(param)
            return (s,), {}

        # perform benchmark
        benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)

    @pytest.mark.parametrize("shape", [(256, 256)], ids=lambda x: f"{x[0]}x{x[1]}")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", SINGULAR_TRIPLETS)
    @pytest.mark.benchmark(group="singular_triplet_full_backward")
    def test_full_backward(
        self,
        benchmark: BenchmarkFixture,
        name: str,
        *,
        device: str,
        shape: tuple[int, int],
    ) -> None:
        r"""Test full backward when singular triplet used."""
        benchmark.group = f"singular_triplet_forward/{shape[0]}x{shape[1]}/{device}"
        impl = self.SINGULAR_TRIPLETS[name]

        torch.manual_seed(0)
        m, n = shape
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        A_original = self.make_test_case(shape, device=device, generator=generator)
        g_s = torch.randn((), device=device)
        g_u = torch.randn(m, device=device)
        g_v = torch.randn(n, device=device)

        def backward(s: Tensor, u: Tensor, v: Tensor, /) -> None:
            loss = g_s * s + g_u.dot(u) + g_v.dot(v)
            loss.backward()

        # get reference gradient
        A_native = nn.Parameter(A_original.clone().detach())
        s_native, u_native, v_native = singular_triplet_native(A_native)
        backward(s_native, u_native, v_native)
        assert A_native.grad is not None
        g_native = A_native.grad.clone().detach()

        # get custom gradient
        A_custom = nn.Parameter(A_original.clone().detach())
        s_custom, u_custom, v_custom = impl(A_custom)
        backward(s_custom, u_custom, v_custom)
        assert A_custom.grad is not None
        g_custom = A_custom.grad.clone().detach()

        # check correctness
        residual = g_custom - g_native
        assert residual.norm() < self.ATOL + self.RTOL * g_native.norm(), (
            "Large error in spectral norm gradient!"
        )

        def setup() -> tuple[tuple, dict]:  # get args and kwargs for benchmark
            param = self.make_test_case(shape, device=device, generator=generator)
            s, u, v = impl(param)
            return (s, u, v), {}

        # perform benchmark
        benchmark.pedantic(backward, setup=setup, rounds=512, warmup_rounds=128)
