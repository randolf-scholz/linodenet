r"""Tests for linear state propagation flows."""

from typing import NamedTuple

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from linodenet.state_propagation import (
    linear_flow,
    linear_gaussian_flow,
)


class LinearSystem(NamedTuple):
    A: torch.Tensor
    b: torch.Tensor | None
    x0: torch.Tensor
    timedeltas: torch.Tensor


class LinearGaussianSystem(NamedTuple):
    A: torch.Tensor
    b: torch.Tensor | None
    Q: torch.Tensor
    mu0: torch.Tensor
    sigma0: torch.Tensor
    timedeltas: torch.Tensor


class TestLinear:
    BATCH_SIZE = 8
    NUM_STEPS = 10
    SEEDS = [0, 1, 2, 3, 4]

    @classmethod
    def _make_system(
        cls,
        seed: int,
        *,
        with_bias: bool,
        batched: bool,
        dim: int = 4,
        num_steps: int | None = None,
    ) -> LinearSystem:
        num_steps = cls.NUM_STEPS if num_steps is None else num_steps
        assert num_steps >= 2
        rng = np.random.default_rng(seed)
        A = torch.tensor(
            rng.normal(size=(dim, dim)) / np.sqrt(dim),
            dtype=torch.float64,
        )
        b = (
            torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
            if with_bias
            else None
        )
        if batched:
            x0 = torch.tensor(
                rng.normal(size=(cls.BATCH_SIZE, dim)),
                dtype=torch.float64,
            )
            timedeltas = torch.tensor(
                np.stack([
                    np.unique([0.0, *rng.uniform(0.0, 1.0, size=num_steps - 2), 2.0])
                    for _ in range(cls.BATCH_SIZE)
                ]),
                dtype=torch.float64,
            )  # fmt: skip
        else:
            x0 = torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
            timedeltas = torch.tensor(
                # note: unique also sorts
                np.unique([0.0, *rng.uniform(0.0, 1.0, size=num_steps - 2), 2.0]),
                dtype=torch.float64,
            )
        return LinearSystem(A, b, x0, timedeltas)

    @staticmethod
    def _solve_ivp(
        A: torch.Tensor,
        b: torch.Tensor | None,
        x0: torch.Tensor,
        timedeltas: torch.Tensor,
    ) -> torch.Tensor:
        if timedeltas.ndim == 2:
            return torch.stack(
                [
                    TestLinear._solve_ivp(A, b, x0_i, timedeltas_i)
                    for x0_i, timedeltas_i in zip(x0, timedeltas, strict=True)
                ]
            )

        A_np = A.numpy()
        b_np = None if b is None else b.numpy()
        x0_np = x0.numpy()
        t_eval = timedeltas.numpy()

        def func(_: float, x: np.ndarray) -> np.ndarray:
            return A_np @ x if b_np is None else A_np @ x + b_np

        sol = solve_ivp(
            func,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=x0_np,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        assert sol.success, sol.message
        return torch.tensor(sol.y.T, dtype=torch.float64)

    @pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
    @pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
    def test_flow_matches_solve_ivp_fp64(
        self, seed: int, with_bias: bool, batched: bool
    ) -> None:
        case = self._make_system(seed, with_bias=with_bias, batched=batched)
        A = case.A
        b = case.b
        x0 = case.x0
        timedeltas = case.timedeltas

        actual = linear_flow(
            timedeltas.to(dtype=torch.float32),
            x0.to(dtype=torch.float32),
            A.to(dtype=torch.float32),
            None if b is None else b.to(dtype=torch.float32),
        ).to(dtype=torch.float64)
        expected = self._solve_ivp(A, b, x0, timedeltas)

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


class TestLinearGaussian:
    BATCH_SIZE = 8
    NUM_STEPS = 10
    SEEDS = [0, 1, 2, 3, 4]

    @classmethod
    def _make_system(
        cls,
        seed: int,
        *,
        with_bias: bool,
        batched: bool,
        dim: int = 4,
        num_steps: int | None = None,
    ) -> LinearGaussianSystem:
        num_steps = cls.NUM_STEPS if num_steps is None else num_steps
        assert num_steps >= 2
        rng = np.random.default_rng(seed)
        A = torch.tensor(
            rng.normal(size=(dim, dim)) / np.sqrt(dim),
            dtype=torch.float64,
        )
        b = (
            torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
            if with_bias
            else None
        )
        mu0 = torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)

        sigma_factor = rng.normal(size=(dim, dim)) / np.sqrt(dim)
        sigma0 = torch.tensor(
            sigma_factor @ sigma_factor.T + 0.1 * np.eye(dim),
            dtype=torch.float64,
        )

        q_factor = rng.normal(size=(dim, dim)) / np.sqrt(dim)
        Q = torch.tensor(
            q_factor @ q_factor.T + 0.1 * np.eye(dim),
            dtype=torch.float64,
        )

        if batched:
            mu0 = torch.tensor(
                rng.normal(size=(cls.BATCH_SIZE, dim)),
                dtype=torch.float64,
            )
            sigma_factors = rng.normal(size=(cls.BATCH_SIZE, dim, dim)) / np.sqrt(dim)
            sigma0 = torch.tensor(
                sigma_factors @ np.swapaxes(sigma_factors, -1, -2) + 0.1 * np.eye(dim),
                dtype=torch.float64,
            )
            timedeltas = torch.tensor(
                np.stack([
                    np.unique([0.0, *rng.uniform(0.0, 1.0, size=num_steps - 2), 1.5])
                    for _ in range(cls.BATCH_SIZE)
                ]),
                dtype=torch.float64,
            )  # fmt: skip
        else:
            timedeltas = torch.tensor(
                # note: unique also sorts
                np.unique([0.0, *rng.uniform(0.0, 1.0, size=num_steps - 2), 1.5]),
                dtype=torch.float64,
            )
        return LinearGaussianSystem(A, b, Q, mu0, sigma0, timedeltas)

    @staticmethod
    def _solve_moment_ivp(
        A: torch.Tensor,
        b: torch.Tensor | None,
        Q: torch.Tensor,
        mu0: torch.Tensor,
        sigma0: torch.Tensor,
        timedeltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timedeltas.ndim == 2:
            expected = [
                TestLinearGaussian._solve_moment_ivp(
                    A, b, Q, mu0_i, sigma0_i, timedeltas_i
                )
                for mu0_i, sigma0_i, timedeltas_i in zip(
                    mu0, sigma0, timedeltas, strict=True
                )
            ]
            expected_mu, expected_sigma = zip(*expected, strict=True)
            return torch.stack(list(expected_mu)), torch.stack(list(expected_sigma))

        A_np = A.numpy()
        b_np = None if b is None else b.numpy()
        Q_np = Q.numpy()
        mu0_np = mu0.numpy()
        sigma0_np = sigma0.numpy()
        dim = A.shape[-1]
        t_eval = timedeltas.numpy()

        def func(_: float, state: np.ndarray) -> np.ndarray:
            mu = state[:dim]
            sigma = state[dim:].reshape(dim, dim)
            dmu = A_np @ mu if b_np is None else A_np @ mu + b_np
            dsigma = A_np @ sigma + sigma @ A_np.T + Q_np
            return np.concatenate((dmu, dsigma.reshape(-1)))

        initial_state = np.concatenate((mu0_np, sigma0_np.reshape(-1)))
        sol = solve_ivp(
            func,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=initial_state,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        assert sol.success, sol.message

        expected_mu = torch.tensor(sol.y[:dim].T, dtype=torch.float64)
        expected_sigma = torch.tensor(
            sol.y[dim:].T.reshape(len(t_eval), dim, dim),
            dtype=torch.float64,
        )
        return expected_mu, expected_sigma

    @pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
    @pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
    def test_flow_matches_solve_ivp_fp64(
        self, seed: int, with_bias: bool, batched: bool
    ) -> None:
        case = self._make_system(seed, with_bias=with_bias, batched=batched)
        A = case.A
        b = case.b
        Q = case.Q
        mu0 = case.mu0
        sigma0 = case.sigma0
        timedeltas = case.timedeltas

        actual_mu_tensor, actual_sigma_tensor = linear_gaussian_flow(
            timedeltas.to(dtype=torch.float32),
            (mu0.to(dtype=torch.float32), sigma0.to(dtype=torch.float32)),
            A.to(dtype=torch.float32),
            Q.to(dtype=torch.float32),
            None if b is None else b.to(dtype=torch.float32),
        )
        actual_mu_tensor = actual_mu_tensor.to(dtype=torch.float64)
        actual_sigma_tensor = actual_sigma_tensor.to(dtype=torch.float64)

        expected_mu, expected_sigma = self._solve_moment_ivp(
            A, b, Q, mu0, sigma0, timedeltas
        )

        expected_sigma = 0.5 * (expected_sigma + expected_sigma.transpose(-1, -2))
        torch.testing.assert_close(
            actual_sigma_tensor,
            actual_sigma_tensor.transpose(-1, -2),
            atol=5e-6,
            rtol=5e-6,
        )

        eigenvalues = torch.linalg.eigvalsh(actual_sigma_tensor)
        assert torch.all(eigenvalues > 1e-5), (
            "Expected propagated covariance to be positive definite, "
            f"got min eigenvalue {eigenvalues.min().item():.3e}."
        )

        torch.testing.assert_close(actual_mu_tensor, expected_mu, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(
            actual_sigma_tensor,
            expected_sigma,
            atol=8e-5,
            rtol=8e-4,
        )
