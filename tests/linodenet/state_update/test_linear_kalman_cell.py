r"""Tests for linear Kalman-style state updaters."""

import pytest
import torch

from linodenet.state_update import LinearKalmanCell


def test_linear_kalman_cell_matches_masked_scalar_blup_by_default() -> None:
    r"""The default update should match the masked closed-form BLUP with scalar noise."""
    cell = LinearKalmanCell(3, 3, gate="identity")
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[4.0, float("nan"), 9.0]])

    with torch.no_grad():
        cell.observation_map.weight.copy_(torch.eye(3))
        cell.correlation_cholesky.zero_()
        cell.correlation_cholesky.diagonal().copy_(torch.tensor([2.0, 3.0, 4.0]))
        cell.noise_cholesky.copy_(torch.sqrt(torch.tensor(3.0)) * torch.eye(3))

    expected_correction = torch.tensor(
        [
            [
                (4.0 / 7.0) * (4.0 - 1.0),
                0.0,
                (16.0 / 19.0) * (9.0 - 3.0),
            ]
        ]
    )
    expected = x + expected_correction

    torch.testing.assert_close(cell(y, x), expected)


def test_linear_kalman_cell_matches_general_masked_formula_with_diagonal_noise() -> (
    None
):
    r"""The update should match the masked LMMSE formula with diagonal noise."""
    cell = LinearKalmanCell(3, 2, noise="diagonal", gate="identity")
    x = torch.tensor([[1.0, -1.0]])
    y = torch.tensor([[2.0, float("nan"), 0.5]])

    H = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    sigma_xx = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    noise = torch.diag(torch.tensor([9.0, 16.0, 25.0]))
    observed = torch.tensor([0, 2])

    with torch.no_grad():
        cell.observation_map.weight.copy_(H)
        cell.correlation_cholesky.copy_(torch.linalg.cholesky(sigma_xx))
        cell.noise_cholesky.copy_(torch.diag(torch.tensor([3.0, 4.0, 5.0])))

    y_pred = x @ H.mT
    innovation = (y - y_pred)[0, observed]
    system = H @ sigma_xx @ H.mT + noise
    expected_correction = (
        sigma_xx
        @ H.mT[:, observed]
        @ torch.linalg.solve(system[observed][:, observed], innovation)
    )
    expected = x + expected_correction

    torch.testing.assert_close(cell(y, x), expected)


def test_linear_kalman_cell_rejects_dense_noise() -> None:
    r"""Dense observation noise is currently unsupported."""
    with pytest.raises(ValueError, match="Expected 'scalar' or 'diagonal'"):
        LinearKalmanCell(3, 2, noise="dense")


def test_linear_kalman_cell_ignores_fully_missing_observations() -> None:
    r"""If no coordinates are observed, the state should remain unchanged."""
    cell = LinearKalmanCell(4, 6, gate="identity")
    x = torch.randn(5, 6)
    y = torch.full((5, 4), float("nan"))

    torch.testing.assert_close(cell(y, x), x)


def test_linear_kalman_cell_gate_variants() -> None:
    r"""Identity-like gates should match and ReZero should start at zero correction."""
    none_gate = LinearKalmanCell(3, 3, gate=None)
    identity_gate = LinearKalmanCell(3, 3, gate="identity")
    rezero_gate = LinearKalmanCell(3, 3)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[4.0, float("nan"), 9.0]])

    with torch.no_grad():
        for cell in (none_gate, identity_gate, rezero_gate):
            cell.observation_map.weight.copy_(torch.eye(3))
            cell.correlation_cholesky.zero_()
            cell.correlation_cholesky.diagonal().copy_(torch.tensor([2.0, 3.0, 4.0]))
            cell.noise_cholesky.copy_(torch.sqrt(torch.tensor(3.0)) * torch.eye(3))

    torch.testing.assert_close(none_gate(y, x), identity_gate(y, x))
    torch.testing.assert_close(rezero_gate(y, x), x)


def test_linear_kalman_cell_masked_backward_has_finite_gradients() -> None:
    r"""Masked observations should not introduce NaNs into gradients."""
    torch.manual_seed(0)

    cell = LinearKalmanCell(5, 7, noise="diagonal", gate="identity")
    x = torch.randn(8, 7, requires_grad=True)
    y = torch.randn(8, 5)
    mask = torch.rand(8, 5) < 0.5
    y = y.masked_fill(mask, float("nan"))

    output = cell(y, x)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in cell.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
