r"""Tests for linear Kalman-style state updaters."""

import torch

from linodenet.state_update import LinearKalmanCell


def test_linear_kalman_cell_matches_masked_diagonal_blup() -> None:
    r"""The update should match the masked closed-form BLUP in a diagonal setup."""
    cell = LinearKalmanCell(3, 3)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[4.0, float("nan"), 9.0]])

    with torch.no_grad():
        cell.observation_map.weight.copy_(torch.eye(3))
        cell.state_scale.zero_()
        cell.noise_scale.zero_()
        cell.state_scale.diagonal().copy_(torch.tensor([2.0, 3.0, 4.0]))
        cell.noise_scale.diagonal().copy_(torch.tensor([3.0, 4.0, 5.0]))

    expected_correction = torch.tensor(
        [
            [
                (4.0 / 13.0) * (4.0 - 1.0),
                0.0,
                (16.0 / 41.0) * (9.0 - 3.0),
            ]
        ]
    )
    expected = x + expected_correction

    torch.testing.assert_close(cell(y, x), expected)


def test_linear_kalman_cell_matches_general_masked_formula() -> None:
    r"""The update should match the masked LMMSE formula in a nonsquare setup."""
    cell = LinearKalmanCell(3, 2)
    x = torch.tensor([[1.0, -1.0]])
    y = torch.tensor([[2.0, float("nan"), 0.5]])

    H = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    sigma_xx = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    noise = torch.diag(torch.tensor([9.0, 16.0, 25.0]))
    observed = torch.tensor([0, 2])

    with torch.no_grad():
        cell.observation_map.weight.copy_(H)
        cell.state_scale.copy_(torch.linalg.cholesky(sigma_xx))
        cell.noise_scale.copy_(torch.diag(torch.tensor([3.0, 4.0, 5.0])))

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


def test_linear_kalman_cell_ignores_fully_missing_observations() -> None:
    r"""If no coordinates are observed, the state should remain unchanged."""
    cell = LinearKalmanCell(4, 6)
    x = torch.randn(5, 6)
    y = torch.full((5, 4), float("nan"))

    torch.testing.assert_close(cell(y, x), x)
