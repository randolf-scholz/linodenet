r"""Tests for linear Kalman-style state updaters."""

import pytest
import torch
from torch import nan, nn

from linodenet.nn.containers import Constant
from linodenet.state_update import AttentionCovarianceFactor, KalmanCell


class TestKalmanCell:
    def test_matches_masked_scalar_blup_by_default(self) -> None:
        r"""The default update should match the masked closed-form BLUP with scalar noise."""
        cell = KalmanCell(3, 3, gate="identity")
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[4.0, nan, 9.0]])
        mask = torch.tensor([[True, False, True]])

        with torch.no_grad():
            assert isinstance(cell.observation_map, nn.Linear)
            assert isinstance(cell.covariance_factor, Constant)
            cell.observation_map.weight.copy_(torch.eye(3))
            cell.covariance_factor.value.zero_()
            cell.covariance_factor.value.diagonal().copy_(torch.tensor([2.0, 3.0, 4.0]))
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

        torch.testing.assert_close(cell(y, x, mask=mask), expected)

    def test_matches_general_masked_formula_with_diagonal_noise(self) -> None:
        r"""The update should match the masked LMMSE formula with diagonal noise."""
        cell = KalmanCell(3, 2, noise="diagonal", gate="identity")
        x = torch.tensor([[1.0, -1.0]])
        y = torch.tensor([[2.0, nan, 0.5]])
        mask = torch.tensor([[True, False, True]])

        H = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
        sigma_xx = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
        noise = torch.diag(torch.tensor([9.0, 16.0, 25.0]))
        observed = torch.tensor([0, 2])

        with torch.no_grad():
            assert isinstance(cell.observation_map, nn.Linear)
            assert isinstance(cell.covariance_factor, Constant)
            cell.observation_map.weight.copy_(H)
            cell.covariance_factor.value.copy_(torch.linalg.cholesky(sigma_xx))
            cell.noise_cholesky.copy_(torch.diag(torch.tensor([3.0, 4.0, 5.0])))

        y_pred = x @ H.mT
        innovation = (y_pred - y)[0, observed]
        system = H @ sigma_xx @ H.mT + noise
        expected_correction = (
            sigma_xx
            @ H.mT[:, observed]
            @ torch.linalg.solve(system[observed][:, observed], innovation)
        )
        expected = x - expected_correction

        torch.testing.assert_close(cell(y, x, mask=mask), expected)

    def test_rejects_dense_noise(self) -> None:
        r"""Dense observation noise is currently unsupported."""
        with pytest.raises(ValueError, match="Expected 'scalar' or 'diagonal'"):
            KalmanCell(3, 2, noise="dense")

    def test_rejects_unknown_covariance_factor(self) -> None:
        r"""Unknown covariance-factor strings should fail explicitly."""
        with pytest.raises(
            ValueError,
            match=(
                r"Unknown covariance_factor: 'other'. "
                r"Expected 'constant', 'attention', or an nn.Module."
            ),
        ):
            KalmanCell(3, 2, covariance_factor="other")

    def test_identity_observation_map_uses_x_directly(self) -> None:
        r"""Identity observation maps should use the hidden state directly."""
        cell = KalmanCell(3, 3, observation_map="identity", gate="identity")
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[4.0, nan, 9.0]])
        mask = torch.tensor([[True, False, True]])

        with torch.no_grad():
            assert isinstance(cell.covariance_factor, Constant)
            cell.covariance_factor.value.zero_()
            cell.covariance_factor.value.diagonal().copy_(torch.tensor([2.0, 3.0, 4.0]))
            cell.noise_cholesky.copy_(torch.sqrt(torch.tensor(3.0)) * torch.eye(3))

        assert isinstance(cell.observation_map, nn.Identity)

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

        torch.testing.assert_close(cell(y, x, mask=mask), expected)

    def test_accepts_custom_observation_map(self) -> None:
        r"""Custom observation maps should be used verbatim."""
        observation_map = nn.Linear(5, 3, bias=False)
        cell = KalmanCell(3, 5, observation_map=observation_map)

        assert cell.observation_map is observation_map

    def test_accepts_custom_covariance_factor(self) -> None:
        r"""Custom covariance factors should be used verbatim."""
        covariance_factor = nn.Linear(5, 25, bias=False)
        cell = KalmanCell(3, 5, covariance_factor=covariance_factor)

        assert cell.covariance_factor is covariance_factor

    def test_attention_covariance_factor_uses_attention_module(self) -> None:
        r"""The attention covariance-factor option should instantiate the attention module."""
        cell = KalmanCell(3, 5, covariance_factor="attention")
        x = torch.randn(7, 5)

        assert isinstance(cell.covariance_factor, AttentionCovarianceFactor)
        factor = cell.covariance_factor(x)

        assert factor.shape == (7, 5, 5)
        assert torch.isfinite(factor).all()
        torch.testing.assert_close(factor, factor.tril())
        assert torch.all(factor.diagonal(dim1=-2, dim2=-1) > 0)

    def test_supports_batched_attention_like_covariance_factor(self) -> None:
        r"""State-dependent covariance factors may produce a batched lower-triangular $L(x)$."""

        class AttentionFactor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.key = nn.Linear(4, 16, bias=False)
                self.query = nn.Linear(4, 16, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                xk = self.key(x).unflatten(-1, (4, 4))
                xq = self.query(x).unflatten(-1, (4, 4))
                return torch.tril(xk @ xq.mT)

        covariance_factor = AttentionFactor()
        cell = KalmanCell(
            4,
            4,
            covariance_factor=covariance_factor,
            observation_map="identity",
            gate="identity",
        )
        x = torch.randn(6, 4, requires_grad=True)
        y = torch.randn(6, 4)
        mask = torch.rand(6, 4) < 0.6
        y[~mask] = nan

        assert cell.covariance_factor is covariance_factor

        output = cell(y, x, mask=mask)
        loss = output.square().mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        for parameter in covariance_factor.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_attention_covariance_factor_backward_has_finite_gradients(self) -> None:
        r"""Attention-based covariance factors should support stable masked updates."""
        torch.manual_seed(0)

        cell = KalmanCell(4, 6, covariance_factor="attention", gate="identity")
        x = torch.randn(8, 6, requires_grad=True)
        y = torch.randn(8, 4)
        mask = torch.rand(8, 4) < 0.6
        y[~mask] = nan

        output = cell(y, x, mask=mask)
        loss = output.square().mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        for parameter in cell.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_rejects_identity_for_nonsquare_shapes(self) -> None:
        r"""Identity observation maps require matching input and hidden sizes."""
        with pytest.raises(
            ValueError,
            match=r"observation_map='identity' requires input_size == hidden_size!",
        ):
            KalmanCell(3, 5, observation_map="identity")

    def test_ignores_fully_missing_observations(self) -> None:
        r"""If no coordinates are observed, the state should remain unchanged."""
        cell = KalmanCell(4, 6, gate="identity")
        x = torch.randn(5, 6)
        y = torch.zeros(5, 4)
        mask = torch.zeros(5, 4, dtype=torch.bool)
        y[~mask] = nan

        torch.testing.assert_close(cell(y, x, mask=mask), x)

    def test_gate_variants(self) -> None:
        r"""Identity-like gates should match and ReZero should start at zero correction."""
        none_gate = KalmanCell(3, 3, gate=None)
        identity_gate = KalmanCell(3, 3, gate="identity")
        rezero_gate = KalmanCell(3, 3)

        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[4.0, nan, 9.0]])
        mask = torch.tensor([[True, False, True]])

        with torch.no_grad():
            for cell in (none_gate, identity_gate, rezero_gate):
                assert isinstance(cell.observation_map, nn.Linear)
                assert isinstance(cell.covariance_factor, Constant)
                cell.observation_map.weight.copy_(torch.eye(3))
                cell.covariance_factor.value.zero_()
                cell.covariance_factor.value.diagonal().copy_(
                    torch.tensor([2.0, 3.0, 4.0])
                )
                cell.noise_cholesky.copy_(torch.sqrt(torch.tensor(3.0)) * torch.eye(3))

        torch.testing.assert_close(
            none_gate(y, x, mask=mask),
            identity_gate(y, x, mask=mask),
        )
        torch.testing.assert_close(rezero_gate(y, x, mask=mask), x)

    def test_masked_backward_has_finite_gradients(self) -> None:
        r"""Masked observations should not destabilize gradients."""
        torch.manual_seed(0)

        cell = KalmanCell(5, 7, noise="diagonal", gate="identity")
        x = torch.randn(8, 7, requires_grad=True)
        y = torch.randn(8, 5)
        mask = torch.rand(8, 5) < 0.5
        y[~mask] = nan

        output = cell(y, x, mask=mask)
        loss = output.square().mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        for parameter in cell.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
