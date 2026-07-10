r"""Tests for state-update imputers and missing-value wrappers."""

import torch
from torch import Tensor, nn

from linodenet.state_update.base import MissingValueCell
from linodenet.state_update.imputation import (
    CorrelationImputer,
    LinearImputer,
    ZeroImputer,
)


class IdentityDecoder(nn.Module):
    r"""Decoder that returns the hidden state unchanged."""

    output_size: int

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return x


class CaptureImputer(nn.Module):
    r"""Imputer that records the received mask."""

    received_mask: Tensor | None
    fill_value: Tensor

    def __init__(self, fill_value: float = -1.0) -> None:
        super().__init__()
        self.received_mask = None
        self.register_buffer("fill_value", torch.tensor(fill_value))

    def forward(
        self, y_obs: Tensor, x: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        del x
        self.received_mask = mask
        effective_mask = y_obs.isnan() if mask is None else mask
        return torch.where(effective_mask, self.fill_value, y_obs)


class RecorderFilter(nn.Module):
    r"""State updater that records the imputed input and returns the state unchanged."""

    input_size: int
    hidden_size: int
    last_u: Tensor

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_buffer("last_u", torch.empty(()))

    def forward(self, u: Tensor, x: Tensor, /) -> Tensor:
        self.last_u = u
        return x


class TestImputers:
    def test_zero_imputer_uses_nan_mask_by_default(self) -> None:
        r"""NaN entries should be imputed when no explicit mask is given."""
        imputer = ZeroImputer()
        y = torch.tensor([[1.0, torch.nan, 3.0]])
        x = torch.randn(1, 2)

        actual = imputer(y, x)
        expected = torch.tensor([[1.0, 0.0, 3.0]])

        torch.testing.assert_close(actual, expected)

    def test_linear_imputer_respects_explicit_imputation_mask(self) -> None:
        r"""Explicit masks should control which coordinates are replaced."""
        imputer = LinearImputer(input_size=3, hidden_size=3)
        y = torch.tensor([[1.0, 2.0, 3.0]])
        x = torch.tensor([[10.0, 20.0, 30.0]])
        mask = torch.tensor([[False, True, False]])

        with torch.no_grad():
            imputer.linear.weight.copy_(torch.eye(3))

        actual = imputer(y, x, mask=mask)
        expected = torch.tensor([[1.0, 20.0, 3.0]])

        torch.testing.assert_close(actual, expected)

    def test_correlation_imputer_respects_explicit_imputation_mask(self) -> None:
        r"""Correlation imputation should treat False mask entries as observed."""
        imputer = CorrelationImputer(decoder=IdentityDecoder(output_size=2))
        y = torch.tensor([[10.0, 2.0]])
        x = torch.tensor([[1.0, 3.0]])
        mask = torch.tensor([[False, True]])

        actual = imputer(y, x, mask=mask)
        expected = torch.tensor([[10.0, 3.0]])

        torch.testing.assert_close(actual, expected)


class TestMissingValueCell:
    def test_forwards_explicit_mask_to_imputer(self) -> None:
        r"""MissingValueCell should pass the explicit imputation mask through."""
        imputer = CaptureImputer()
        cell = MissingValueCell(
            input_size=3,
            hidden_size=3,
            filter_type=RecorderFilter,
            concat_mask=False,
            imputation=imputer,
        )
        y = torch.tensor([[1.0, 2.0, 3.0]])
        x = torch.tensor([[4.0, 5.0, 6.0]])
        mask = torch.tensor([[False, True, False]])

        output = cell(y, x, mask=mask)

        torch.testing.assert_close(output, x)
        torch.testing.assert_close(cell.mask, mask)
        assert imputer.received_mask is not None
        torch.testing.assert_close(imputer.received_mask, mask)
        torch.testing.assert_close(cell.imputed, torch.tensor([[1.0, -1.0, 3.0]]))

    def test_infers_nan_mask_and_concatenates_it(self) -> None:
        r"""MissingValueCell should derive the mask from NaNs when none is given."""
        cell = MissingValueCell(
            input_size=3,
            hidden_size=3,
            filter_type=RecorderFilter,
            concat_mask=True,
            imputation=CaptureImputer(),
        )
        y = torch.tensor([[1.0, torch.nan, 3.0]])
        x = torch.tensor([[4.0, 5.0, 6.0]])

        output = cell(y, x)

        torch.testing.assert_close(output, x)
        torch.testing.assert_close(cell.mask, torch.tensor([[False, True, False]]))
        torch.testing.assert_close(cell.imputed, torch.tensor([[1.0, -1.0, 3.0]]))
        torch.testing.assert_close(
            cell.filter.last_u,
            torch.tensor([[1.0, -1.0, 3.0, 0.0, 1.0, 0.0]]),
        )
