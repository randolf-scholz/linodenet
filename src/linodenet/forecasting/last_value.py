r"""Last-value forecasting baseline."""

__all__ = ["LastValue"]

import torch
from torch import Tensor, nn


class LastValue(nn.Module):
    r"""Forecast each feature with its latest observed context value.

    Missing context features are represented by ``NaN`` and skipped independently
    per feature. If no observed value exists at or before a query time, the model
    falls back to the initial state.
    """

    def forward(
        self,
        query_times: Tensor,  # (..., $K)
        *,
        context_times: Tensor,  # (..., $N)        context_mask: Tensor,  # (..., $N, D), bool
        context_mask: Tensor,  # (..., $N, D), bool
        context_values: Tensor,  # (..., $N, D)
        initial_state: Tensor | None = None,  # (...?, D)
    ) -> Tensor:  # (..., $K, D)
        r"""Compute last-value forecasts.

        Args:
            query_times: Query times with shape ``(..., num_queries)``.
            context_times: Context times with shape ``(..., num_context)``.
            context_mask: Boolean mask selecting observed entries in ``values``.
            context_values: Context values with shape ``(..., num_context, dim)``.
                Use ``NaN`` to indicate missing feature values.
            initial_state: Optional initial state with shape ``(..., dim)``. This
                is used as the last value for features that have no observed
                context value at or before a query time.

        Returns:
            Forecasts with shape ``(..., num_queries, dim)``.
        """
        # sanitize arguments
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        assert torch.equal(context_values.isfinite(), context_mask)

        Q = query_times
        T = context_times
        X = context_values
        X0 = initial_state

        # check that Q and T are sorted per batch element.
        # note that they may contain trailing NaN values.
        neginf = torch.full_like(Q[..., :1], -torch.inf)
        ΔQ = Q.diff(dim=-1, prepend=neginf)
        ΔT = T.diff(dim=-1, prepend=neginf)
        Q_valid = Q.isfinite()
        T_valid = T.isfinite()
        # check that non-valid values are at the tail
        assert (Q_valid[..., :-1] | ~Q_valid[..., 1:]).all(dim=-1).all()
        assert (T_valid[..., :-1] | ~T_valid[..., 1:]).all(dim=-1).all()
        # check that valid values are increasing
        assert (~Q_valid | (ΔQ >= 0)).all(dim=-1).all()
        assert (~T_valid | (ΔT >= 0)).all(dim=-1).all()

        # Treat the initial state as an extra context row at time -inf.
        T0 = torch.full_like(T[..., :1], -torch.inf)
        X0 = (
            torch.full_like(X[..., :1, :], torch.nan)
            if X0 is None
            else X0.expand_as(X[..., :1, :])
        )
        T = torch.cat([T0, T], dim=-1)
        X = torch.cat([X0, X], dim=-2)

        # Forward-fill observed context values feature-wise along the time axis.
        X_ffilled = X.gather(-2, X.isfinite().cummax(dim=-2).indices)

        # Map each query to the latest context row that is not in its future.
        # forecast_shape = Q.shape + X.shape[-1:]
        index = torch.searchsorted(T, Q, right=True) - 1  # (..., $K)
        index = index.unsqueeze(-1).expand(*Q.shape, X.shape[-1])  # (..., $K, D)
        # gather_index = latest_context.unsqueeze(-1).expand(forecast_shape)
        Y = torch.where(
            Q_valid.unsqueeze(-1),
            X_ffilled.gather(-2, index),
            torch.nan,
        )
        assert Y.shape == Q.shape + X.shape[-1:]
        return Y
