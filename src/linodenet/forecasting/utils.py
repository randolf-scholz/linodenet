r"""Utility classes for forecasting."""

__all__ = [
    "BatchedCombinedArgs",
    "BatchedForecastingRequest",
    "BatchedTripletArgs",
    "TripletArg",
    "ForecastingRequest",
    "CombinedArg",
    "EventBatch",
    # functions
    "is_prefix_mask",
    "scatter_fill",
    "unique_count",
]


import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nan
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


def _all_or_none[T](vals: Iterable[T | None], /) -> list[T] | None:
    result = []
    has_none = False
    for arg in vals:
        if arg is not None:
            result.append(arg)
        else:
            has_none = True
    if has_none and result:
        raise ValueError("Either all or none of the given values must be None.")
    return None if has_none else result


def is_prefix_mask(x: Tensor, /, *, dim: int = -1) -> Tensor:
    r"""Check that the given boolean tensor is valid up to the tail."""
    # check that a True value cannot follow a False value
    return (x[..., :-1] | ~x[..., 1:]).all(dim=dim)


def unique_count(x: Tensor, /) -> Tensor:
    r"""Count unique non-NaN rows in each batch item."""
    *batch_shape, num_items, num_dims = x.shape
    rows = x.reshape(-1, num_items, num_dims)
    num_batches = rows.shape[0]

    batch_ids = torch.arange(num_batches, device=x.device)
    batch_ids = batch_ids.reshape(-1, 1).expand(-1, num_items)

    if rows.is_complex():
        valid = ~torch.isnan(rows).any(dim=-1)

        # Promote to complex128 so batch ids are stored in float64 precision.
        keys = rows.to(torch.complex128)
        batch_keys = torch.complex(
            batch_ids.to(torch.float64),
            torch.zeros_like(batch_ids, dtype=torch.float64),
        )

    elif rows.is_floating_point():
        valid = ~torch.isnan(rows).any(dim=-1)

        # Promoting floats is fine and helps standardize dtype.
        keys = rows.to(torch.float64)
        batch_keys = batch_ids.to(torch.float64)

    elif rows.dtype == torch.bool:
        valid = torch.ones_like(batch_ids, dtype=torch.bool)

        keys = rows.to(torch.int64)
        batch_keys = batch_ids.to(torch.int64)

    else:
        valid = torch.ones_like(batch_ids, dtype=torch.bool)

        # Preserve integer exactness.
        # Caveat: uint64 values above int64 max cannot be represented exactly here.
        keys = rows.to(torch.int64)
        batch_keys = batch_ids.to(torch.int64)

    if not valid.any():
        return torch.zeros(batch_shape, dtype=torch.long, device=x.device)

    augmented = torch.cat(
        [batch_keys[valid].unsqueeze(-1), keys[valid]],
        dim=-1,
    )

    unique_rows = torch.unique(augmented, dim=0)

    return torch.bincount(
        unique_rows[:, 0].real.to(torch.long),
        minlength=num_batches,
    ).reshape(batch_shape)


def _consecutive_group_indices(
    values: Tensor, mask: Tensor, /
) -> tuple[Tensor, Tensor]:
    r"""Compute group indices for runs of consecutive equal values.

    The last dimension of `values` is interpreted as a sequence. For each batch
    item, every valid element starts a new group when it is the first valid
    element or when its value differs from the previous element. Invalid elements
    are ignored and get group index `-1`.

    This is similar to `torch.unique_consecutive(..., return_inverse=True)`,
    but computed for all batch items at once and with explicit masking.

    Args:
        values: Values with shape `(..., N)`.
        mask: Boolean mask with shape `(..., N)`.

    Returns:
        A pair `(inverse, counts)`, where `inverse` has shape `(..., N)` and
        gives each valid element's consecutive group index, and `counts` has
        shape `(...)` with the number of valid groups per batch item.

    Example:
        >>> vals = torch.tensor([[1.0, 1.0, 2.0, float("nan")], [3.0, 4.0, 4.0, 5.0]])
        >>> m = torch.isfinite(vals)
        >>> inv, counts = _consecutive_group_indices(vals, m)
        >>> inv
        tensor([[ 0,  0,  1, -1],
                [ 0,  1,  1,  2]])
        >>> counts
        tensor([2, 3])
    """
    is_new = torch.ones_like(mask, dtype=torch.bool)
    is_new[..., 1:] = values[..., 1:] != values[..., :-1]
    is_new &= mask
    inverse = is_new.cumsum(dim=-1) - 1
    return inverse.masked_fill(~mask, -1), is_new.sum(dim=-1)


def scatter_fill(
    shape: Sequence[int],
    indices: tuple[Tensor, ...],
    reference: Tensor,
    /,
    *,
    fill_value: bool | float,
) -> Tensor:
    r"""Create a filled tensor and write reference values at the given indices."""
    result = reference.new_full(shape, fill_value)
    result[indices] = reference
    return result


class EventBatch(NamedTuple):
    r"""Represents a batch of events.

    Other than ForecastingRequest, this class does not perform any validation.
    """

    timestamps: Tensor  # Float[(..., $T)], padded NaN, non-decreasing

    context_values: Tensor  # Float[(..., $T, D)], padded NaN, sparse
    context_mask: Tensor  # Bool[(..., $T, D)], padded False

    query_mask: Tensor  # Bool[(..., $T, F)], padded False
    query_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., K, F)`` from ``target_values``.

    Usage::

        result = event_batch.target_values[event_batch.query_indices]
        # result.shape == (*batch_shape, K, F)
    """

    target_values: Tensor | None = None  # Float[(..., $T, F)], padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(..., M)], padded NaN, sparse

    @staticmethod
    def from_request(
        *,
        query_times: Tensor,  # Float[(..., K)], padded NaN, strictly increasing
        query_mask: Tensor,  # Bool[(..., K, F)]  padded False
        context_times: Tensor,  # Float[(..., N)], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[(..., N, D)], padded False
        context_values: Tensor,  # Float[(..., N, D)], padded NaN, sparse
        target_values: Tensor | None = None,  # Float[(..., K, F)]  padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[(..., M)]  padded NaN, sparse
        batch_first: bool = True,
    ) -> EventBatch:
        T = context_times
        X = context_values
        C = context_mask
        Q = query_times
        M = query_mask
        Y = target_values

        if batch_first:
            *batch_shape, ctx_size, ctx_dim = X.shape
            *_, q_size, q_dim = M.shape
            ctx_pad_shape = (*batch_shape, q_size, ctx_dim)
            qry_pad_shape = (*batch_shape, ctx_size, q_dim)
        else:
            ctx_size, *batch_shape, ctx_dim = X.shape
            q_size, *_, q_dim = M.shape
            ctx_pad_shape = (q_size, *batch_shape, ctx_dim)
            qry_pad_shape = (ctx_size, *batch_shape, q_dim)

        time_seq_dim = -1 if batch_first else 0
        mask_seq_dim = -2 if batch_first else 0

        times = torch.cat([T, Q], dim=time_seq_dim)
        permutation = torch.argsort(
            times.nan_to_num(nan=torch.inf),
            dim=time_seq_dim,
            stable=True,
        ).unsqueeze(-1)

        # inv_perm[j] = sorted position of original index j
        inv_perm = torch.argsort(permutation.squeeze(-1), dim=time_seq_dim, stable=True)
        batch_idx = tuple(
            torch.arange(size, device=times.device)
            .reshape(
                *(size if j == i else 1 for j in range(len(batch_shape))),
            )
            .unsqueeze(time_seq_dim)
            for i, size in enumerate(batch_shape)
        )

        if batch_first:
            time_idx = inv_perm[..., ctx_size:]  # (*batch_shape, K)
            query_indices: tuple[Tensor, ...] = (*batch_idx, time_idx)
        else:
            time_idx = inv_perm[ctx_size:]  # (K, *batch_shape)
            query_indices = (time_idx, *batch_idx)

        return EventBatch(
            timestamps=times.take_along_dim(permutation.squeeze(-1), dim=time_seq_dim),
            context_values=torch.cat(
                [X, X.new_full(ctx_pad_shape, nan)],
                dim=mask_seq_dim,
            ).take_along_dim(permutation, dim=mask_seq_dim),
            context_mask=torch.cat(
                [C, C.new_zeros(ctx_pad_shape)],
                dim=mask_seq_dim,
            ).take_along_dim(permutation, dim=mask_seq_dim),
            query_mask=torch.cat(
                [M.new_zeros(qry_pad_shape), M],
                dim=mask_seq_dim,
            ).take_along_dim(permutation, dim=mask_seq_dim),
            query_indices=query_indices,
            target_values=(
                torch.cat(
                    [Y.new_full(qry_pad_shape, nan), Y],
                    dim=mask_seq_dim,
                ).take_along_dim(permutation, dim=mask_seq_dim)
                if Y is not None
                else None
            ),
            static_covariates=static_covariates,
        )


@dataclass(frozen=True)
class ForecastingRequest:
    r"""Default representation of forecasting arguments.

    Assumptions:
        - context time stamps are finite and non-decreasing
        - context values are finite exactly at entries selected by the context mask
        - query time stamps are finite and strictly increasing
        - if query values are given, they are finite exactly at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one target value observed per time stamp
        - there is at least one target mask selected per time stamp
    """

    context_times: Tensor  # Float[(N)], finite, non-decreasing
    context_values: Tensor  # Float[(N, D)], sparse
    context_mask: Tensor  # Bool[(N, D)]

    query_times: Tensor  # Float[(K)], finite, strictly increasing
    query_mask: Tensor  # Bool[(K, F)]
    target_values: Tensor | None = None  # Float[(K, F)], sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    @classmethod
    def from_triplet(cls, arg: TripletArg, /) -> ForecastingRequest:
        return arg.to_dense()

    @classmethod
    def from_combined(cls, arg: CombinedArg, /) -> ForecastingRequest:
        return arg.to_dense()

    @classmethod
    def from_batched(
        cls, arg: BatchedForecastingRequest, /
    ) -> list[ForecastingRequest]:
        return arg.unbatch()

    def __post_init__(self) -> None:
        T = self.context_times
        C = self.context_mask
        Q = self.query_times
        M = self.query_mask
        X = self.context_values.masked_fill(~C, nan)
        object.__setattr__(self, "context_values", X)

        *_, context_size, context_dim = X.shape
        assert C.dtype == torch.bool
        assert T.shape == (context_size,)
        assert T.isfinite().all()
        assert (T.diff(dim=-1) >= 0.0).all()  # non-decreasing
        assert X.shape == (context_size, context_dim)
        assert C.shape == (context_size, context_dim)
        assert torch.equal(X.isfinite(), C)
        assert C.any(dim=-1).all()  # at least one value per step

        *_, query_size = Q.shape
        assert Q.shape == (query_size,)
        assert Q.isfinite().all()
        assert (Q.diff(dim=-1) > 0.0).all()  # strictly increasing

        *_, query_dim = M.shape
        assert M.dtype == torch.bool
        assert M.shape == (query_size, query_dim)
        assert M.any(dim=-1).all()  # at least one value per step

        if (V := self.target_values) is not None:
            V = V.masked_fill(~M, nan)
            object.__setattr__(self, "target_values", V)
            *_, query_dim = V.shape
            assert V.shape == (query_size, query_dim)
            assert V.isfinite().any(dim=-1).all()  # at least one value per step
            assert V.shape == M.shape
            assert torch.equal(V.isfinite(), M)

    def to_triplet(self) -> TripletArg:
        T = self.context_times
        X = self.context_values
        C = self.context_mask
        Q = self.query_times
        M = self.query_mask
        Y = self.target_values

        time_indices, context_channels = C.nonzero(as_tuple=True)  # O×2
        query_indices, query_channels = M.nonzero(as_tuple=True)

        return TripletArg(
            context_times=T[time_indices],
            context_channels=context_channels,
            context_values=X[time_indices, context_channels],
            query_times=Q[query_indices],
            query_channels=query_channels,
            target_values=Y[query_indices, query_channels] if Y is not None else None,
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> CombinedArg:
        X = self.context_values
        C = self.context_mask
        M = self.query_mask
        Y = self.target_values
        T = self.context_times
        Q = self.query_times

        *_, ctx_size, ctx_dim = X.shape
        *_, q_size, q_dim = M.shape

        times = torch.cat([T, Q], dim=-1)
        perm = torch.argsort(times, dim=-1, stable=True).unsqueeze(-1)

        return CombinedArg(
            times=times.take_along_dim(perm.squeeze(-1), dim=-1),
            context_values=torch.cat(
                [X, X.new_full((q_size, ctx_dim), nan)],
                dim=-2,
            ).take_along_dim(perm, dim=-2),
            context_mask=torch.cat(
                [C, C.new_zeros((q_size, ctx_dim))],
                dim=-2,
            ).take_along_dim(perm, dim=-2),
            target_values=(
                torch.cat(
                    [Y.new_full((ctx_size, q_dim), nan), Y],
                    dim=-2,
                ).take_along_dim(perm, dim=-2)
                if Y is not None
                else None
            ),
            query_mask=torch.cat(
                [M.new_zeros((ctx_size, q_dim)), M],
                dim=-2,
            ).take_along_dim(perm, dim=-2),
            static_covariates=self.static_covariates,
        )


@dataclass(frozen=True)
class BatchedForecastingRequest:
    r"""Batched forecasting arguments.

    Shapes:
        K: max(Kᵢ) query size
        N: max(Nᵢ) context size
        D: input dimensionality
        F: output dimensionality
        M: static covariate dimensionality

    Assumptions: (up to tail padding)
        - context time stamps are finite and non-decreasing
        - context values are finite exactly at entries selected by the context mask
        - query time stamps are finite and strictly increasing
        - if query values are given, they are finite exactly at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one target value observed per time stamp
    """

    context_times: Tensor  # Float[(..., N)], padded NaN, non-decreasing
    context_values: Tensor  # Float[(..., N, D)], padded NaN, sparse
    context_mask: Tensor  # Bool[(..., N, D)], padded False

    query_times: Tensor  # Float[(..., K)], padded NaN, strictly increasing
    query_mask: Tensor  # Bool[(..., K, F)]  padded False
    target_values: Tensor | None = None  # Float[(..., K, F)]  padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(..., M)]  padded NaN, sparse

    @classmethod
    def from_combined(cls, arg: BatchedCombinedArgs, /) -> BatchedForecastingRequest:
        return arg.to_dense()

    @classmethod
    def from_triplet(cls, arg: BatchedTripletArgs, /) -> BatchedForecastingRequest:
        return arg.to_dense()

    def __post_init__(self) -> None:
        T = self.context_times
        C = self.context_mask
        Q = self.query_times
        M = self.query_mask
        X = self.context_values.masked_fill(~C, nan)
        object.__setattr__(self, "context_values", X.detach().requires_grad_())

        # check shapes
        *batch_shape, context_size, _ = X.shape
        *_, query_size = Q.shape
        *_, query_dim = M.shape
        assert C.dtype == torch.bool
        assert M.dtype == torch.bool
        assert T.shape == (*batch_shape, context_size)
        assert C.shape == X.shape
        assert Q.shape == (*batch_shape, query_size)
        assert M.shape == (*batch_shape, query_size, query_dim)
        assert torch.equal(X.isfinite(), C)

        # check that non-valid values are at the tail
        T_valid = T.isfinite()
        Q_valid = Q.isfinite()
        X_valid = C.any(dim=-1)  # at least one value per step
        assert is_prefix_mask(Q_valid).all()
        assert is_prefix_mask(T_valid).all()
        assert is_prefix_mask(X_valid).all()
        Q_increasing = Q.diff(dim=-1).gt(0.0)  # query times are strictly increasing
        T_increasing = T.diff(dim=-1).ge(0.0)  # context times are non-decreasing
        assert is_prefix_mask(Q_increasing).all()
        assert is_prefix_mask(T_increasing).all()

        # check padding
        context_lengths = T.isfinite().sum(dim=-1)
        assert torch.equal(X_valid.sum(dim=-1), context_lengths)

        assert torch.equal(M.any(dim=-1), Q_valid)

        if (V := self.target_values) is not None:
            V = V.masked_fill(~M, nan)
            object.__setattr__(self, "target_values", V.detach().requires_grad_())
            *_, query_dim = V.shape
            assert V.shape == (*batch_shape, query_size, query_dim)
            V_valid = V.isfinite()
            assert M.shape == V.shape
            assert torch.equal(V_valid, M)

        if (S := self.static_covariates) is not None:
            *_, static_dim = S.shape
            assert S.shape == (*batch_shape, static_dim)

    @classmethod
    def from_unbatched(
        cls, args: Sequence[ForecastingRequest], /
    ) -> BatchedForecastingRequest:
        if not args:
            raise ValueError("Expected at least one DenseArg.")

        return cls(
            context_times=pad_sequence(
                [arg.context_times for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            context_values=pad_sequence(
                [arg.context_values for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            context_mask=pad_sequence(
                [arg.context_mask for arg in args],
                batch_first=True,
                padding_value=False,
            ),
            query_times=pad_sequence(
                [arg.query_times for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            query_mask=pad_sequence(
                [arg.query_mask for arg in args],
                batch_first=True,
                padding_value=False,
            ),
            target_values=(
                pad_sequence(V, batch_first=True, padding_value=nan)
                if (V := _all_or_none(arg.target_values for arg in args)) is not None
                else None
            ),
            static_covariates=(
                torch.stack(S)
                if (S := _all_or_none(arg.static_covariates for arg in args))
                is not None
                else None
            ),
        )

    def unbatch(self) -> list[ForecastingRequest]:
        T = self.context_times.unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-3)
        C = self.context_mask.unsqueeze(0).flatten(end_dim=-3)
        Q = self.query_times.unsqueeze(0).flatten(end_dim=-2)
        M = self.query_mask.unsqueeze(0).flatten(end_dim=-3)

        context_lengths = T.isfinite().sum(dim=-1)
        query_lengths = Q.isfinite().sum(dim=-1)
        num_samples = T.shape[0]

        return [
            ForecastingRequest(
                context_times=c_time,
                context_values=c_value,
                context_mask=c_mask,
                query_times=q_time,
                query_mask=q_mask,
                target_values=q_value,
                static_covariates=static_arg,
            )
            for c_time, c_mask, c_value, q_time, q_mask, q_value, static_arg in zip(
                unpad_sequence(T, context_lengths, batch_first=True),
                unpad_sequence(C, context_lengths, batch_first=True),
                unpad_sequence(X, context_lengths, batch_first=True),
                unpad_sequence(Q, query_lengths, batch_first=True),
                unpad_sequence(M, query_lengths, batch_first=True),
                (
                    unpad_sequence(
                        self.target_values.unsqueeze(0).flatten(end_dim=-3),
                        query_lengths,
                        batch_first=True,
                    )
                    if self.target_values is not None
                    else [None] * num_samples
                ),
                (
                    self.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                    if self.static_covariates is not None
                    else [None] * num_samples
                ),
                strict=True,
            )
        ]

    def to_triplet(self) -> BatchedTripletArgs:
        T = self.context_times
        X = self.context_values
        C = self.context_mask
        Q = self.query_times
        Y = self.target_values
        M = self.query_mask
        batch_shape = X.shape[:-2]

        # `nonzero` orders entries by batch, then time, then channel. Subtracting
        # the number of earlier valid entries in the same batch converts the global
        # nonzero order into a 0-based ragged position for that batch item.
        *ctx_batch_idx, ctx_time, c_channel = C.nonzero(as_tuple=True)
        counts = C.sum(dim=(-2, -1))  # (...)
        positions = torch.arange(ctx_time.numel(), device=ctx_time.device)
        offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
        context_indices = (*ctx_batch_idx, positions - offsets[*ctx_batch_idx])
        num_context = int(counts.max().item())

        *q_batch_idx, q_time, q_channel = M.nonzero(as_tuple=True)
        counts = M.sum(dim=(-2, -1))  # (...)
        positions = torch.arange(q_time.numel(), device=q_time.device)
        offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
        query_indices = (*q_batch_idx, positions - offsets[*q_batch_idx])
        num_query = int(counts.max().item())

        return BatchedTripletArgs(
            context_times=scatter_fill(
                (*batch_shape, num_context),
                context_indices,
                T[*ctx_batch_idx, ctx_time],
                fill_value=nan,
            ),
            context_channels=scatter_fill(
                (*batch_shape, num_context), context_indices, c_channel, fill_value=-1
            ),
            context_values=scatter_fill(
                (*batch_shape, num_context),
                context_indices,
                X[*ctx_batch_idx, ctx_time, c_channel],
                fill_value=nan,
            ),
            query_times=scatter_fill(
                (*batch_shape, num_query),
                query_indices,
                Q[*q_batch_idx, q_time],
                fill_value=nan,
            ),
            query_channels=scatter_fill(
                (*batch_shape, num_query),
                query_indices,
                q_channel,
                fill_value=-1,
            ),
            target_values=(
                scatter_fill(
                    (*batch_shape, num_query),
                    query_indices,
                    Y[*q_batch_idx, q_time, q_channel],
                    fill_value=nan,
                )
                if Y is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> BatchedCombinedArgs:
        T = self.context_times
        X = self.context_values
        C = self.context_mask
        Q = self.query_times
        M = self.query_mask
        Y = self.target_values

        *batch_shape, ctx_size, ctx_dim = X.shape
        *_, q_size, q_dim = M.shape

        times = torch.cat([T, Q], dim=-1)
        permutation = torch.argsort(
            times.nan_to_num(nan=torch.inf), dim=-1, stable=True
        ).unsqueeze(-1)

        return BatchedCombinedArgs(
            times=times.take_along_dim(permutation.squeeze(-1), dim=-1),
            context_values=torch.cat(
                [X, X.new_full((*batch_shape, q_size, ctx_dim), nan)],
                dim=-2,
            ).take_along_dim(permutation, dim=-2),
            context_mask=torch.cat(
                [C, C.new_zeros((*batch_shape, q_size, ctx_dim))],
                dim=-2,
            ).take_along_dim(permutation, dim=-2),
            target_values=(
                torch.cat(
                    [Y.new_full((*batch_shape, ctx_size, q_dim), nan), Y],
                    dim=-2,
                ).take_along_dim(permutation, dim=-2)
                if Y is not None
                else None
            ),
            query_mask=torch.cat(
                [M.new_zeros((*batch_shape, ctx_size, q_dim)), M],
                dim=-2,
            ).take_along_dim(permutation, dim=-2),
            static_covariates=self.static_covariates,
        )


@dataclass(frozen=True)
class TripletArg:
    r"""Triplet representation of forecasting arguments.

    Assumptions:
        - context time stamps are finite, non-decreasing
        - context channels are non-negative integers
        - context values are finite
        - (context_time, context_channel) pairs are not necessarily unique
        - query time stamps are finite, non-decreasing
        - query channels are non-negative integers
        - (query_time, query_channel) pairs are unique
        - if query values are given, they are finite
    """

    context_times: Tensor  # Float[(Oᵢ)], finite, non-decreasing
    context_channels: Tensor  # Long[(Oᵢ)]
    context_values: Tensor  # Float[(Oᵢ)], finite

    query_times: Tensor  # Float[(Qᵢ)], finite, non-decreasing
    query_channels: Tensor  # Long[(Qᵢ)]
    target_values: Tensor | None = None  # Float[(Qᵢ)], finite
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    @classmethod
    def from_dense(cls, arg: ForecastingRequest, /) -> TripletArg:
        return arg.to_triplet()

    @classmethod
    def from_combined(cls, arg: CombinedArg, /) -> TripletArg:
        return arg.to_triplet()

    @classmethod
    def from_batched(cls, arg: BatchedTripletArgs, /) -> list[TripletArg]:
        return arg.unbatch()

    def __post_init__(self) -> None:
        T = self.context_times
        C = self.context_channels
        X = self.context_values
        Q = self.query_times
        M = self.query_channels

        *_, num_context = T.shape
        assert C.dtype == torch.long
        assert T.shape == (num_context,)
        assert C.shape == (num_context,)
        assert X.shape == (num_context,)
        assert T.isfinite().all()
        assert C.isfinite().all()
        assert X.isfinite().all()
        assert C.ge(0).all()  # channels non-negative
        assert T.diff(dim=-1).ge(0.0).all()  # non-decreasing

        *_, num_query = Q.shape
        assert Q.shape == (num_query,)
        assert Q.isfinite().all()
        assert Q.diff(dim=-1).ge(0.0).all()

        assert M.dtype == torch.long
        assert M.shape == (num_query,)
        assert M.isfinite().all()
        assert M.ge(0).all()
        query_pairs = torch.stack([Q, M], dim=-1)  # (Q, 2)
        assert (unique_count(query_pairs.unsqueeze(0)) == num_query).all()

        if (V := self.target_values) is not None:
            assert V.shape == (num_query,)
            assert V.isfinite().all()

    def to_dense(
        self,
        /,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> ForecastingRequest:
        C = self.context_channels
        M = self.query_channels

        context_dim = (
            context_dim if context_dim is not None else int(C.max().item()) + 1
        )
        query_dim = query_dim if query_dim is not None else int(M.max().item()) + 1

        if (self.context_channels >= context_dim).any():
            raise ValueError("Expected context channel indices below context_dim.")
        if (self.query_channels >= query_dim).any():
            raise ValueError("Expected query channel indices below query_dim.")

        context_times, context_inverse = torch.unique_consecutive(
            self.context_times, return_inverse=True
        )

        query_times, query_inverse = torch.unique_consecutive(
            self.query_times, return_inverse=True
        )

        return ForecastingRequest(
            context_times=context_times,
            context_values=scatter_fill(
                (context_times.shape[0], context_dim),
                (context_inverse, self.context_channels),
                self.context_values,
                fill_value=nan,
            ),
            context_mask=scatter_fill(
                (context_times.shape[0], context_dim),
                (context_inverse, self.context_channels),
                torch.ones_like(self.context_channels, dtype=torch.bool),
                fill_value=False,
            ),
            query_times=query_times,
            query_mask=scatter_fill(
                (query_times.shape[0], query_dim),
                (query_inverse, self.query_channels),
                torch.ones_like(self.query_channels, dtype=torch.bool),
                fill_value=False,
            ),
            target_values=(
                scatter_fill(
                    (query_times.shape[0], query_dim),
                    (query_inverse, self.query_channels),
                    self.target_values,
                    fill_value=nan,
                )
                if self.target_values is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> CombinedArg:
        return self.to_dense().to_combined()


@dataclass(frozen=True)
class BatchedTripletArgs:
    r"""Triplet representation of forecasting arguments.

    Shapes:
        Q: max(Qᵢ) number of query values
        O: max(Oᵢ) number of observed values
        M: static covariate dimensionality

    Assumptions: (up to tail padding)
        - context time stamps are finite, non-decreasing
        - context channels are non-negative integers
        - context values are finite
        - query time stamps are finite, non-decreasing
        - query channels are non-negative integers
        - per sample, (query_time, query_channel) pairs are unique
        - if query values are given, they are finite
    """

    context_times: Tensor  # Float[(..., O)], padded NaN, non-decreasing
    context_channels: Tensor  # Long[(..., O)], padded -1
    context_values: Tensor  # Float[(..., O)], padded NaN

    query_times: Tensor  # Float[(..., Q)], padded NaN, non-decreasing
    query_channels: Tensor  # Long[(..., Q)], padded -1
    target_values: Tensor | None = None  # Float[(..., Q)], padded NaN
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(..., M)], padded NaN, sparse

    def __post_init__(self) -> None:
        T = self.context_times
        C = self.context_channels
        X = self.context_values
        Q = self.query_times
        M = self.query_channels

        *batch_shape, num_context = T.shape
        assert T.shape == (*batch_shape, num_context)
        assert C.shape == (*batch_shape, num_context)
        assert X.shape == (*batch_shape, num_context)
        assert is_prefix_mask(T.isfinite()).all()
        assert is_prefix_mask(X.isfinite()).all()
        assert is_prefix_mask(T.diff(dim=-1).ge(0.0)).all()
        C_valid = C >= 0
        assert is_prefix_mask(C_valid).all()
        assert torch.equal(C_valid, T.isfinite())
        assert torch.equal(C_valid, X.isfinite())

        *_, num_query = self.query_times.shape
        assert Q.shape == (*batch_shape, num_query)
        assert is_prefix_mask(Q.isfinite()).all()
        assert is_prefix_mask(Q.diff(dim=-1).ge(0.0)).all()

        M_valid = M >= 0
        assert M.shape == (*batch_shape, num_query)
        assert is_prefix_mask(M_valid).all()
        assert torch.equal(M_valid, Q.isfinite())
        query_pairs = torch.stack([Q, M], dim=-1)
        query_pairs = query_pairs.masked_fill(~M_valid.unsqueeze(-1), nan)
        assert torch.equal(unique_count(query_pairs), M_valid.sum(dim=-1))

        if (V := self.target_values) is not None:
            V_valid = V.isfinite()
            assert V.shape == (*batch_shape, num_query)
            assert torch.equal(V_valid, M_valid)

    @classmethod
    def from_combined(cls, arg: BatchedCombinedArgs, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_dense(cls, arg: BatchedForecastingRequest, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_unbatched(cls, args: Sequence[TripletArg]) -> BatchedTripletArgs:
        if not args:
            raise ValueError("Expected at least one TripletArg.")

        return cls(
            context_times=pad_sequence(
                [arg.context_times for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            context_channels=pad_sequence(
                [arg.context_channels for arg in args],
                batch_first=True,
                padding_value=-1,
            ),
            context_values=pad_sequence(
                [arg.context_values for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            query_times=pad_sequence(
                [arg.query_times for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            query_channels=pad_sequence(
                [arg.query_channels for arg in args],
                batch_first=True,
                padding_value=-1,
            ),
            target_values=(
                pad_sequence(V, batch_first=True, padding_value=nan)
                if (V := _all_or_none(arg.target_values for arg in args)) is not None
                else None
            ),
            static_covariates=(
                torch.stack(S)
                if (S := _all_or_none(arg.static_covariates for arg in args))
                is not None
                else None
            ),
        )

    def unbatch(self) -> list[TripletArg]:
        T = self.context_times.unsqueeze(0).flatten(end_dim=-2)
        C = self.context_channels.unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-2)
        Q = self.query_times.unsqueeze(0).flatten(end_dim=-2)
        M = self.query_channels.unsqueeze(0).flatten(end_dim=-2)

        context_lengths = T.isfinite().sum(dim=-1)
        query_lengths = Q.isfinite().sum(dim=-1)

        return [
            TripletArg(
                context_times=c_time,
                context_channels=c_channel,
                context_values=c_value,
                query_times=q_time,
                query_channels=q_channel,
                target_values=q_value,
                static_covariates=static_arg,
            )
            for c_time, c_channel, c_value, q_time, q_channel, q_value, static_arg in zip(
                unpad_sequence(T, context_lengths, batch_first=True),
                unpad_sequence(C, context_lengths, batch_first=True),
                unpad_sequence(X, context_lengths, batch_first=True),
                unpad_sequence(Q, query_lengths, batch_first=True),
                unpad_sequence(M, query_lengths, batch_first=True),
                (
                    unpad_sequence(
                        self.target_values.unsqueeze(0).flatten(end_dim=-2),
                        query_lengths,
                        batch_first=True,
                    )
                    if self.target_values is not None
                    else [None] * len(Q)
                ),
                (
                    self.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                    if self.static_covariates is not None
                    else [None] * len(T)
                ),
                strict=True,
            )
        ]

    def to_dense(
        self,
        /,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> BatchedForecastingRequest:
        T = self.context_times
        C = self.context_channels
        X = self.context_values
        Q = self.query_times
        M = self.query_channels
        Y = self.target_values

        ctx_dim = context_dim if context_dim is not None else int(C.max().item()) + 1
        qry_dim = query_dim if query_dim is not None else int(M.max().item()) + 1
        if (ctx_dim <= C).any():
            raise ValueError("Expected context channel indices below context_dim.")
        if (qry_dim <= M).any():
            raise ValueError("Expected query channel indices below query_dim.")

        *batch_shape, num_context = T.shape
        *_, num_query = Q.shape
        num_batches = math.prod(batch_shape)

        # Collapse arbitrary batch dimensions into one axis for vectorized grouping.
        T_flat = T.reshape(-1, num_context)
        C_flat = C.reshape(-1, num_context)
        X_flat = X.reshape(-1, num_context)
        Q_flat = Q.reshape(-1, num_query)
        M_flat = M.reshape(-1, num_query)
        Y_flat = Y.reshape(-1, num_query) if Y is not None else None

        # Map context triplets to dense row indices by grouping consecutive times.
        ctx_valid = C_flat.ge(0)
        ctx_inverse, ctx_lengths = _consecutive_group_indices(T_flat, ctx_valid)
        ctx_size = int(ctx_lengths.max().item())
        ctx_batch = (
            torch.arange(num_batches, device=T.device)
            .unsqueeze(-1)
            .expand_as(ctx_valid)
        )
        ctx_indices = (ctx_batch[ctx_valid], ctx_inverse[ctx_valid])
        ctx_channels = C_flat[ctx_valid]

        # Map query triplets to dense row indices by grouping consecutive times.
        qry_valid = M_flat.ge(0)
        qry_inverse, qry_lengths = _consecutive_group_indices(Q_flat, qry_valid)
        qry_size = int(qry_lengths.max().item())
        qry_batch = (
            torch.arange(num_batches, device=Q.device)
            .unsqueeze(-1)
            .expand_as(qry_valid)
        )
        qry_indices = (qry_batch[qry_valid], qry_inverse[qry_valid])
        qry_channels = M_flat[qry_valid]

        return BatchedForecastingRequest(
            context_times=scatter_fill(
                (num_batches, ctx_size),
                ctx_indices,
                T_flat[ctx_valid],
                fill_value=nan,
            ).reshape(*batch_shape, ctx_size),
            context_values=scatter_fill(
                (num_batches, ctx_size, ctx_dim),
                (*ctx_indices, ctx_channels),
                X_flat[ctx_valid],
                fill_value=nan,
            ).reshape(*batch_shape, ctx_size, ctx_dim),
            context_mask=scatter_fill(
                (num_batches, ctx_size, ctx_dim),
                (*ctx_indices, ctx_channels),
                torch.ones_like(ctx_channels, dtype=torch.bool),
                fill_value=False,
            ).reshape(*batch_shape, ctx_size, ctx_dim),
            query_times=scatter_fill(
                (num_batches, qry_size),
                qry_indices,
                Q_flat[qry_valid],
                fill_value=nan,
            ).reshape(*batch_shape, qry_size),
            query_mask=scatter_fill(
                (num_batches, qry_size, qry_dim),
                (*qry_indices, qry_channels),
                torch.ones_like(qry_channels, dtype=torch.bool),
                fill_value=False,
            ).reshape(*batch_shape, qry_size, qry_dim),
            target_values=(
                scatter_fill(
                    (num_batches, qry_size, qry_dim),
                    (*qry_indices, qry_channels),
                    Y_flat[qry_valid],
                    fill_value=nan,
                ).reshape(*batch_shape, qry_size, qry_dim)
                if Y_flat is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> BatchedCombinedArgs:
        return self.to_dense().to_combined()


@dataclass(frozen=True)
class CombinedArg:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        N: context size
        K: query size
        D: context data dimensionality
        E: query data dimensionality

    Assumptions:
        - time stamps are finite and non-decreasing
        - context time stamps are finite and non-decreasing
        - query time stamps are finite and strictly increasing
        - context values are finite exactly at entries selected by the context mask
        - if query values are given, they are finite exactly at entries selected by the query mask
        - each time stamp has at least one context or query mask entry
    """

    times: Tensor  # Float[($N + $K)], finite, non-decreasing
    context_values: Tensor  # Float[($N + $K, D)], sparse
    context_mask: Tensor  # Bool[($N + $K, D)]
    query_mask: Tensor  # Bool[($N + $K, E)]
    target_values: Tensor | None = None  # Float[($N + $K, E)], sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    def __post_init__(self) -> None:
        # sanitize context_values
        object.__setattr__(
            self,
            "context_values",
            self.context_values.masked_fill(~self.context_mask, nan),
        )
        # sanitize target_values
        object.__setattr__(
            self,
            "target_values",
            (
                self.target_values.masked_fill(~self.query_mask, nan)
                if self.target_values is not None
                else None
            ),
        )

        T = self.times
        X = self.context_values
        C = self.context_mask
        Y = self.target_values
        Q = self.query_mask

        *_, num_combined, context_dim = X.shape
        *_, query_combined, query_dim = Q.shape
        assert C.dtype == torch.bool
        assert T.shape == (num_combined,)
        assert T.isfinite().all()
        assert T.diff(dim=-1).ge(0.0).all()  # sorted in ascending order

        assert X.shape == (num_combined, context_dim)
        assert C.shape == (num_combined, context_dim)
        assert torch.equal(X.isfinite(), C)

        assert Q.dtype == torch.bool
        assert Q.shape == (query_combined, query_dim)
        assert query_combined == num_combined
        assert (C.any(dim=-1) | Q.any(dim=-1)).all()
        assert T[C.any(dim=-1)].diff(dim=-1).ge(0.0).all()
        assert T[Q.any(dim=-1)].diff(dim=-1).gt(0.0).all()
        if Y is not None:
            assert Y.shape == (num_combined, query_dim)
            assert torch.equal(Y.isfinite(), Q)

    @classmethod
    def from_dense(cls, arg: ForecastingRequest, /) -> CombinedArg:
        return arg.to_combined()

    @classmethod
    def from_triplet(cls, arg: TripletArg, /) -> CombinedArg:
        return arg.to_combined()

    @classmethod
    def from_batched(cls, arg: BatchedCombinedArgs, /) -> list[CombinedArg]:
        return arg.unbatch()

    def to_dense(self) -> ForecastingRequest:
        context_filter = self.context_mask.any(dim=-1)
        query_filter = self.query_mask.any(dim=-1)
        return ForecastingRequest(
            context_times=self.times[..., context_filter],
            context_values=self.context_values[..., context_filter, :],
            context_mask=self.context_mask[..., context_filter, :],
            query_times=self.times[..., query_filter],
            query_mask=self.query_mask[..., query_filter, :],
            target_values=(
                self.target_values[..., query_filter, :]
                if self.target_values is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_triplet(self) -> TripletArg:
        return self.to_dense().to_triplet()


@dataclass(frozen=True)
class BatchedCombinedArgs:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        N: max(Nᵢ) context size
        K: max(Kᵢ) query size
        D: context data dimensionality
        E: query data dimensionality
        M: static covariate dimensionality

    Assumptions: (up to tail padding)
        - time stamps are finite and non-decreasing
        - context time stamps are finite and non-decreasing
        - query time stamps are finite and strictly increasing
        - context values are finite exactly at entries selected by the context mask
        - if query values are given, they are finite exactly at entries selected by the query mask
        - each valid time stamp has at least one context or query mask entry
    """

    times: Tensor  # Float[(..., $N + $K)], padded NaN, non-decreasing
    context_values: Tensor  # Float[(..., $N + $K, D)], padded NaN, sparse
    context_mask: Tensor  # Bool[(..., $N + $K, D)], padded False
    query_mask: Tensor  # Bool[(..., $N + $K, E)], padded False
    target_values: Tensor | None = None  # Float[(..., $N + $K, E)], padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[(..., M)], padded NaN, sparse

    def __post_init__(self) -> None:
        # sanitize context_values
        object.__setattr__(
            self,
            "context_values",
            self.context_values.masked_fill(~self.context_mask, nan),
        )
        # sanitize target_values
        object.__setattr__(
            self,
            "target_values",
            (
                self.target_values.masked_fill(~self.query_mask, nan)
                if self.target_values is not None
                else None
            ),
        )

        T = self.times
        X = self.context_values
        C = self.context_mask
        Y = self.target_values
        Q = self.query_mask

        *batch_shape, num_combined, context_dim = X.shape
        *query_batch_shape, query_combined, query_dim = Q.shape
        T_valid = T.isfinite()
        T_ascending = T.diff(dim=-1).ge(0.0)
        assert T.shape == (*batch_shape, num_combined)
        assert is_prefix_mask(T_valid).all()
        assert is_prefix_mask(T_ascending).all()  # sorted in ascending order

        assert C.dtype == torch.bool
        assert C.shape == (*batch_shape, num_combined, context_dim)
        assert torch.equal(X.isfinite(), C)
        mask_valid = C.any(dim=-1) | Q.any(dim=-1)

        assert X.shape == (*batch_shape, num_combined, context_dim)
        assert Q.dtype == torch.bool
        assert Q.shape == (*query_batch_shape, query_combined, query_dim)
        assert query_batch_shape == batch_shape
        assert query_combined == num_combined
        assert is_prefix_mask(mask_valid).all()  # at least one value per step

        if Y is not None:
            assert Y.shape == (*batch_shape, num_combined, query_dim)
            assert torch.equal(Y.isfinite(), Q)

        context_filter = C.any(dim=-1)
        query_filter = Q.any(dim=-1)
        for times, context, query in zip(
            T.reshape(-1, num_combined),
            context_filter.reshape(-1, num_combined),
            query_filter.reshape(-1, num_combined),
            strict=True,
        ):
            assert times[context].diff(dim=-1).ge(0.0).all()
            assert times[query].diff(dim=-1).gt(0.0).all()

    @classmethod
    def from_dense(cls, arg: BatchedForecastingRequest, /) -> BatchedCombinedArgs:
        return arg.to_combined()

    @classmethod
    def from_triplet(cls, arg: BatchedTripletArgs, /) -> BatchedCombinedArgs:
        return arg.to_combined()

    @classmethod
    def from_unbatched(cls, args: Sequence[CombinedArg], /) -> BatchedCombinedArgs:
        if not args:
            raise ValueError("Expected at least one CombinedArg.")

        return cls(
            times=pad_sequence(
                [arg.times for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            context_values=pad_sequence(
                [arg.context_values for arg in args],
                batch_first=True,
                padding_value=nan,
            ),
            context_mask=pad_sequence(
                [arg.context_mask for arg in args],
                batch_first=True,
                padding_value=False,
            ),
            query_mask=pad_sequence(
                [arg.query_mask for arg in args],
                batch_first=True,
                padding_value=False,
            ),
            target_values=(
                pad_sequence(V, batch_first=True, padding_value=nan)
                if (V := _all_or_none(arg.target_values for arg in args)) is not None
                else None
            ),
            static_covariates=(
                torch.stack(S)
                if (S := _all_or_none(arg.static_covariates for arg in args))
                is not None
                else None
            ),
        )

    def unbatch(self) -> list[CombinedArg]:
        T = self.times.unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-3)
        C = self.context_mask.unsqueeze(0).flatten(end_dim=-3)
        M = self.query_mask.unsqueeze(0).flatten(end_dim=-3)
        lengths = T.isfinite().sum(dim=-1)

        return [
            CombinedArg(
                times=time,
                context_values=c_value,
                context_mask=c_mask,
                target_values=q_value,
                query_mask=q_mask,
                static_covariates=static_arg,
            )
            for time, c_mask, c_value, q_mask, q_value, static_arg in zip(
                unpad_sequence(T, lengths, batch_first=True),
                unpad_sequence(C, lengths, batch_first=True),
                unpad_sequence(X, lengths, batch_first=True),
                unpad_sequence(M, lengths, batch_first=True),
                (
                    unpad_sequence(
                        self.target_values.unsqueeze(0).flatten(end_dim=-3),
                        lengths,
                        batch_first=True,
                    )
                    if self.target_values is not None
                    else [None] * len(T)
                ),
                (
                    self.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                    if self.static_covariates is not None
                    else [None] * len(T)
                ),
                strict=True,
            )
        ]

    def to_dense(self) -> BatchedForecastingRequest:
        T = self.times
        X = self.context_values
        C = self.context_mask
        Y = self.target_values
        M = self.query_mask

        # Gather the selected steps to the front of each batch item: a stable sort
        # of `~valid` keeps the selected steps (key False) in order ahead of the
        # rest, so the first `size` columns hold them. The gathered mask/values
        # tails are already all-False/all-NaN (unselected steps), so only the
        # gathered times need their padding tail (`~keep`) reset to NaN.
        ctx_valid = C.any(dim=-1)
        ctx_count = ctx_valid.sum(dim=-1)
        ctx_size = int(ctx_count.max().item())
        ctx_perm = torch.argsort(~ctx_valid, dim=-1, stable=True)[..., :ctx_size]
        ctx_keep = torch.arange(ctx_size, device=T.device) < ctx_count[..., None]

        qry_valid = M.any(dim=-1)
        qry_count = qry_valid.sum(dim=-1)
        qry_size = int(qry_count.max().item())
        qry_perm = torch.argsort(~qry_valid, dim=-1, stable=True)[..., :qry_size]
        qry_keep = torch.arange(qry_size, device=T.device) < qry_count[..., None]

        return BatchedForecastingRequest(
            context_times=T.take_along_dim(ctx_perm, dim=-1).masked_fill(
                ~ctx_keep, nan
            ),
            context_mask=C.take_along_dim(ctx_perm[..., None], dim=-2),
            context_values=X.take_along_dim(ctx_perm[..., None], dim=-2),
            query_times=T.take_along_dim(qry_perm, dim=-1).masked_fill(~qry_keep, nan),
            query_mask=M.take_along_dim(qry_perm[..., None], dim=-2),
            target_values=(
                Y.take_along_dim(qry_perm[..., None], dim=-2) if Y is not None else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_triplet(self) -> BatchedTripletArgs:
        return self.to_dense().to_triplet()
