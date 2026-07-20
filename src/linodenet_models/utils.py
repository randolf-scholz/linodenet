r"""Utility classes for forecasting."""

__all__ = [
    "JointTimeData",
    "SplitTimeData",
    "TripletTimeData",
    "EventBatch",
    "DiscreteTimeEventBatch",
    # functions
    "is_prefix_mask",
    "unique_count",
]


import math
from collections.abc import Collection, Iterable
from dataclasses import InitVar, dataclass, field
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


def _tensor_values_equal(lhs: Tensor, rhs: Tensor, /) -> bool:
    if lhs.shape != rhs.shape or lhs.device != rhs.device:
        return False
    if lhs.is_floating_point() != rhs.is_floating_point():
        return False
    if lhs.is_floating_point():
        return bool(torch.allclose(lhs, rhs, atol=0.0, rtol=0.0, equal_nan=True))
    return bool(torch.equal(lhs, rhs))


def _optional_tensor_values_equal(lhs: Tensor | None, rhs: Tensor | None, /) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return _tensor_values_equal(lhs, rhs)


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


def _has_unique_time_channel_pairs(times: Tensor, channels: Tensor, /) -> Tensor:
    r"""Check whether valid `(time, channel)` pairs are unique per batch item."""
    valid = times.isfinite()
    pairs = torch.stack([times, channels.to(times.dtype)], dim=-1)
    pairs = pairs.masked_fill(~valid.unsqueeze(-1), nan)
    return unique_count(pairs).eq(valid.sum(dim=-1))


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


class EventBatch(NamedTuple):
    r"""Lightweight alternative to JointTimeData."""

    timestamps: Tensor  # Float[..., $T], padded NaN, non-decreasing

    context_mask: Tensor  # Bool[..., $T, D], padded False
    context_values: Tensor  # Float[..., $T, D], padded NaN, sparse
    context_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., K, D)`` from ``context_mask``."""

    query_mask: Tensor  # Bool[..., $T, F], padded False
    query_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., K, F)`` from ``query_mask``."""

    target_values: Tensor | None = None  # Float[..., $T, F], padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[..., M], padded NaN, sparse

    @staticmethod
    def from_request(
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F],  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        target_values: Tensor | None = None,  # Float[..., $K, F],  padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[..., M],  padded NaN, sparse
        batch_first: bool = True,
    ) -> EventBatch:
        seq_dim = -2 if batch_first else 0

        T = context_times.unsqueeze(-1).movedim(seq_dim, 0)
        C = context_mask.movedim(seq_dim, 0)
        X = context_values.movedim(seq_dim, 0)
        Q = query_times.unsqueeze(-1).movedim(seq_dim, 0)
        M = query_mask.movedim(seq_dim, 0)
        Y = target_values.movedim(seq_dim, 0) if target_values is not None else None

        ctx_size, *batch_shape, ctx_dim = X.shape
        q_size, *_, q_dim = M.shape
        ctx_pad_shape = (q_size, *batch_shape, ctx_dim)
        qry_pad_shape = (ctx_size, *batch_shape, q_dim)

        times = torch.cat([T, Q], dim=0)  # (..., $N+$K, 1)
        permutation = torch.argsort(  # (..., $N+$K, 1)
            times.nan_to_num(nan=torch.inf),
            dim=0,
            stable=True,
        )
        inv_perm = torch.argsort(permutation, dim=0, stable=True)  # (..., $N+$K, 1)
        ctx_idx = inv_perm[:ctx_size].movedim(0, seq_dim).squeeze(-1)
        qry_idx = inv_perm[ctx_size:].movedim(0, seq_dim).squeeze(-1)

        batch_idx = tuple(
            torch.arange(size, device=times.device)
            .reshape(
                *(size if j == i else 1 for j in range(len(batch_shape))),
            )
            .unsqueeze(-1 if batch_first else 0)
            for i, size in enumerate(batch_shape)
        )

        return EventBatch(
            timestamps=(
                times.take_along_dim(permutation, dim=0).movedim(0, seq_dim).squeeze(-1)
            ),
            context_values=(
                torch.cat([X, X.new_full(ctx_pad_shape, nan)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            context_mask=(
                torch.cat([C, C.new_zeros(ctx_pad_shape)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            query_mask=(
                torch.cat([M.new_zeros(qry_pad_shape), M], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            target_values=(
                torch.cat([Y.new_full(qry_pad_shape, nan), Y], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
                if Y is not None
                else None
            ),
            static_covariates=static_covariates,
            context_indices=(
                (*batch_idx, ctx_idx) if batch_first else (ctx_idx, *batch_idx)
            ),
            query_indices=(
                (*batch_idx, qry_idx) if batch_first else (qry_idx, *batch_idx)
            ),
        )

    def validate(self) -> None:
        T = self.timestamps
        C = self.context_mask
        X = self.context_values
        M = self.query_mask
        Y = self.target_values

        *batch_shape, num_combined, context_dim = X.shape
        *query_batch_shape, query_combined, query_dim = M.shape
        T_valid = T.isfinite()
        T_ascending = T.diff(dim=-1).ge(0.0)
        assert T.shape == (*batch_shape, num_combined)
        assert is_prefix_mask(T_valid).all()
        assert is_prefix_mask(T_ascending).all()  # sorted in ascending order

        assert C.dtype == torch.bool
        assert C.shape == (*batch_shape, num_combined, context_dim)
        assert torch.equal(X.isfinite(), C)
        mask_valid = C.any(dim=-1) | M.any(dim=-1)

        assert X.shape == (*batch_shape, num_combined, context_dim)
        assert M.dtype == torch.bool
        assert M.shape == (*query_batch_shape, query_combined, query_dim)
        assert query_batch_shape == batch_shape
        assert query_combined == num_combined
        assert is_prefix_mask(mask_valid).all()  # at least one value per step

        if Y is not None:
            assert Y.shape == (*batch_shape, num_combined, query_dim)
            assert torch.equal(Y.isfinite(), M)

        context_filter = C.any(dim=-1)
        query_filter = M.any(dim=-1)
        for times, context, query in zip(
            T.reshape(-1, num_combined),
            context_filter.reshape(-1, num_combined),
            query_filter.reshape(-1, num_combined),
            strict=True,
        ):
            assert times[context].diff(dim=-1).ge(0.0).all()
            assert times[query].diff(dim=-1).gt(0.0).all()


class DiscreteTimeEventBatch(NamedTuple):
    r"""Lightweight integer-step event batch for discrete-time models.

    Padded step entries are canonicalized to zero in the returned joint step
    tensor. Validity is represented by the context/query masks, not by sentinel
    step values.
    """

    steps: Tensor  # Long[..., $T], padded 0, non-decreasing over valid entries

    context_mask: Tensor  # Bool[..., $T, D], padded False
    context_values: Tensor  # Float[..., $T, D], padded NaN, sparse
    context_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., K, D)`` from ``context_mask``."""

    query_mask: Tensor  # Bool[..., $T, F], padded False
    query_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., K, F)`` from ``query_mask``."""

    target_values: Tensor | None = None  # Float[..., $T, F], padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[..., M], padded NaN, sparse

    @staticmethod
    def from_request(
        *,
        query_steps: Tensor,  # Long[..., $K], padded 0, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_steps: Tensor,  # Long[..., $N], padded 0, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        target_values: Tensor | None = None,  # Float[..., $K, F] padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[..., M] padded NaN, sparse
        batch_first: bool = True,
    ) -> DiscreteTimeEventBatch:
        seq_dim = -2 if batch_first else 0
        if query_steps.dtype != torch.long or context_steps.dtype != torch.long:
            raise TypeError("Discrete step indices must be Long tensors.")

        T = context_steps.unsqueeze(-1).movedim(seq_dim, 0)
        C = context_mask.movedim(seq_dim, 0)
        X = context_values.movedim(seq_dim, 0)
        Q = query_steps.unsqueeze(-1).movedim(seq_dim, 0)
        M = query_mask.movedim(seq_dim, 0)
        Y = target_values.movedim(seq_dim, 0) if target_values is not None else None

        ctx_size, *batch_shape, ctx_dim = X.shape
        q_size, *_, q_dim = M.shape
        ctx_pad_shape = (q_size, *batch_shape, ctx_dim)
        qry_pad_shape = (ctx_size, *batch_shape, q_dim)

        valid = torch.cat(
            [
                context_mask.any(dim=-1).unsqueeze(-1).movedim(seq_dim, 0),
                query_mask.any(dim=-1).unsqueeze(-1).movedim(seq_dim, 0),
            ],
            dim=0,
        )
        steps = torch.cat([T, Q], dim=0)
        sort_keys = steps.masked_fill(~valid, torch.iinfo(steps.dtype).max)
        permutation = torch.argsort(sort_keys, dim=0, stable=True)
        inv_perm = torch.argsort(permutation, dim=0, stable=True)
        ctx_idx = inv_perm[:ctx_size].movedim(0, seq_dim).squeeze(-1)
        qry_idx = inv_perm[ctx_size:].movedim(0, seq_dim).squeeze(-1)

        batch_idx = tuple(
            torch.arange(size, device=steps.device)
            .reshape(
                *(size if j == i else 1 for j in range(len(batch_shape))),
            )
            .unsqueeze(-1 if batch_first else 0)
            for i, size in enumerate(batch_shape)
        )

        return DiscreteTimeEventBatch(
            steps=(
                steps.take_along_dim(permutation, dim=0)
                .masked_fill(~valid.take_along_dim(permutation, dim=0), 0)
                .movedim(0, seq_dim)
                .squeeze(-1)
            ),
            context_values=(
                torch.cat([X, X.new_full(ctx_pad_shape, nan)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            context_mask=(
                torch.cat([C, C.new_zeros(ctx_pad_shape)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            query_mask=(
                torch.cat([M.new_zeros(qry_pad_shape), M], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            target_values=(
                torch.cat([Y.new_full(qry_pad_shape, nan), Y], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
                if Y is not None
                else None
            ),
            static_covariates=static_covariates,
            context_indices=(
                (*batch_idx, ctx_idx) if batch_first else (ctx_idx, *batch_idx)
            ),
            query_indices=(
                (*batch_idx, qry_idx) if batch_first else (qry_idx, *batch_idx)
            ),
        )


@dataclass(frozen=True)
class SplitTimeData:
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
        - query time stamps are finite and non-decreasing
        - if query values are given, they are finite exactly at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one target value observed per time stamp
    """

    context_times: Tensor  # Float[..., $N], padded NaN, non-decreasing
    context_values: Tensor  # Float[..., $N, D], padded NaN, sparse
    context_mask: Tensor  # Bool[..., $N, D], padded False

    query_times: Tensor  # Float[..., $K], padded NaN, non-decreasing
    query_mask: Tensor  # Bool[..., $K, F]  padded False
    target_values: Tensor | None = None  # Float[..., $K, F]  padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[..., M]  padded NaN, sparse
    r"""Optional time-independent data."""

    # metadata
    batch_first: bool = True
    r"""Whether the batch axes come before or after the time axes."""
    batch_shape: tuple[int, ...] = field(init=False)
    r"""The shape of the batch dimension."""
    context_size: int = field(init=False)
    r"""The maximum context size observed in the batch."""
    query_size: int = field(init=False)
    r"""The maximum query size observed in the batch."""
    context_dim: int = field(init=False)
    r"""The shape of the context dimension."""
    query_dim: int = field(init=False)
    r"""The shape of the query dimension."""

    # init options
    validate_args: InitVar[bool] = True
    r"""Whether to validate the data."""

    def __post_init__(self, validate_args: bool) -> None:
        self._normalize()
        if validate_args:
            self.validate()

    def _normalize(self) -> None:
        seq_dim = -2 if self.batch_first else 0
        batch_shape = (
            self.context_times.shape[:-1]
            if self.batch_first
            else self.context_times.shape[1:]
        )
        context_size = self.context_mask.shape[seq_dim]
        query_size = self.query_mask.shape[seq_dim]
        context_dim = self.context_mask.shape[-1]
        query_dim = self.query_mask.shape[-1]

        # sanitize context and target values
        with torch.no_grad():
            context_values = self.context_values.masked_fill_(~self.context_mask, nan)
            target_values = (
                self.target_values.masked_fill_(~self.query_mask, nan)
                if self.target_values is not None
                else None
            )

        # set metadata
        object.__setattr__(self, "batch_shape", batch_shape)
        object.__setattr__(self, "query_dim", query_dim)
        object.__setattr__(self, "context_dim", context_dim)
        object.__setattr__(self, "context_size", context_size)
        object.__setattr__(self, "query_size", query_size)
        object.__setattr__(self, "context_values", context_values)
        object.__setattr__(self, "target_values", target_values)

    def validate(self) -> None:
        # normalize to batch_first for validation
        seq_dim = -2 if self.batch_first else 0
        T = self.context_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        C = self.context_mask.movedim(seq_dim, -2)
        X = self.context_values.movedim(seq_dim, -2)
        Q = self.query_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        M = self.query_mask.movedim(seq_dim, -2)
        Y = (
            self.target_values.movedim(seq_dim, -2)
            if self.target_values is not None
            else None
        )

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
        Q_increasing = Q.diff(dim=-1).ge(0.0)
        T_increasing = T.diff(dim=-1).ge(0.0)
        assert is_prefix_mask(Q_increasing).all()  # query times are non-decreasing
        assert is_prefix_mask(T_increasing).all()  # context times are non-decreasing

        # check padding
        context_lengths = T.isfinite().sum(dim=-1)
        assert torch.equal(X_valid.sum(dim=-1), context_lengths)

        assert M.shape == (*batch_shape, query_size, query_dim)
        assert torch.equal(M.any(dim=-1), Q_valid)

        if Y is not None:
            assert Y.shape == (*batch_shape, query_size, query_dim)
            assert torch.equal(Y.isfinite(), M)

        if (S := self.static_covariates) is not None:
            *_, static_dim = S.shape
            assert S.shape == (*batch_shape, static_dim)

    def is_trimmed(self) -> bool:
        seq_dim = -2 if self.batch_first else 0
        C = self.context_mask.movedim(seq_dim, -2)
        M = self.query_mask.movedim(seq_dim, -2)
        return bool(
            C.any(dim=-1).reshape(-1, C.shape[-2]).any(dim=0).all()
            and M.any(dim=-1).reshape(-1, M.shape[-2]).any(dim=0).all()
        )

    def is_simple(self) -> bool:
        seq_dim = -2 if self.batch_first else 0
        T = self.context_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        Q = self.query_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        T_increasing = T.diff(dim=-1).gt(0.0)
        Q_increasing = Q.diff(dim=-1).gt(0.0)
        T_valid = T.isfinite()
        Q_valid = Q.isfinite()
        return bool(
            self.is_trimmed()
            and (
                is_prefix_mask(T_increasing)
                & is_prefix_mask(Q_increasing)
                & (T_increasing | ~T_valid[..., 1:]).all(dim=-1)
                & (Q_increasing | ~Q_valid[..., 1:]).all(dim=-1)
            ).all()
        )

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, SplitTimeData):
            return NotImplemented
        return (
            _tensor_values_equal(self.context_times, other.context_times)
            and _tensor_values_equal(self.context_values, other.context_values)
            and _tensor_values_equal(self.context_mask, other.context_mask)
            and _tensor_values_equal(self.query_times, other.query_times)
            and _tensor_values_equal(self.query_mask, other.query_mask)
            and _optional_tensor_values_equal(self.target_values, other.target_values)
            and _optional_tensor_values_equal(
                self.static_covariates, other.static_covariates
            )
        )

    @classmethod
    def from_joint_time(cls, arg: JointTimeData, /) -> SplitTimeData:
        return arg.to_split_time()

    @classmethod
    def from_triplets(cls, arg: TripletTimeData, /) -> SplitTimeData:
        return arg.to_split_time()

    @classmethod
    def from_unbatched(cls, args: Collection[SplitTimeData], /) -> SplitTimeData:
        if not args:
            raise ValueError("Expected at least one DenseArg.")

        if not all(arg.batch_first for arg in args):
            raise NotImplementedError("Only batch_first=True is supported.")

        return cls(
            validate_args=False,  # skip validation since we trust the arguments
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

    def unbatch(self) -> list[SplitTimeData]:
        if not self.batch_first:
            raise NotImplementedError

        T = self.context_times.unsqueeze(0).flatten(end_dim=-2)
        C = self.context_mask.unsqueeze(0).flatten(end_dim=-3)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-3)
        Q = self.query_times.unsqueeze(0).flatten(end_dim=-2)
        M = self.query_mask.unsqueeze(0).flatten(end_dim=-3)

        context_lengths = T.isfinite().sum(dim=-1)
        query_lengths = Q.isfinite().sum(dim=-1)
        num_samples = T.shape[0]

        return [
            SplitTimeData(
                validate_args=False,  # skip validation since we trust the arguments
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

    def to_triplets(self) -> TripletTimeData:
        seq_dim = -2 if self.batch_first else 0
        T = self.context_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        C = self.context_mask.movedim(seq_dim, -2)
        X = self.context_values.movedim(seq_dim, -2)
        Q = self.query_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        M = self.query_mask.movedim(seq_dim, -2)
        Y = (
            self.target_values.movedim(seq_dim, -2)
            if self.target_values is not None
            else None
        )

        *batch_shape, _, _ = X.shape

        # `nonzero` orders entries by batch, then time, then channel. Subtracting
        # the number of earlier valid entries in the same batch converts the global
        # nonzero order into a 0-based ragged position for that batch item.
        *ctx_batch_idx, ctx_size, ctx_dim = C.nonzero(as_tuple=True)
        counts = C.sum(dim=(-2, -1))  # (...)
        positions = torch.arange(ctx_size.numel(), device=ctx_size.device)
        offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
        ctx_idx = (*ctx_batch_idx, positions - offsets[*ctx_batch_idx])
        num_context = int(counts.max().item())

        *qry_batch_idx, qry_size, qry_dim = M.nonzero(as_tuple=True)
        counts = M.sum(dim=(-2, -1))  # (...)
        positions = torch.arange(qry_size.numel(), device=qry_size.device)
        offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
        qry_idx = (*qry_batch_idx, positions - offsets[*qry_batch_idx])
        num_query = int(counts.max().item())

        # move the sequence dim to the target position
        tgt_dim = -1 if self.batch_first else 0
        return TripletTimeData(
            # metadata
            query_size=self.query_size,
            query_dim=self.query_dim,
            context_dim=self.context_dim,
            batch_first=self.batch_first,
            validate_args=False,  # skip validation since we trust the arguments
            # data
            context_times=(
                T.new_full((*batch_shape, num_context), nan)
                .index_put(ctx_idx, T[*ctx_batch_idx, ctx_size])
                .movedim(-1, tgt_dim)
            ),
            context_channels=(
                ctx_dim.new_full((*batch_shape, num_context), -1)
                .index_put(ctx_idx, ctx_dim)
                .movedim(-1, tgt_dim)
            ),
            context_values=(
                X.new_full((*batch_shape, num_context), nan)
                .index_put(ctx_idx, X[*ctx_batch_idx, ctx_size, ctx_dim])
                .movedim(-1, tgt_dim)
            ),
            query_times=(
                Q.new_full((*batch_shape, num_query), nan)
                .index_put(qry_idx, Q[*qry_batch_idx, qry_size])
                .movedim(-1, tgt_dim)
            ),
            query_channels=(
                qry_dim.new_full((*batch_shape, num_query), -1)
                .index_put(qry_idx, qry_dim)
                .movedim(-1, tgt_dim)
            ),
            target_values=(
                Y.new_full((*batch_shape, num_query), nan)
                .index_put(qry_idx, Y[*qry_batch_idx, qry_size, qry_dim])
                .movedim(-1, tgt_dim)
                if Y is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_joint_time(self) -> JointTimeData:
        if not self.batch_first:
            raise NotImplementedError

        T = self.context_times
        C = self.context_mask
        X = self.context_values
        Q = self.query_times
        M = self.query_mask
        Y = self.target_values

        *batch_shape, ctx_size, ctx_dim = X.shape
        *_, q_size, q_dim = M.shape

        times = torch.cat([T, Q], dim=-1)
        permutation = torch.argsort(
            times.nan_to_num(nan=torch.inf), dim=-1, stable=True
        ).unsqueeze(-1)

        return JointTimeData(
            # metadata
            context_size=self.context_size,
            query_size=self.query_size,
            validate_args=False,  # skip validation since we trust the arguments
            batch_first=self.batch_first,
            # data
            timestamps=times.take_along_dim(permutation.squeeze(-1), dim=-1),
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
class JointTimeData:
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
        - query time stamps are finite and non-decreasing
        - context values are finite exactly at entries selected by the context mask
        - if query values are given, they are finite exactly at entries selected by the query mask
        - each valid time stamp has at least one context or query mask entry
    """

    timestamps: Tensor  # Float[..., $T], padded NaN, non-decreasing
    context_mask: Tensor  # Bool[..., $T, D], padded False
    context_values: Tensor  # Float[..., $T, D], padded NaN, sparse
    query_mask: Tensor  # Bool[..., $T, E], padded False
    target_values: Tensor | None = None  # Float[..., $T, E], padded NaN, sparse
    r"""Only available during training, otherwise None."""
    static_covariates: Tensor | None = None  # Float[..., M], padded NaN, sparse
    r"""Optional time-independent data."""

    # metadata
    batch_first: bool = True
    r"""Whether the batch axes come before or after the time axes."""
    batch_shape: tuple[int, ...] = field(init=False)
    r"""The shape of the batch dimension."""
    context_size: int = -1
    r"""The maximum context size observed in the batch."""
    query_size: int = -1
    r"""The maximum query size observed in the batch."""
    context_dim: int = field(init=False)
    r"""The shape of the context dimension."""
    query_dim: int = field(init=False)
    r"""The shape of the query dimension."""

    validate_args: InitVar[bool] = True
    r"""Whether to validate the arguments."""

    @property
    def context_indices(self) -> tuple[Tensor, ...]:
        seq_dim = -1 if self.batch_first else 0
        ctx_valid = self.context_mask.any(dim=-1)
        ctx_count = ctx_valid.sum(dim=seq_dim)
        return self._split_indices(ctx_valid, ctx_count, self.context_size)

    @property
    def query_indices(self) -> tuple[Tensor, ...]:
        seq_dim = -1 if self.batch_first else 0
        qry_valid = self.query_mask.any(dim=-1)
        qry_count = qry_valid.sum(dim=seq_dim)
        return self._split_indices(qry_valid, qry_count, self.query_size)

    def __post_init__(self, validate_args: bool) -> None:
        self._normalize()
        if validate_args:
            self.validate()

    def _normalize(self) -> None:
        seq_dim = -1 if self.batch_first else 0
        batch_shape = (
            self.timestamps.shape[:-1]
            if self.batch_first
            else self.timestamps.shape[1:]
        )
        context_size = (
            self.context_size
            if self.context_size >= 0
            else int(self.context_mask.any(dim=-1).sum(dim=seq_dim).max().item())
        )
        query_size = (
            self.query_size
            if self.query_size >= 0
            else int(self.query_mask.any(dim=-1).sum(dim=seq_dim).max().item())
        )
        context_dim = self.context_mask.shape[-1]
        query_dim = self.query_mask.shape[-1]

        # sanitize context and target values
        with torch.no_grad():
            context_values = self.context_values.masked_fill_(~self.context_mask, nan)
            target_values = (
                self.target_values.masked_fill_(~self.query_mask, nan)
                if self.target_values is not None
                else None
            )

        # set metadata
        object.__setattr__(self, "batch_shape", batch_shape)
        object.__setattr__(self, "query_dim", query_dim)
        object.__setattr__(self, "context_dim", context_dim)
        object.__setattr__(self, "context_size", context_size)
        object.__setattr__(self, "query_size", query_size)
        object.__setattr__(self, "context_values", context_values)
        object.__setattr__(self, "target_values", target_values)

    def validate(self) -> None:
        # normalize to batch_first for validation
        seq_dim = -2 if self.batch_first else 0
        T = self.timestamps[..., None].movedim(seq_dim, -2).squeeze(-1)
        C = self.context_mask.movedim(seq_dim, -2)
        X = self.context_values.movedim(seq_dim, -2)
        M = self.query_mask.movedim(seq_dim, -2)
        Y = (
            self.target_values.movedim(seq_dim, -2)
            if self.target_values is not None
            else None
        )

        *batch_shape, num_combined, context_dim = X.shape
        *query_batch_shape, query_combined, query_dim = M.shape
        assert query_batch_shape == batch_shape
        assert query_combined == num_combined

        assert T.shape == (*batch_shape, num_combined)
        assert is_prefix_mask(T.isfinite()).all()
        assert is_prefix_mask(T.diff(dim=-1).ge(0.0)).all()  # sorted in ascending order

        assert C.dtype == torch.bool
        assert C.shape == (*batch_shape, num_combined, context_dim)
        assert torch.equal(X.isfinite(), C)

        ctx_steps = C.any(dim=-1)
        qry_steps = M.any(dim=-1)

        assert X.shape == (*batch_shape, num_combined, context_dim)
        assert M.dtype == torch.bool
        assert M.shape == (*query_batch_shape, query_combined, query_dim)
        assert is_prefix_mask(ctx_steps | qry_steps).all()  # at least 1 value per step

        if Y is not None:
            assert Y.shape == (*batch_shape, num_combined, query_dim)
            assert torch.equal(Y.isfinite(), M)

        for times, context, query in zip(
            T.reshape(-1, num_combined),
            ctx_steps.reshape(-1, num_combined),
            qry_steps.reshape(-1, num_combined),
            strict=True,
        ):
            assert times[context].diff(dim=-1).ge(0.0).all()
            assert times[query].diff(dim=-1).ge(0.0).all()

    def is_trimmed(self) -> bool:
        seq_dim = -2 if self.batch_first else 0
        C = self.context_mask.movedim(seq_dim, -2)
        M = self.query_mask.movedim(seq_dim, -2)
        return bool((C | M).any(dim=-1).reshape(-1, C.shape[-2]).any(dim=0).all())

    def is_simple(self) -> bool:
        seq_dim = -2 if self.batch_first else 0
        T = self.timestamps[..., None].movedim(seq_dim, -2).squeeze(-1)
        T_increasing = T.diff(dim=-1).gt(0.0)
        T_valid = T.isfinite()
        return bool(
            self.is_trimmed()
            and (
                is_prefix_mask(T_increasing)
                & (T_increasing | ~T_valid[..., 1:]).all(dim=-1)
            ).all()
        )

    def _split_indices(
        self,
        valid: Tensor,
        count: Tensor,
        size: int,
        /,
    ) -> tuple[Tensor, ...]:
        if self.batch_first:
            *batch_shape, num_combined = self.timestamps.shape
        else:
            num_combined, *batch_shape = self.timestamps.shape

        device = self.timestamps.device
        seq_dim = -1 if self.batch_first else 0

        # The joint representation comes from a stable sort of context/query
        # steps, so stably sorting by `~valid` recovers the split order: all
        # selected steps move to the front while preserving their relative order.
        valid_idx = torch.argsort(~valid, dim=seq_dim, stable=True).narrow(
            seq_dim, 0, size
        )
        # Under the non-degenerate contract, the final joint position is padded
        # whenever a split slot is padded, so it can stand in for every tail NaN.
        pad_idx = torch.full_like(valid_idx, num_combined - 1)

        keep = torch.arange(size, device=device)
        keep = (
            keep if self.batch_first else keep.reshape(size, *(1 for _ in batch_shape))
        ) < count.unsqueeze(seq_dim)
        # Build an integer advanced-index tensor with valid joint positions in
        # front and a single padded joint position filling the split tail.
        time_idx = torch.where(keep, valid_idx, pad_idx)

        batch_idx = tuple(
            torch.arange(batch_size, device=device)
            .reshape(
                *(batch_size if j == i else 1 for j in range(len(batch_shape))),
            )
            .unsqueeze(seq_dim)
            for i, batch_size in enumerate(batch_shape)
        )
        return (*batch_idx, time_idx) if self.batch_first else (time_idx, *batch_idx)

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, JointTimeData):
            return NotImplemented
        return (
            _tensor_values_equal(self.timestamps, other.timestamps)
            and _tensor_values_equal(self.context_values, other.context_values)
            and _tensor_values_equal(self.context_mask, other.context_mask)
            and _tensor_values_equal(self.query_mask, other.query_mask)
            and _optional_tensor_values_equal(self.target_values, other.target_values)
            and _optional_tensor_values_equal(
                self.static_covariates, other.static_covariates
            )
        )

    @classmethod
    def from_request(
        cls,
        *,
        context_times: Tensor,  # Float[(..., $N)], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[(..., $N, D)], padded False
        context_values: Tensor,  # Float[(..., $N, D)], padded NaN, sparse
        query_times: Tensor,  # Float[(..., $K)], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[(..., $K, F)]  padded False
        target_values: Tensor | None = None,  # Float[(..., $K, F)]  padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[(..., M)]  padded NaN, sparse
        # extra args
        batch_first: bool = True,
        validate: bool = False,
    ) -> JointTimeData:
        # normalize to batch_last for construction
        seq_dim = -2 if batch_first else 0
        T = context_times.unsqueeze(-1).movedim(seq_dim, 0)
        C = context_mask.movedim(seq_dim, 0)
        X = context_values.movedim(seq_dim, 0)
        Q = query_times.unsqueeze(-1).movedim(seq_dim, 0)
        M = query_mask.movedim(seq_dim, 0)
        Y = target_values.movedim(seq_dim, 0) if target_values is not None else None

        ctx_size, *batch_shape, ctx_dim = X.shape
        q_size, *_, q_dim = M.shape
        ctx_pad_shape = (q_size, *batch_shape, ctx_dim)
        qry_pad_shape = (ctx_size, *batch_shape, q_dim)

        times = torch.cat([T, Q], dim=0)  # (..., $N+$K, 1)
        permutation = torch.argsort(  # (..., $N+$K, 1)
            times.nan_to_num(nan=torch.inf),
            dim=0,
            stable=True,
        )
        # inv_perm = torch.argsort(permutation, dim=0, stable=True)  # (..., $N+$K, 1)
        # time_idx = inv_perm[ctx_size:].movedim(0, seq_dim).squeeze(-1)
        # batch_idx = tuple(
        #     torch.arange(size, device=times.device)
        #     .reshape(
        #         *(size if j == i else 1 for j in range(len(batch_shape))),
        #     )
        #     .unsqueeze(-1 if batch_first else 0)
        #     for i, size in enumerate(batch_shape)
        # )
        # query_idx = (*batch_idx, time_idx) if batch_first else (time_idx, *batch_idx)

        return JointTimeData(
            context_size=ctx_size,
            query_size=q_size,
            batch_first=batch_first,
            validate_args=validate,
            timestamps=(
                times.take_along_dim(permutation, dim=0).movedim(0, seq_dim).squeeze(-1)
            ),
            context_values=(
                torch.cat([X, X.new_full(ctx_pad_shape, nan)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            context_mask=(
                torch.cat([C, C.new_zeros(ctx_pad_shape)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            query_mask=(
                torch.cat([M.new_zeros(qry_pad_shape), M], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            target_values=(
                torch.cat([Y.new_full(qry_pad_shape, nan), Y], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
                if Y is not None
                else None
            ),
            static_covariates=static_covariates,
        )

    @classmethod
    def from_split_time(cls, arg: SplitTimeData, /) -> JointTimeData:
        return arg.to_joint_time()

    @classmethod
    def from_triplets(cls, arg: TripletTimeData, /) -> JointTimeData:
        return arg.to_joint_time()

    @classmethod
    def from_unbatched(cls, args: Collection[JointTimeData], /) -> JointTimeData:
        if not args:
            raise ValueError("Expected at least one CombinedArg.")

        if not all(arg.batch_first for arg in args):
            raise NotImplementedError("Only batch_first is supported.")

        return cls(
            context_size=max(arg.context_size for arg in args),
            query_size=max(arg.query_size for arg in args),
            validate_args=False,  # skip validation since we trust the arguments
            timestamps=pad_sequence(
                [arg.timestamps for arg in args],
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

    def unbatch(self) -> list[JointTimeData]:
        if not self.batch_first:
            raise NotImplementedError

        T = self.timestamps.unsqueeze(0).flatten(end_dim=-2)
        C = self.context_mask.unsqueeze(0).flatten(end_dim=-3)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-3)
        M = self.query_mask.unsqueeze(0).flatten(end_dim=-3)
        lengths = T.isfinite().sum(dim=-1)

        return [
            JointTimeData(
                context_size=self.context_size,
                query_size=self.query_size,
                validate_args=False,  # skip validation since we trust the arguments
                timestamps=time,
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

    def to_split_time(self) -> SplitTimeData:
        if not self.batch_first:
            raise NotImplementedError

        T = self.timestamps
        C = self.context_mask
        X = self.context_values
        M = self.query_mask
        Y = self.target_values

        # Gather the selected steps to the front of each batch item: a stable sort
        # of `~valid` keeps the selected steps (key False) in order ahead of the
        # rest, so the first `size` columns hold them. The gathered mask/values
        # tails are already all-False/all-NaN (unselected steps), so only the
        # gathered times need their padding tail (`~keep`) reset to NaN.
        ctx_valid = C.any(dim=-1)
        ctx_count = ctx_valid.sum(dim=-1)
        ctx_size = self.context_size
        ctx_perm = torch.argsort(~ctx_valid, dim=-1, stable=True)[..., :ctx_size]
        ctx_keep = torch.arange(ctx_size, device=T.device) < ctx_count[..., None]

        qry_valid = M.any(dim=-1)
        qry_count = qry_valid.sum(dim=-1)
        qry_size = self.query_size
        qry_perm = torch.argsort(~qry_valid, dim=-1, stable=True)[..., :qry_size]
        qry_keep = torch.arange(qry_size, device=T.device) < qry_count[..., None]

        return SplitTimeData(
            validate_args=False,  # skip validation since we trust the arguments
            batch_first=self.batch_first,
            context_times=(
                T.take_along_dim(ctx_perm, dim=-1).masked_fill(~ctx_keep, nan)
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

    def to_triplets(self) -> TripletTimeData:
        return self.to_split_time().to_triplets()


@dataclass(frozen=True)
class TripletTimeData:
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

    context_times: Tensor  # Float[..., $O], padded NaN, non-decreasing
    context_channels: Tensor  # Long[..., $O], padded -1
    context_values: Tensor  # Float[..., $O], padded NaN

    query_times: Tensor  # Float[..., $Q], padded NaN, non-decreasing
    query_channels: Tensor  # Long[..., $Q], padded -1
    target_values: Tensor | None = None  # Float[..., $Q], padded NaN
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[..., M], padded NaN, sparse
    r"""Optional time-independent data."""

    # metadata
    batch_first: bool = True
    r"""Whether the batch axes come before or after the time axes."""
    batch_shape: tuple[int, ...] = field(init=False)
    r"""The shape of the batch dimension."""
    context_size: int = -1
    r"""The maximum context size observed in the batch."""
    query_size: int = -1
    r"""The maximum query size observed in the batch."""
    context_dim: int = -1
    r"""The shape of the context dimension."""
    query_dim: int = -1
    r"""The shape of the query dimension."""

    validate_args: InitVar[bool] = True

    @property
    def context_indices(self) -> tuple[Tensor, ...]:
        r"""Advanced indices recovering the simple split context layout."""
        return self._simple_indices(
            self.context_times,
            self.context_channels,
            size=self.context_size,
            dim=self.context_dim,
        )

    @property
    def query_indices(self) -> tuple[Tensor, ...]:
        r"""Advanced indices recovering the simple split query layout."""
        return self._simple_indices(
            self.query_times,
            self.query_channels,
            size=self.query_size,
            dim=self.query_dim,
        )

    def __post_init__(self, validate_args: bool) -> None:
        self._normalize()

        if validate_args:
            self.validate()

    def _normalize(self) -> None:
        seq_dim = -1 if self.batch_first else 0
        T = self.context_times.movedim(seq_dim, -1)
        C = self.context_channels.movedim(seq_dim, -1)
        Q = self.query_times.movedim(seq_dim, -1)
        M = self.query_channels.movedim(seq_dim, -1)
        batch_shape = (
            self.context_channels.shape[:-1]
            if self.batch_first
            else self.context_channels.shape[1:]
        )
        context_size = (
            self.context_size
            if self.context_size >= 0
            else int(_consecutive_group_indices(T, C.ge(0))[1].max().item())
        )
        query_size = (
            self.query_size
            if self.query_size >= 0
            else int(_consecutive_group_indices(Q, M.ge(0))[1].max().item())
        )
        context_dim = (
            self.context_dim
            if self.context_dim >= 0
            else int(self.context_channels.max().item()) + 1
        )
        query_dim = (
            self.query_dim
            if self.query_dim >= 0
            else int(self.query_channels.max().item()) + 1
        )

        # sanitize context and target values
        with torch.no_grad():
            context_values = self.context_values.masked_fill_(
                self.context_times.isnan(), nan
            )
            target_values = (
                self.target_values.masked_fill_(self.query_times.isnan(), nan)
                if self.target_values is not None
                else None
            )

        # set metadata
        object.__setattr__(self, "batch_shape", batch_shape)
        object.__setattr__(self, "query_dim", query_dim)
        object.__setattr__(self, "context_dim", context_dim)
        object.__setattr__(self, "context_size", context_size)
        object.__setattr__(self, "query_size", query_size)
        object.__setattr__(self, "context_values", context_values)
        object.__setattr__(self, "target_values", target_values)

    def validate(self) -> None:
        # normalize to batch_first for validation
        seq_dim = -1 if self.batch_first else 0
        T = self.context_times.movedim(seq_dim, -1)
        C = self.context_channels.movedim(seq_dim, -1)
        X = self.context_values.movedim(seq_dim, -1)
        Q = self.query_times.movedim(seq_dim, -1)
        M = self.query_channels.movedim(seq_dim, -1)
        Y = (
            self.target_values.movedim(seq_dim, -1)
            if self.target_values is not None
            else None
        )

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

        *_, num_query = Q.shape
        assert Q.shape == (*batch_shape, num_query)
        assert is_prefix_mask(Q.isfinite()).all()
        assert is_prefix_mask(Q.diff(dim=-1).ge(0.0)).all()

        M_valid = M >= 0
        assert M.shape == (*batch_shape, num_query)
        assert is_prefix_mask(M_valid).all()
        assert torch.equal(M_valid, Q.isfinite())
        # query_pairs = torch.stack([Q, M], dim=-1)
        # query_pairs = query_pairs.masked_fill(~M_valid.unsqueeze(-1), nan)
        # assert torch.equal(unique_count(query_pairs), M_valid.sum(dim=-1))

        if Y is not None:
            assert Y.shape == (*batch_shape, num_query)
            assert torch.equal(Y.isfinite(), M_valid)

    def is_trimmed(self) -> bool:
        seq_dim = -1 if self.batch_first else 0
        C = self.context_channels.movedim(seq_dim, -1)
        M = self.query_channels.movedim(seq_dim, -1)
        return bool(
            C.ge(0).reshape(-1, C.shape[-1]).any(dim=0).all()
            and M.ge(0).reshape(-1, M.shape[-1]).any(dim=0).all()
        )

    def is_simple(self) -> bool:
        seq_dim = -1 if self.batch_first else 0
        T = self.context_times.movedim(seq_dim, -1)
        C = self.context_channels.movedim(seq_dim, -1)
        Q = self.query_times.movedim(seq_dim, -1)
        M = self.query_channels.movedim(seq_dim, -1)
        return bool(
            self.is_trimmed()
            and (
                _has_unique_time_channel_pairs(T, C)
                & _has_unique_time_channel_pairs(Q, M)
            ).all()
        )

    def _simple_indices(
        self,
        times: Tensor,
        channels: Tensor,
        /,
        *,
        size: int,
        dim: int,
    ) -> tuple[Tensor, ...]:
        if not self.is_simple():
            raise ValueError("Simple split indices are only available for simple data.")

        seq_dim = -1 if self.batch_first else 0
        T = times.movedim(seq_dim, -1)
        C = channels.movedim(seq_dim, -1)
        valid = C.ge(0)
        inverse, _ = _consecutive_group_indices(T, valid)
        *batch_shape, _ = T.shape

        index = torch.zeros(
            (*batch_shape, size, dim), dtype=torch.long, device=T.device
        )
        *batch_idx, flat_idx = valid.nonzero(as_tuple=True)
        index[*batch_idx, inverse[valid], C[valid]] = flat_idx

        total_dims = len(batch_shape) + 2
        if self.batch_first:
            batch_indices = tuple(
                torch.arange(batch_size, device=T.device).reshape(
                    *(batch_size if j == i else 1 for j in range(total_dims))
                )
                for i, batch_size in enumerate(batch_shape)
            )
            return (*batch_indices, index)

        batch_indices = tuple(
            torch.arange(batch_size, device=T.device).reshape(
                *(batch_size if j == i + 1 else 1 for j in range(total_dims))
            )
            for i, batch_size in enumerate(batch_shape)
        )
        return (index.movedim(-2, 0), *batch_indices)

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, TripletTimeData):
            return NotImplemented
        return (
            _tensor_values_equal(self.context_times, other.context_times)
            and _tensor_values_equal(self.context_channels, other.context_channels)
            and _tensor_values_equal(self.context_values, other.context_values)
            and _tensor_values_equal(self.query_times, other.query_times)
            and _tensor_values_equal(self.query_channels, other.query_channels)
            and _optional_tensor_values_equal(self.target_values, other.target_values)
            and _optional_tensor_values_equal(
                self.static_covariates, other.static_covariates
            )
        )

    @classmethod
    def from_request(
        cls,
        *,
        context_times: Tensor,  # Float[..., $N] or Float[$N, ...], padded NaN
        context_mask: Tensor,  # Bool[..., $N, D] or Bool[$N, ..., D]
        context_values: Tensor,  # Float[..., $N, D] or Float[N, ..., D]
        query_times: Tensor,  # Float[..., $K] or Float[$K, ...], padded NaN
        query_mask: Tensor,  # Bool[..., $K, F] or Bool[$K, ..., F]
        target_values: Tensor | None = None,  # Float[..., $K, F] or Float[$K, ..., F]
        static_covariates: Tensor | None = None,  # Float[..., M], padded NaN, sparse
        # extra args
        batch_first: bool = True,
        validate: bool = True,
    ) -> TripletTimeData:
        # normalize to batch_first for conversion
        seq_dim = -2 if batch_first else 0
        context_times = context_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        context_mask = context_mask.movedim(seq_dim, -2)
        context_values = context_values.movedim(seq_dim, -2)
        query_times = query_times[..., None].movedim(seq_dim, -2).squeeze(-1)
        query_mask = query_mask.movedim(seq_dim, -2)
        target_values = (
            target_values.movedim(seq_dim, -2) if target_values is not None else None
        )

        *batch_shape, _, _ = context_values.shape
        *ctx_batch_idx, ctx_time, ctx_dim = context_mask.nonzero(as_tuple=True)
        ctx_counts = context_mask.sum(dim=(-2, -1))
        ctx_positions = torch.arange(ctx_time.numel(), device=ctx_time.device)
        ctx_offsets = (
            ctx_counts.flatten().cumsum(dim=0).reshape(batch_shape) - ctx_counts
        )
        ctx_idx = (*ctx_batch_idx, ctx_positions - ctx_offsets[*ctx_batch_idx])
        num_context = int(ctx_counts.max().item())

        *qry_batch_idx, qry_time, qry_dim = query_mask.nonzero(as_tuple=True)
        qry_counts = query_mask.sum(dim=(-2, -1))
        qry_positions = torch.arange(qry_time.numel(), device=qry_time.device)
        qry_offsets = (
            qry_counts.flatten().cumsum(dim=0).reshape(batch_shape) - qry_counts
        )
        qry_idx = (*qry_batch_idx, qry_positions - qry_offsets[*qry_batch_idx])
        num_query = int(qry_counts.max().item())

        seq_dim = -1 if batch_first else 0
        return TripletTimeData(
            query_size=query_times.shape[seq_dim],
            context_dim=context_mask.shape[-1],
            query_dim=query_mask.shape[-1],
            batch_first=batch_first,
            validate_args=validate,
            context_times=(
                context_times.new_full((*batch_shape, num_context), nan)
                .index_put(ctx_idx, context_times[*ctx_batch_idx, ctx_time])
                .movedim(-1, seq_dim)
            ),
            context_channels=(
                ctx_dim.new_full((*batch_shape, num_context), -1)
                .index_put(ctx_idx, ctx_dim)
                .movedim(-1, seq_dim)
            ),
            context_values=(
                context_values.new_full((*batch_shape, num_context), nan)
                .index_put(ctx_idx, context_values[*ctx_batch_idx, ctx_time, ctx_dim])
                .movedim(-1, seq_dim)
            ),
            query_times=(
                query_times.new_full((*batch_shape, num_query), nan)
                .index_put(qry_idx, query_times[*qry_batch_idx, qry_time])
                .movedim(-1, seq_dim)
            ),
            query_channels=(
                qry_dim.new_full((*batch_shape, num_query), -1)
                .index_put(qry_idx, qry_dim)
                .movedim(-1, seq_dim)
            ),
            target_values=(
                target_values.new_full((*batch_shape, num_query), nan)
                .index_put(qry_idx, target_values[*qry_batch_idx, qry_time, qry_dim])
                .movedim(-1, seq_dim)
                if target_values is not None
                else None
            ),
            static_covariates=static_covariates,
        )

    @classmethod
    def from_merged_timedata(cls, arg: JointTimeData, /) -> TripletTimeData:
        return arg.to_triplets()

    @classmethod
    def from_split_timedata(cls, arg: SplitTimeData, /) -> TripletTimeData:
        return arg.to_triplets()

    @classmethod
    def from_unbatched(cls, args: Collection[TripletTimeData], /) -> TripletTimeData:
        if not all(arg.batch_first for arg in args):
            raise NotImplementedError("Only batch_first is supported.")

        if not args:
            raise ValueError("Expected at least one TripletArg.")

        return cls(
            context_size=max(arg.context_size for arg in args),
            query_size=max(arg.query_size for arg in args),
            context_dim=max(arg.context_dim for arg in args),
            query_dim=max(arg.query_dim for arg in args),
            validate_args=False,  # skip validation since we trust the arguments
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

    def unbatch(self) -> list[TripletTimeData]:
        # normalize to batch_first for conversion & flatten batch dimensions
        seq_dim = -1 if self.batch_first else 0
        T = self.context_times.movedim(seq_dim, -1).unsqueeze(0).flatten(end_dim=-2)
        C = self.context_channels.movedim(seq_dim, -1).unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.movedim(seq_dim, -1).unsqueeze(0).flatten(end_dim=-2)
        Q = self.query_times.movedim(seq_dim, -1).unsqueeze(0).flatten(end_dim=-2)
        M = self.query_channels.movedim(seq_dim, -1).unsqueeze(0).flatten(end_dim=-2)

        context_lengths = T.isfinite().sum(dim=-1)
        query_lengths = Q.isfinite().sum(dim=-1)

        return [
            TripletTimeData(
                context_size=self.context_size,
                query_size=self.query_size,
                context_dim=self.context_dim,
                query_dim=self.query_dim,
                validate_args=False,  # skip validation since we trust the arguments
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

    def to_split_time(
        self,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> SplitTimeData:
        # normalize to batch_first for conversion
        seq_dim = -1 if self.batch_first else 0
        T = self.context_times.movedim(seq_dim, -1)
        C = self.context_channels.movedim(seq_dim, -1)
        X = self.context_values.movedim(seq_dim, -1)
        Q = self.query_times.movedim(seq_dim, -1)
        M = self.query_channels.movedim(seq_dim, -1)
        Y = (
            self.target_values.movedim(seq_dim, -1)
            if self.target_values is not None
            else None
        )

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
        ctx_steps = C_flat.ge(0)
        ctx_inverse, _ = _consecutive_group_indices(T_flat, ctx_steps)
        ctx_size = self.context_size
        ctx_batch = (
            torch.arange(num_batches, device=T.device)
            .unsqueeze(-1)
            .expand_as(ctx_steps)
        )
        ctx_indices = (ctx_batch[ctx_steps], ctx_inverse[ctx_steps])
        ctx_channels = C_flat[ctx_steps]

        # Map query triplets to dense row indices by grouping consecutive times.
        qry_steps = M_flat.ge(0)
        qry_inverse, _ = _consecutive_group_indices(Q_flat, qry_steps)
        qry_size = self.query_size
        qry_batch = (
            torch.arange(num_batches, device=Q.device)
            .unsqueeze(-1)
            .expand_as(qry_steps)
        )
        qry_indices = (qry_batch[qry_steps], qry_inverse[qry_steps])
        qry_channels = M_flat[qry_steps]

        seq_dim = -2 if self.batch_first else 0
        return SplitTimeData(
            validate_args=False,  # skip validate since we trust the arguments
            batch_first=self.batch_first,
            context_times=(
                T_flat.new_full((num_batches, ctx_size), nan)
                .index_put(ctx_indices, T_flat[ctx_steps])
                .reshape(*batch_shape, ctx_size, 1)
                .movedim(seq_dim, -2)
                .squeeze(-1)
            ),
            context_values=(
                X_flat.new_full((num_batches, ctx_size, ctx_dim), nan)
                .index_put((*ctx_indices, ctx_channels), X_flat[ctx_steps])
                .reshape(*batch_shape, ctx_size, ctx_dim)
                .movedim(seq_dim, -2)
            ),
            context_mask=(
                C_flat.new_zeros((num_batches, ctx_size, ctx_dim), dtype=torch.bool)
                .index_put(
                    (*ctx_indices, ctx_channels),
                    ctx_channels.new_ones((), dtype=torch.bool),
                )
                .reshape(*batch_shape, ctx_size, ctx_dim)
                .movedim(seq_dim, -2)
            ),
            query_times=(
                Q_flat.new_full((num_batches, qry_size), nan)
                .index_put(qry_indices, Q_flat[qry_steps])
                .reshape(*batch_shape, qry_size, 1)
                .movedim(seq_dim, -2)
                .squeeze(-1)
            ),
            query_mask=(
                M_flat.new_zeros((num_batches, qry_size, qry_dim), dtype=torch.bool)
                .index_put(
                    (*qry_indices, qry_channels),
                    qry_channels.new_ones((), dtype=torch.bool),
                )
                .reshape(*batch_shape, qry_size, qry_dim)
                .movedim(seq_dim, -2)
            ),
            target_values=(
                Y_flat.new_full((num_batches, qry_size, qry_dim), nan)
                .index_put((*qry_indices, qry_channels), Y_flat[qry_steps])
                .reshape(*batch_shape, qry_size, qry_dim)
                .movedim(seq_dim, -2)
                if Y_flat is not None
                else None
            ),
            static_covariates=self.static_covariates,
        )

    def to_joint_time(
        self,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> JointTimeData:
        return self.to_split_time(
            context_dim=context_dim, query_dim=query_dim
        ).to_joint_time()
