r"""Utility classes for forecasting."""

__all__ = [
    # Protocol
    "AbstractSplitTimeData",
    "AbstractMergedTimeData",
    "AbstractTripletTimeData",
    # classes
    "MergedTimeData",
    "SplitTimeData",
    "TripletTimeData",
    "EventBatch",
    "DiscreteTimeEventBatch",
    # functions
    "is_prefix_mask",
    "split_to_triplet",
    "split_to_merged",
    "merged_to_triplet",
    "merged_to_split",
    "triplet_to_merged",
    "triplet_to_split",
    "batch_split",
    "batch_merged",
    "batch_triplet",
    "unbatch_merged",
    "unbatch_split",
    "unbatch_triplet",
]


import math
from collections.abc import Collection, Iterable
from dataclasses import InitVar, dataclass, replace
from typing import NamedTuple, Protocol

import torch
from torch import Tensor, nan
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


class AbstractSplitTimeData(Protocol):
    r"""Protocol for split time representation.

    Attributes:
        context_times:     Float[..., $N], padded NaN, non-decreasing
        context_values:    Float[..., $N, D], padded NaN
        context_mask:      Bool[..., $N, D], padded False
        query_times:       Float[..., $K], padded NaN, non-decreasing
        query_mask:        Bool[..., $K, F],  padded False
        target_values:     Float[..., $K, F],  padded NaN
        static_covariates: Float[..., M],  padded NaN
    """

    @property
    def context_times(self) -> Tensor: ...
    @property
    def context_values(self) -> Tensor: ...
    @property
    def context_mask(self) -> Tensor: ...

    @property
    def query_times(self) -> Tensor: ...
    @property
    def query_mask(self) -> Tensor: ...
    @property
    def target_values(self) -> Tensor | None: ...

    @property
    def static_covariates(self) -> Tensor | None: ...


class AbstractMergedTimeData(Protocol):
    r"""Protocol for joint time representation.

    Attributes:
        timestamps:        Float[..., $T], padded NaN, non-decreasing
        context_mask:      Bool[..., $T, D], padded False
        context_values:    Float[..., $T, D], padded NaN
        query_mask:        Bool[..., $T, E], padded False
        target_values:     Float[..., $T, E], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    @property
    def timestamps(self) -> Tensor: ...

    @property
    def context_mask(self) -> Tensor: ...
    @property
    def context_values(self) -> Tensor: ...

    @property
    def query_mask(self) -> Tensor: ...
    @property
    def target_values(self) -> Tensor | None: ...

    @property
    def static_covariates(self) -> Tensor | None: ...


class AbstractTripletTimeData(Protocol):
    r"""Protocol for triplet time representation.

    Attributes:
        context_times:     Float[..., $X], padded NaN, non-decreasing
        context_channels:  Long[..., $X], padded -1
        context_values:    Float[..., $X], padded NaN
        query_times:       Float[..., $Q], padded NaN, non-decreasing
        query_channels:    Long[..., $Q], padded -1
        target_values:     Float[..., $Q], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    @property
    def context_times(self) -> Tensor: ...
    @property
    def context_channels(self) -> Tensor: ...
    @property
    def context_values(self) -> Tensor: ...

    @property
    def query_times(self) -> Tensor: ...
    @property
    def query_channels(self) -> Tensor: ...
    @property
    def target_values(self) -> Tensor | None: ...

    @property
    def static_covariates(self) -> Tensor | None: ...


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


def _triplet_row_indices(times: Tensor, channels: Tensor, /) -> tuple[Tensor, Tensor]:
    r"""Assign ordered triplets to canonical dense rows.

    A new row begins when the timestamp changes or, within the same timestamp,
    the channel is non-increasing. Thus each dense row contains strictly
    increasing channel indices while the original triplet order is preserved.

    Args:
        times: Timestamps with shape `(..., N)`, padded with NaN.
        channels: Channel indices with shape `(..., N)`, padded with `-1`.

    Returns:
        A pair `(indices, counts)`. `indices` contains the dense row index of
        every valid triplet and `-1` at padded positions. `counts` contains the
        number of canonical dense rows per batch item.
    """
    valid = channels.ge(0)
    is_new = torch.ones_like(valid)
    is_new[..., 1:] = (times[..., 1:] != times[..., :-1]) | (
        channels[..., 1:] <= channels[..., :-1]
    )
    is_new &= valid
    indices = is_new.cumsum(dim=-1) - 1
    return indices.masked_fill(~valid, -1), is_new.sum(dim=-1)


def is_prefix_mask(x: Tensor, /, *, dim: int = -1) -> Tensor:
    r"""Check that the given boolean tensor is valid up to the tail."""
    # check that a True value cannot follow a False value
    return (x[..., :-1] | ~x[..., 1:]).all(dim=dim)


class EventBatch(NamedTuple):
    r"""Lightweight alternative to JointTimeData."""

    timestamps: Tensor  # Float[..., $T], padded NaN, non-decreasing

    context_mask: Tensor  # Bool[..., $T, D], padded False
    context_values: Tensor  # Float[..., $T, D], padded NaN, sparse
    context_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., $N, D)`` from ``context_mask``."""

    query_mask: Tensor  # Bool[..., $T, F], padded False
    query_indices: tuple[Tensor, ...]
    r"""Advanced index tuple recovering ``(..., $K, F)`` from ``query_mask``."""

    target_values: Tensor | None = None  # Float[..., $T, F], padded NaN, sparse
    r"""Only available during training, otherwise None."""

    static_covariates: Tensor | None = None  # Float[..., M], padded NaN, sparse

    @staticmethod
    def from_request(
        *,
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F],  padded False
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
        context_times: Tensor,  # Integer[..., $N], padded 0, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        query_times: Tensor,  # Integer[..., $K], padded 0, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        target_values: Tensor | None = None,  # Float[..., $K, F] padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[..., M] padded NaN, sparse
        batch_first: bool = True,
    ) -> DiscreteTimeEventBatch:
        seq_dim = -2 if batch_first else 0
        assert not torch.is_floating_point(query_times)
        assert not torch.is_floating_point(context_times)

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
            context_mask=(
                torch.cat([C, C.new_zeros(ctx_pad_shape)], dim=0)
                .take_along_dim(permutation, dim=0)
                .movedim(0, seq_dim)
            ),
            context_values=(
                torch.cat([X, X.new_full(ctx_pad_shape, nan)], dim=0)
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

    # init options
    validate_args: InitVar[bool] = True
    r"""Whether to validate the data."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (
            self.context_times.shape[:-1]
            if self.batch_first
            else self.context_times.shape[1:]
        )

    @property
    def context_size(self) -> int:
        return self.context_mask.shape[-2 if self.batch_first else 0]

    @property
    def context_dim(self) -> int:
        return self.context_mask.shape[-1]

    @property
    def context_valid(self) -> Tensor:
        return self.context_mask.any(dim=-1)

    @property
    def query_size(self) -> int:
        return self.query_mask.shape[-2 if self.batch_first else 0]

    @property
    def query_dim(self) -> int:
        return self.query_mask.shape[-1]

    @property
    def query_valid(self) -> Tensor:
        return self.query_mask.any(dim=-1)

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
            and self.batch_first == other.batch_first
        )

    def __post_init__(self, validate_args: bool) -> None:
        if validate_args:
            self.validate()

    def normalize(self) -> SplitTimeData:
        # sanitize context and target values
        context_mask = self.context_mask
        context_times = self.context_times.masked_fill(~self.context_valid, nan)
        context_values = self.context_values.masked_fill(~context_mask, nan)
        query_mask = self.query_mask
        query_times = self.query_times.masked_fill(~self.query_valid, nan)
        target_values = (
            self.target_values.masked_fill(~query_mask, nan)
            if self.target_values is not None
            else None
        )

        return SplitTimeData(
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=self.static_covariates,
            batch_first=self.batch_first,
            validate_args=False,  # trust that the original data was valid, normalization does not change validity
        )

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
        *_, query_size, query_dim = M.shape
        assert C.dtype == torch.bool
        assert M.dtype == torch.bool
        assert T.shape == (*batch_shape, context_size)
        assert C.shape == X.shape
        assert Q.shape == (*batch_shape, query_size)
        assert M.shape == (*batch_shape, query_size, query_dim)
        assert not (C & ~X.isfinite()).any()  # C = True implies X is finite

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
        assert not (M.any(dim=-1) & ~Q_valid).any()  # M = True implies Q is finite

        if Y is not None:
            assert Y.shape == (*batch_shape, query_size, query_dim)
            assert not (M & ~Y.isfinite()).any()  # M = True implies Y is finite

        if (S := self.static_covariates) is not None:
            *_, static_dim = S.shape
            assert S.shape == (*batch_shape, static_dim)

    def is_trimmed(self) -> bool:
        seq_dim = -1 if self.batch_first else 0
        C = self.context_valid.movedim(seq_dim, -1)
        M = self.query_valid.movedim(seq_dim, -1)
        return bool(
            C.reshape(-1, C.shape[-1]).any(dim=0).all()
            and M.reshape(-1, M.shape[-1]).any(dim=0).all()
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

    @classmethod
    def from_unbatched(
        cls, args: Collection[AbstractSplitTimeData], /, *, batch_first: bool
    ) -> SplitTimeData:
        return batch_split(args, batch_first=batch_first)

    @classmethod
    def from_split(
        cls, arg: AbstractSplitTimeData, /, *, batch_first: bool
    ) -> SplitTimeData:
        if isinstance(arg, SplitTimeData) and arg.batch_first != batch_first:
            raise ValueError("arg.batch_first does not match batch_first.")
        return SplitTimeData(
            context_times=arg.context_times,
            context_mask=arg.context_mask,
            context_values=arg.context_values,
            query_times=arg.query_times,
            query_mask=arg.query_mask,
            target_values=arg.target_values,
            static_covariates=arg.static_covariates,
            batch_first=batch_first,
        )

    @classmethod
    def from_merged(
        cls, arg: AbstractMergedTimeData, /, *, batch_first: bool
    ) -> SplitTimeData:
        return merged_to_split(arg, batch_first=batch_first)

    @classmethod
    def from_triplet(
        cls, arg: AbstractTripletTimeData, /, *, batch_first: bool
    ) -> SplitTimeData:
        return triplet_to_split(arg, batch_first=batch_first)

    def to(
        self, *, batch_first: bool | None = None, device: torch.device | None = None
    ) -> SplitTimeData:
        r"""Convert batch layout and optionally move tensors to a device.

        Args:
            batch_first: Target batch layout. Retains the current layout when None.
            device: Target device. Retains the current device when None.
        """
        if batch_first is None and device is None:
            return self

        batch_first = self.batch_first if batch_first is None else batch_first
        t_dim_in = -1 if self.batch_first else 0
        t_dim_out = -1 if batch_first else 0
        seq_dim = -2 if self.batch_first else 0
        tgt_dim = -2 if batch_first else 0
        return replace(
            self,
            context_times=self.context_times.movedim(t_dim_in, t_dim_out).to(device),
            context_values=self.context_values.movedim(seq_dim, tgt_dim).to(device),
            context_mask=self.context_mask.movedim(seq_dim, tgt_dim).to(device),
            query_times=self.query_times.movedim(t_dim_in, t_dim_out).to(device),
            query_mask=self.query_mask.movedim(seq_dim, tgt_dim).to(device),
            target_values=(
                self.target_values.movedim(seq_dim, tgt_dim).to(device)
                if self.target_values is not None
                else None
            ),
            static_covariates=(
                self.static_covariates.to(device)
                if self.static_covariates is not None
                else None
            ),
            batch_first=batch_first,
            validate_args=False,
        )

    def unbatch(self) -> list[SplitTimeData]:
        return unbatch_split(self, batch_first=self.batch_first)

    def to_split(self) -> SplitTimeData:
        return self

    def to_merged(self) -> MergedTimeData:
        return split_to_merged(self, batch_first=self.batch_first)

    def to_triplet(self) -> TripletTimeData:
        return split_to_triplet(self, batch_first=self.batch_first)


@dataclass(frozen=True)
class MergedTimeData:
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
    context_size: int = -1
    r"""The maximum context size observed in the batch."""
    query_size: int = -1
    r"""The maximum query size observed in the batch."""

    validate_args: InitVar[bool] = True
    r"""Whether to validate the arguments."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (
            self.timestamps.shape[:-1]
            if self.batch_first
            else self.timestamps.shape[1:]
        )

    @property
    def context_dim(self) -> int:
        return self.context_mask.shape[-1]

    @property
    def context_valid(self) -> Tensor:
        return self.context_mask.any(dim=-1)

    @property
    def context_indices(self) -> tuple[Tensor, ...]:
        return self._split_indices(self.context_valid, self.context_size)

    @property
    def query_dim(self) -> int:
        return self.query_mask.shape[-1]

    @property
    def query_valid(self) -> Tensor:
        return self.query_mask.any(dim=-1)

    @property
    def query_indices(self) -> tuple[Tensor, ...]:
        return self._split_indices(self.query_valid, self.query_size)

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, MergedTimeData):
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
            and self.batch_first == other.batch_first
        )

    def __post_init__(self, validate_args: bool) -> None:
        seq_dim = -1 if self.batch_first else 0
        context_size = int(
            self.context_size
            if self.context_size >= 0
            else self.context_valid.sum(dim=seq_dim).max().item()
        )
        query_size = int(
            self.query_size
            if self.query_size >= 0
            else self.query_valid.sum(dim=seq_dim).max().item()
        )
        object.__setattr__(self, "context_size", context_size)
        object.__setattr__(self, "query_size", query_size)

        if validate_args:
            self.validate()

    def normalize(self) -> MergedTimeData:
        # sanitize context and target values
        context_mask = self.context_mask
        query_mask = self.query_mask
        timestamps = self.timestamps.masked_fill(
            ~(self.context_valid | self.query_valid), nan
        )
        context_values = self.context_values.masked_fill(~context_mask, nan)
        target_values = (
            self.target_values.masked_fill(~query_mask, nan)
            if self.target_values is not None
            else None
        )

        return MergedTimeData(
            timestamps=timestamps,
            context_mask=context_mask,
            context_values=context_values,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=self.static_covariates,
            batch_first=self.batch_first,
            context_size=self.context_size,
            query_size=self.query_size,
            validate_args=False,  # trust that the original data was valid, normalization does not change validity
        )

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
        assert not (C & ~X.isfinite()).any()  # C = True implies X is finite

        ctx_steps = C.any(dim=-1)
        qry_steps = M.any(dim=-1)

        assert X.shape == (*batch_shape, num_combined, context_dim)
        assert M.dtype == torch.bool
        assert M.shape == (*query_batch_shape, query_combined, query_dim)
        assert is_prefix_mask(ctx_steps | qry_steps).all()  # at least 1 value per step

        if Y is not None:
            assert Y.shape == (*batch_shape, num_combined, query_dim)
            assert not (M & ~Y.isfinite()).any()  # M = True implies Y is finite

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

    def _split_indices(self, valid: Tensor, size: int, /) -> tuple[Tensor, ...]:
        if self.batch_first:
            count = valid.sum(dim=-1)
            *batch_shape, num_combined = self.timestamps.shape
        else:
            count = valid.sum(dim=0)
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

    @classmethod
    def from_request(
        cls,
        *,
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        target_values: Tensor | None = None,  # Float[..., $K, F], padded NaN, sparse
        static_covariates: Tensor | None = None,  # Float[..., M], padded NaN, sparse
        # extra args
        batch_first: bool = True,
        validate: bool = False,
    ) -> MergedTimeData:
        return SplitTimeData(
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=static_covariates,
            batch_first=batch_first,
            validate_args=validate,
        ).to_merged()

    @classmethod
    def from_unbatched(
        cls, args: Collection[AbstractMergedTimeData], /, *, batch_first: bool
    ) -> MergedTimeData:
        return batch_merged(args, batch_first=batch_first)

    @classmethod
    def from_split(
        cls, arg: AbstractSplitTimeData, /, *, batch_first: bool
    ) -> MergedTimeData:
        return split_to_merged(arg, batch_first=batch_first)

    @classmethod
    def from_merged(
        cls, arg: AbstractMergedTimeData, /, *, batch_first: bool
    ) -> MergedTimeData:
        if isinstance(arg, MergedTimeData) and arg.batch_first != batch_first:
            raise ValueError("arg.batch_first does not match batch_first.")
        return MergedTimeData(
            timestamps=arg.timestamps,
            context_mask=arg.context_mask,
            context_values=arg.context_values,
            query_mask=arg.query_mask,
            target_values=arg.target_values,
            static_covariates=arg.static_covariates,
            batch_first=batch_first,
        )

    @classmethod
    def from_triplet(
        cls, arg: AbstractTripletTimeData, /, *, batch_first: bool
    ) -> MergedTimeData:
        return triplet_to_merged(arg, batch_first=batch_first)

    def to(
        self, *, batch_first: bool | None = None, device: torch.device | None = None
    ) -> MergedTimeData:
        r"""Convert batch layout and optionally move tensors to a device.

        Args:
            batch_first: Target batch layout. Retains the current layout when None.
            device: Target device. Retains the current device when None.
        """
        if batch_first is None and device is None:
            return self

        batch_first = self.batch_first if batch_first is None else batch_first
        t_dim_in = -1 if self.batch_first else 0
        t_dim_out = -1 if batch_first else 0
        seq_dim = -2 if self.batch_first else 0
        tgt_dim = -2 if batch_first else 0
        return replace(
            self,
            timestamps=self.timestamps.movedim(t_dim_in, t_dim_out).to(device),
            context_mask=self.context_mask.movedim(seq_dim, tgt_dim).to(device),
            context_values=self.context_values.movedim(seq_dim, tgt_dim).to(device),
            query_mask=self.query_mask.movedim(seq_dim, tgt_dim).to(device),
            target_values=(
                self.target_values.movedim(seq_dim, tgt_dim).to(device)
                if self.target_values is not None
                else None
            ),
            static_covariates=(
                self.static_covariates.to(device)
                if self.static_covariates is not None
                else None
            ),
            batch_first=batch_first,
            validate_args=False,
        )

    def unbatch(self) -> list[MergedTimeData]:
        return unbatch_merged(self, batch_first=self.batch_first)

    def to_split(self) -> SplitTimeData:
        return merged_to_split(self, batch_first=self.batch_first)

    def to_merged(self) -> MergedTimeData:
        return self

    def to_triplet(self) -> TripletTimeData:
        return merged_to_triplet(self, batch_first=self.batch_first)


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
    context_dim: int = -1
    r"""The shape of the context dimension."""
    query_dim: int = -1
    r"""The shape of the query dimension."""

    validate_args: InitVar[bool] = True

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (
            self.context_channels.shape[:-1]
            if self.batch_first
            else self.context_channels.shape[1:]
        )

    @property
    def context_indices(self) -> tuple[Tensor, ...]:
        r"""Advanced indices recovering the simple split context layout."""
        return self._split_indices(
            self.context_times,
            self.context_channels,
            dim=self.context_dim,
        )

    @property
    def context_valid(self) -> Tensor:
        return self.context_channels.ge(0)

    @property
    def query_indices(self) -> tuple[Tensor, ...]:
        r"""Advanced indices recovering the simple split query layout."""
        return self._split_indices(
            self.query_times,
            self.query_channels,
            dim=self.query_dim,
        )

    @property
    def query_valid(self) -> Tensor:
        return self.query_channels.ge(0)

    def __post_init__(self, validate_args: bool) -> None:
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

        # set metadata
        object.__setattr__(self, "context_dim", context_dim)
        object.__setattr__(self, "query_dim", query_dim)

        if validate_args:
            self.validate()

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
            and self.batch_first == other.batch_first
        )

    def normalize(self) -> TripletTimeData:
        # sanitize context and target values
        context_mask = self.context_valid
        query_mask = self.query_valid

        context_times = self.context_times.masked_fill(~context_mask, nan)
        context_values = self.context_values.masked_fill(~context_mask, nan)
        context_channels = self.context_channels.masked_fill(~context_mask, -1)
        query_times = self.query_times.masked_fill(~query_mask, nan)
        query_channels = self.query_channels.masked_fill(~query_mask, -1)
        target_values = (
            self.target_values.masked_fill(~query_mask, nan)
            if self.target_values is not None
            else None
        )

        return TripletTimeData(
            context_times=context_times,
            context_channels=context_channels,
            context_values=context_values,
            query_times=query_times,
            query_channels=query_channels,
            target_values=target_values,
            static_covariates=self.static_covariates,
            batch_first=self.batch_first,
            context_dim=self.context_dim,
            query_dim=self.query_dim,
            validate_args=False,  # trust that the original data was valid, normalization does not change validity
        )

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
        assert not (C_valid & ~T.isfinite()).any()  # C = True implies T is finite
        assert not (C_valid & ~X.isfinite()).any()  # C = True implies X is finite

        *_, num_query = Q.shape
        assert Q.shape == (*batch_shape, num_query)
        assert is_prefix_mask(Q.isfinite()).all()
        assert is_prefix_mask(Q.diff(dim=-1).ge(0.0)).all()

        M_valid = M >= 0
        assert M.shape == (*batch_shape, num_query)
        assert is_prefix_mask(M_valid).all()
        assert not (M_valid & ~Q.isfinite()).any()  # M = True implies Q is finite
        # query_pairs = torch.stack([Q, M], dim=-1)
        # query_pairs = query_pairs.masked_fill(~M_valid.unsqueeze(-1), nan)
        # assert torch.equal(unique_count(query_pairs), M_valid.sum(dim=-1))

        if Y is not None:
            assert Y.shape == (*batch_shape, num_query)
            assert not (M_valid & ~Y.isfinite()).any()  # M = True implies Y is finite

    def is_trimmed(self) -> bool:
        seq_dim = -1 if self.batch_first else 0
        C = self.context_valid.movedim(seq_dim, -1)
        M = self.query_valid.movedim(seq_dim, -1)
        return bool(
            C.reshape(-1, C.shape[-1]).any(dim=0).all()
            and M.reshape(-1, M.shape[-1]).any(dim=0).all()
        )

    def is_simple(self) -> bool:
        return self.is_trimmed()

    def _split_indices(
        self,
        times: Tensor,
        channels: Tensor,
        /,
        *,
        dim: int,
    ) -> tuple[Tensor, ...]:
        if not self.is_trimmed():
            raise ValueError(
                "Simple split indices are only available for trimmed data."
            )

        seq_dim = -1 if self.batch_first else 0
        T = times.movedim(seq_dim, -1)
        C = channels.movedim(seq_dim, -1)
        valid = C.ge(0)
        inverse, counts = _triplet_row_indices(T, C)
        size = int(counts.max().item())
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

    @classmethod
    def from_request(
        cls,
        *,
        context_times: Tensor,  # Float[..., $N], padded NaN
        context_mask: Tensor,  # Bool[..., $N, D]
        context_values: Tensor,  # Float[..., $N, D]
        query_times: Tensor,  # Float[..., $K], padded NaN
        query_mask: Tensor,  # Bool[..., $K, F]
        target_values: Tensor | None = None,  # Float[..., $K, F]
        static_covariates: Tensor | None = None,  # Float[..., M], padded NaN, sparse
        # extra args
        batch_first: bool = True,
        validate: bool = True,
    ) -> TripletTimeData:
        return SplitTimeData(
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=static_covariates,
            batch_first=batch_first,
            validate_args=validate,
        ).to_triplet()

    @classmethod
    def from_unbatched(
        cls, args: Collection[AbstractTripletTimeData], /, *, batch_first: bool
    ) -> TripletTimeData:
        return batch_triplet(args, batch_first=batch_first)

    @classmethod
    def from_split(
        cls, arg: AbstractSplitTimeData, /, *, batch_first: bool
    ) -> TripletTimeData:
        return split_to_triplet(arg, batch_first=batch_first)

    @classmethod
    def from_merged(
        cls, arg: AbstractMergedTimeData, /, *, batch_first: bool
    ) -> TripletTimeData:
        return merged_to_triplet(arg, batch_first=batch_first)

    @classmethod
    def from_triplet(
        cls, arg: AbstractTripletTimeData, /, *, batch_first: bool
    ) -> TripletTimeData:
        if isinstance(arg, TripletTimeData) and arg.batch_first != batch_first:
            raise ValueError("arg.batch_first does not match batch_first.")
        return TripletTimeData(
            context_times=arg.context_times,
            context_channels=arg.context_channels,
            context_values=arg.context_values,
            query_times=arg.query_times,
            query_channels=arg.query_channels,
            target_values=arg.target_values,
            static_covariates=arg.static_covariates,
            batch_first=batch_first,
        )

    def to(
        self, *, batch_first: bool | None = None, device: torch.device | None = None
    ) -> TripletTimeData:
        r"""Convert batch layout and optionally move tensors to a device.

        Args:
            batch_first: Target batch layout. Retains the current layout when None.
            device: Target device. Retains the current device when None.
        """
        if batch_first is None and device is None:
            return self

        batch_first = self.batch_first if batch_first is None else batch_first
        seq_dim = -1 if self.batch_first else 0
        tgt_dim = -1 if batch_first else 0
        return replace(
            self,
            context_times=self.context_times.movedim(seq_dim, tgt_dim).to(device),
            context_channels=self.context_channels.movedim(seq_dim, tgt_dim).to(device),
            context_values=self.context_values.movedim(seq_dim, tgt_dim).to(device),
            query_times=self.query_times.movedim(seq_dim, tgt_dim).to(device),
            query_channels=self.query_channels.movedim(seq_dim, tgt_dim).to(device),
            target_values=(
                self.target_values.movedim(seq_dim, tgt_dim).to(device)
                if self.target_values is not None
                else None
            ),
            static_covariates=(
                self.static_covariates.to(device)
                if self.static_covariates is not None
                else None
            ),
            batch_first=batch_first,
            validate_args=False,
        )

    def unbatch(self) -> list[TripletTimeData]:
        return unbatch_triplet(self, batch_first=self.batch_first)

    def to_split(
        self, *, context_dim: int | None = None, query_dim: int | None = None
    ) -> SplitTimeData:
        return triplet_to_split(
            self,
            batch_first=self.batch_first,
            context_dim=context_dim,
            query_dim=query_dim,
        )

    def to_merged(
        self, *, context_dim: int | None = None, query_dim: int | None = None
    ) -> MergedTimeData:
        return triplet_to_merged(
            self,
            batch_first=self.batch_first,
            context_dim=context_dim,
            query_dim=query_dim,
        )

    def to_triplet(self) -> TripletTimeData:
        return self


def split_to_merged(
    arg: AbstractSplitTimeData, /, *, batch_first: bool
) -> MergedTimeData:
    if isinstance(arg, SplitTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # normalize to batch_first for construction
    seq_dim = -2 if batch_first else 0
    T = arg.context_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    C = arg.context_mask.movedim(seq_dim, -2)
    X = arg.context_values.movedim(seq_dim, -2)
    Q = arg.query_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    M = arg.query_mask.movedim(seq_dim, -2)
    Y = (
        arg.target_values.movedim(seq_dim, -2)
        if arg.target_values is not None
        else None
    )

    *batch_shape, ctx_size, ctx_dim = X.shape
    *_, qry_size, qry_dim = M.shape

    timestamps = torch.cat([T, Q], dim=-1)
    permutation = torch.argsort(
        timestamps.nan_to_num(nan=torch.inf), dim=-1, stable=True
    ).unsqueeze(-1)

    return MergedTimeData(
        timestamps=(
            timestamps.take_along_dim(permutation.squeeze(-1), dim=-1)
            .unsqueeze(-1)
            .movedim(-2, seq_dim)
            .squeeze(-1)
        ),
        context_mask=(
            torch.cat([C, C.new_zeros((*batch_shape, qry_size, ctx_dim))], dim=-2)
            .take_along_dim(permutation, dim=-2)
            .movedim(-2, seq_dim)
        ),
        context_values=(
            torch.cat([X, X.new_full((*batch_shape, qry_size, ctx_dim), nan)], dim=-2)
            .take_along_dim(permutation, dim=-2)
            .movedim(-2, seq_dim)
        ),
        query_mask=(
            torch.cat([M.new_zeros((*batch_shape, ctx_size, qry_dim)), M], dim=-2)
            .take_along_dim(permutation, dim=-2)
            .movedim(-2, seq_dim)
        ),
        target_values=(
            torch.cat([Y.new_full((*batch_shape, ctx_size, qry_dim), nan), Y], dim=-2)
            .take_along_dim(permutation, dim=-2)
            .movedim(-2, seq_dim)
            if Y is not None
            else None
        ),
        static_covariates=arg.static_covariates,
        # metadata
        batch_first=batch_first,
        context_size=ctx_size,
        query_size=qry_size,
        validate_args=False,  # skip validation since we trust the arguments
    )


def split_to_triplet(
    arg: AbstractSplitTimeData, /, *, batch_first: bool
) -> TripletTimeData:
    if isinstance(arg, SplitTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # normalize to batch_first for construction
    seq_dim = -2 if batch_first else 0
    T = arg.context_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    C = arg.context_mask.movedim(seq_dim, -2)
    X = arg.context_values.movedim(seq_dim, -2)
    Q = arg.query_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    M = arg.query_mask.movedim(seq_dim, -2)
    Y = (
        arg.target_values.movedim(seq_dim, -2)
        if arg.target_values is not None
        else None
    )

    *batch_shape, _, ctx_dim = C.shape
    qry_dim = M.shape[-1]

    # `nonzero` orders entries by batch, then time, then channel. Subtracting
    # the number of earlier valid entries in the same batch converts the global
    # nonzero order into a 0-based ragged position for that batch item.
    *ctx_batch_idx, ctx_size_idx, ctx_value_idx = C.nonzero(as_tuple=True)
    counts = C.sum(dim=(-2, -1))  # (...)
    positions = torch.arange(ctx_size_idx.numel(), device=ctx_size_idx.device)
    offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
    ctx_idx = (*ctx_batch_idx, positions - offsets[*ctx_batch_idx])
    num_context = int(counts.max().item())

    *qry_batch_idx, qry_size_idx, qry_value_idx = M.nonzero(as_tuple=True)
    counts = M.sum(dim=(-2, -1))  # (...)
    positions = torch.arange(qry_size_idx.numel(), device=qry_size_idx.device)
    offsets = counts.flatten().cumsum(dim=0).reshape(batch_shape) - counts
    qry_idx = (*qry_batch_idx, positions - offsets[*qry_batch_idx])
    num_query = int(counts.max().item())

    # move the sequence dim to the target position
    target_dim = -1 if batch_first else 0

    return TripletTimeData(
        context_times=(
            T.new_full((*batch_shape, num_context), nan)
            .index_put(ctx_idx, T[*ctx_batch_idx, ctx_size_idx])
            .movedim(-1, target_dim)
        ),
        context_channels=(
            ctx_value_idx.new_full((*batch_shape, num_context), -1)
            .index_put(ctx_idx, ctx_value_idx)
            .movedim(-1, target_dim)
        ),
        context_values=(
            X.new_full((*batch_shape, num_context), nan)
            .index_put(ctx_idx, X[*ctx_batch_idx, ctx_size_idx, ctx_value_idx])
            .movedim(-1, target_dim)
        ),
        query_times=(
            Q.new_full((*batch_shape, num_query), nan)
            .index_put(qry_idx, Q[*qry_batch_idx, qry_size_idx])
            .movedim(-1, target_dim)
        ),
        query_channels=(
            qry_value_idx.new_full((*batch_shape, num_query), -1)
            .index_put(qry_idx, qry_value_idx)
            .movedim(-1, target_dim)
        ),
        target_values=(
            Y.new_full((*batch_shape, num_query), nan)
            .index_put(qry_idx, Y[*qry_batch_idx, qry_size_idx, qry_value_idx])
            .movedim(-1, target_dim)
            if Y is not None
            else None
        ),
        static_covariates=arg.static_covariates,
        # metadata
        batch_first=batch_first,
        context_dim=ctx_dim,
        query_dim=qry_dim,
        validate_args=False,  # skip validation since we trust the arguments
    )


def merged_to_split(
    arg: AbstractMergedTimeData,
    /,
    *,
    batch_first: bool,
    context_size: int | None = None,
    query_size: int | None = None,
) -> SplitTimeData:
    if isinstance(arg, MergedTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # normalize to batch_last for conversion
    seq_dim = -2 if batch_first else 0
    T = arg.timestamps.unsqueeze(-1).movedim(seq_dim, 0).squeeze(-1)
    C = arg.context_mask.movedim(seq_dim, 0)
    X = arg.context_values.movedim(seq_dim, 0)
    M = arg.query_mask.movedim(seq_dim, 0)
    Y = arg.target_values.movedim(seq_dim, 0) if arg.target_values is not None else None

    # Gather the selected steps to the front of each batch item: a stable sort
    # of `~valid` keeps the selected steps (key False) in order ahead of the
    # rest, so the first `size` columns hold them. The gathered mask/values
    # tails are already all-False/all-NaN (unselected steps), so only the
    # gathered times need their padding tail (`~keep`) reset to NaN.
    ctx_valid = C.any(dim=-1)  # ($N, ...)
    ctx_count = ctx_valid.sum(dim=0)  # (...)
    ctx_size = ctx_count.max().item() if context_size is None else context_size
    ctx_perm = torch.argsort(~ctx_valid, dim=0, stable=True)[:ctx_size]
    ctx_oob = torch.arange(ctx_size, device=T.device) >= ctx_count[..., None]
    ctx_oob = ctx_oob.movedim(-1, 0)

    qry_valid = M.any(dim=-1)
    qry_count = qry_valid.sum(dim=0)
    qry_size = qry_count.max().item() if query_size is None else query_size
    qry_perm = torch.argsort(~qry_valid, dim=0, stable=True)[:qry_size]
    qry_oob = torch.arange(qry_size, device=T.device) >= qry_count[..., None]
    qry_oob = qry_oob.movedim(-1, 0)

    return SplitTimeData(
        context_times=(
            T.take_along_dim(ctx_perm, dim=0)
            .masked_fill(ctx_oob, nan)
            .unsqueeze(-1)
            .movedim(0, seq_dim)
            .squeeze(-1)
        ),
        context_mask=C.take_along_dim(ctx_perm[..., None], dim=0).movedim(0, seq_dim),
        context_values=X.take_along_dim(ctx_perm[..., None], dim=0).movedim(0, seq_dim),
        query_times=(
            T.take_along_dim(qry_perm, dim=0)
            .masked_fill(qry_oob, nan)
            .unsqueeze(-1)
            .movedim(0, seq_dim)
            .squeeze(-1)
        ),
        query_mask=M.take_along_dim(qry_perm[..., None], dim=0).movedim(0, seq_dim),
        target_values=(
            Y.take_along_dim(qry_perm[..., None], dim=0).movedim(0, seq_dim)
            if Y is not None
            else None
        ),
        static_covariates=arg.static_covariates,
        # metadata
        batch_first=batch_first,
        validate_args=False,  # skip validation since we trust the arguments
    )


def merged_to_triplet(
    arg: AbstractMergedTimeData, /, *, batch_first: bool
) -> TripletTimeData:
    if isinstance(arg, MergedTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")
    return split_to_triplet(
        merged_to_split(arg, batch_first=batch_first),
        batch_first=batch_first,
    )


def triplet_to_split(
    arg: AbstractTripletTimeData,
    /,
    *,
    batch_first: bool,
    context_dim: int | None = None,
    query_dim: int | None = None,
) -> SplitTimeData:
    if isinstance(arg, TripletTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # normalize to batch_first for conversion
    seq_dim = -1 if batch_first else 0
    T = arg.context_times.movedim(seq_dim, -1)
    C = arg.context_channels.movedim(seq_dim, -1)
    X = arg.context_values.movedim(seq_dim, -1)
    Q = arg.query_times.movedim(seq_dim, -1)
    M = arg.query_channels.movedim(seq_dim, -1)
    Y = (
        arg.target_values.movedim(seq_dim, -1)
        if arg.target_values is not None
        else None
    )

    ctx_inverse, ctx_counts = _triplet_row_indices(T, C)
    qry_inverse, qry_counts = _triplet_row_indices(Q, M)
    ctx_size = int(ctx_counts.max().item())
    qry_size = int(qry_counts.max().item())
    ctx_dim = int(C.max().item()) + 1 if context_dim is None else context_dim
    qry_dim = int(M.max().item()) + 1 if query_dim is None else query_dim
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
    ctx_inverse = ctx_inverse.reshape(-1, num_context)
    qry_inverse = qry_inverse.reshape(-1, num_query)

    # Map context triplets to canonical dense row indices.
    ctx_steps = C_flat.ge(0)
    ctx_batch = (
        torch.arange(num_batches, device=T.device).unsqueeze(-1).expand_as(ctx_steps)
    )
    ctx_indices = (ctx_batch[ctx_steps], ctx_inverse[ctx_steps])
    ctx_channels = C_flat[ctx_steps]

    # Map query triplets to canonical dense row indices.
    qry_steps = M_flat.ge(0)
    qry_batch = (
        torch.arange(num_batches, device=Q.device).unsqueeze(-1).expand_as(qry_steps)
    )
    qry_indices = (qry_batch[qry_steps], qry_inverse[qry_steps])
    qry_channels = M_flat[qry_steps]

    target_dim = -2 if batch_first else 0
    return SplitTimeData(
        context_times=(
            T_flat.new_full((num_batches, ctx_size), nan)
            .index_put(ctx_indices, T_flat[ctx_steps])
            .reshape(*batch_shape, ctx_size, 1)
            .movedim(-2, target_dim)
            .squeeze(-1)
        ),
        context_mask=(
            C_flat.new_zeros((num_batches, ctx_size, ctx_dim), dtype=torch.bool)
            .index_put(
                (*ctx_indices, ctx_channels),
                ctx_channels.new_ones((), dtype=torch.bool),
            )
            .reshape(*batch_shape, ctx_size, ctx_dim)
            .movedim(-2, target_dim)
        ),
        context_values=(
            X_flat.new_full((num_batches, ctx_size, ctx_dim), nan)
            .index_put((*ctx_indices, ctx_channels), X_flat[ctx_steps])
            .reshape(*batch_shape, ctx_size, ctx_dim)
            .movedim(-2, target_dim)
        ),
        query_times=(
            Q_flat.new_full((num_batches, qry_size), nan)
            .index_put(qry_indices, Q_flat[qry_steps])
            .reshape(*batch_shape, qry_size, 1)
            .movedim(-2, target_dim)
            .squeeze(-1)
        ),
        query_mask=(
            M_flat.new_zeros((num_batches, qry_size, qry_dim), dtype=torch.bool)
            .index_put(
                (*qry_indices, qry_channels),
                qry_channels.new_ones((), dtype=torch.bool),
            )
            .reshape(*batch_shape, qry_size, qry_dim)
            .movedim(-2, target_dim)
        ),
        target_values=(
            Y_flat.new_full((num_batches, qry_size, qry_dim), nan)
            .index_put((*qry_indices, qry_channels), Y_flat[qry_steps])
            .reshape(*batch_shape, qry_size, qry_dim)
            .movedim(-2, target_dim)
            if Y_flat is not None
            else None
        ),
        static_covariates=arg.static_covariates,
        # metadata
        batch_first=batch_first,
        validate_args=False,  # skip validate since we trust the arguments
    )


def triplet_to_merged(
    arg: AbstractTripletTimeData,
    /,
    *,
    batch_first: bool,
    context_dim: int | None = None,
    query_dim: int | None = None,
) -> MergedTimeData:
    if isinstance(arg, TripletTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")
    return split_to_merged(
        triplet_to_split(
            arg,
            batch_first=batch_first,
            context_dim=context_dim,
            query_dim=query_dim,
        ),
        batch_first=batch_first,
    )


def batch_split(
    args: Collection[AbstractSplitTimeData], /, *, batch_first: bool
) -> SplitTimeData:
    if len(args) < 1:
        raise ValueError("Expected at least one argument.")

    return SplitTimeData(
        context_times=pad_sequence(
            [arg.context_times for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        context_mask=pad_sequence(
            [arg.context_mask for arg in args],
            batch_first=batch_first,
            padding_value=False,
        ),
        context_values=pad_sequence(
            [arg.context_values for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        query_times=pad_sequence(
            [arg.query_times for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        query_mask=pad_sequence(
            [arg.query_mask for arg in args],
            batch_first=batch_first,
            padding_value=False,
        ),
        target_values=(
            pad_sequence(Y, batch_first=batch_first, padding_value=nan)
            if (Y := _all_or_none(arg.target_values for arg in args)) is not None
            else None
        ),
        static_covariates=(
            torch.stack(S)
            if (S := _all_or_none(arg.static_covariates for arg in args)) is not None
            else None
        ),
        # metadata
        batch_first=batch_first,
        validate_args=False,  # skip validation since we trust the arguments
    )


def batch_merged(
    args: Collection[AbstractMergedTimeData], /, *, batch_first: bool
) -> MergedTimeData:
    if len(args) < 1:
        raise ValueError("Expected at least one argument.")

    return MergedTimeData(
        timestamps=pad_sequence(
            [arg.timestamps for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        context_values=pad_sequence(
            [arg.context_values for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        context_mask=pad_sequence(
            [arg.context_mask for arg in args],
            batch_first=batch_first,
            padding_value=False,
        ),
        query_mask=pad_sequence(
            [arg.query_mask for arg in args],
            batch_first=batch_first,
            padding_value=False,
        ),
        target_values=(
            pad_sequence(Y, batch_first=batch_first, padding_value=nan)
            if (Y := _all_or_none(arg.target_values for arg in args)) is not None
            else None
        ),
        static_covariates=(
            torch.stack(S)
            if (S := _all_or_none(arg.static_covariates for arg in args)) is not None
            else None
        ),
        # metadata
        batch_first=batch_first,
        validate_args=False,  # skip validation since we trust the arguments
    )


def batch_triplet(
    args: Collection[AbstractTripletTimeData],
    /,
    *,
    batch_first: bool,
    query_dim: int | None = None,
    context_dim: int | None = None,
) -> TripletTimeData:
    if len(args) < 1:
        raise ValueError("Expected at least one argument.")

    return TripletTimeData(
        context_times=pad_sequence(
            [arg.context_times for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        context_channels=pad_sequence(
            [arg.context_channels for arg in args],
            batch_first=batch_first,
            padding_value=-1,
        ),
        context_values=pad_sequence(
            [arg.context_values for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        query_times=pad_sequence(
            [arg.query_times for arg in args],
            batch_first=batch_first,
            padding_value=nan,
        ),
        query_channels=pad_sequence(
            [arg.query_channels for arg in args],
            batch_first=batch_first,
            padding_value=-1,
        ),
        target_values=(
            pad_sequence(Y, batch_first=batch_first, padding_value=nan)
            if (Y := _all_or_none(arg.target_values for arg in args)) is not None
            else None
        ),
        static_covariates=(
            torch.stack(S)
            if (S := _all_or_none(arg.static_covariates for arg in args)) is not None
            else None
        ),
        # metadata
        batch_first=batch_first,
        context_dim=-1 if context_dim is None else context_dim,
        query_dim=-1 if query_dim is None else query_dim,
        validate_args=False,  # skip validation since we trust the arguments
    )


def unbatch_split(
    arg: AbstractSplitTimeData, /, *, batch_first: bool
) -> list[SplitTimeData]:
    if isinstance(arg, SplitTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # convert to batch_first for unpadding
    seq_dim = -2 if batch_first else 0
    T = arg.context_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    C = arg.context_mask.movedim(seq_dim, -2)
    X = arg.context_values.movedim(seq_dim, -2)
    Q = arg.query_times.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    M = arg.query_mask.movedim(seq_dim, -2)
    Y = (
        arg.target_values.movedim(seq_dim, -2)
        if arg.target_values is not None
        else None
    )

    # flatten the batch dimensions to simplify unpadding, then unpad each sample
    T = T.unsqueeze(0).flatten(end_dim=-2)
    C = C.unsqueeze(0).flatten(end_dim=-3)
    X = X.unsqueeze(0).flatten(end_dim=-3)
    Q = Q.unsqueeze(0).flatten(end_dim=-2)
    M = M.unsqueeze(0).flatten(end_dim=-3)
    Y = Y.unsqueeze(0).flatten(end_dim=-3) if Y is not None else None

    context_lengths = T.isfinite().sum(dim=-1)
    query_lengths = Q.isfinite().sum(dim=-1)
    num_samples = T.shape[0]

    return [
        SplitTimeData(
            context_times=c_time,
            context_values=c_value,
            context_mask=c_mask,
            query_times=q_time,
            query_mask=q_mask,
            target_values=q_value,
            static_covariates=static_arg,
            # metadata
            batch_first=batch_first,
            validate_args=False,  # skip validation since we trust the arguments
        )
        for c_time, c_mask, c_value, q_time, q_mask, q_value, static_arg in zip(
            unpad_sequence(T, context_lengths, batch_first=True),
            unpad_sequence(C, context_lengths, batch_first=True),
            unpad_sequence(X, context_lengths, batch_first=True),
            unpad_sequence(Q, query_lengths, batch_first=True),
            unpad_sequence(M, query_lengths, batch_first=True),
            (
                unpad_sequence(Y, query_lengths, batch_first=True)
                if Y is not None
                else [None] * num_samples
            ),
            (
                arg.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                if arg.static_covariates is not None
                else [None] * num_samples
            ),
            strict=True,
        )
    ]


def unbatch_merged(
    arg: AbstractMergedTimeData, /, *, batch_first: bool
) -> list[MergedTimeData]:
    if isinstance(arg, MergedTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # convert to batch_first for unpadding
    seq_dim = -2 if batch_first else 0
    T = arg.timestamps.unsqueeze(-1).movedim(seq_dim, -2).squeeze(-1)
    C = arg.context_mask.movedim(seq_dim, -2)
    X = arg.context_values.movedim(seq_dim, -2)
    M = arg.query_mask.movedim(seq_dim, -2)
    Y = (
        arg.target_values.movedim(seq_dim, -2)
        if arg.target_values is not None
        else None
    )

    # flatten the batch dimensions to simplify unpadding, then unpad each sample
    T_flat = T.unsqueeze(0).flatten(end_dim=-2)
    C_flat = C.unsqueeze(0).flatten(end_dim=-3)
    X_flat = X.unsqueeze(0).flatten(end_dim=-3)
    M_flat = M.unsqueeze(0).flatten(end_dim=-3)
    Y_flat = Y.unsqueeze(0).flatten(end_dim=-3) if Y is not None else None

    lengths = T_flat.isfinite().sum(dim=-1)

    return [
        MergedTimeData(
            timestamps=time,
            context_values=c_value,
            context_mask=c_mask,
            target_values=q_value,
            query_mask=q_mask,
            static_covariates=static_arg,
            # metadata
            batch_first=batch_first,
            validate_args=False,  # skip validation since we trust the arguments
        )
        for time, c_mask, c_value, q_mask, q_value, static_arg in zip(
            unpad_sequence(T_flat, lengths, batch_first=True),
            unpad_sequence(C_flat, lengths, batch_first=True),
            unpad_sequence(X_flat, lengths, batch_first=True),
            unpad_sequence(M_flat, lengths, batch_first=True),
            (
                unpad_sequence(Y_flat, lengths, batch_first=True)
                if Y_flat is not None
                else [None] * len(T_flat)
            ),
            (
                arg.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                if arg.static_covariates is not None
                else [None] * len(T_flat)
            ),
            strict=True,
        )
    ]


def unbatch_triplet(
    arg: AbstractTripletTimeData, /, *, batch_first: bool
) -> list[TripletTimeData]:
    # normalize to batch_first for conversion & flatten batch dimensions
    if isinstance(arg, TripletTimeData) and arg.batch_first != batch_first:
        raise ValueError("arg.batch_first does not match batch_first.")

    # convert to batch_first for unpadding
    seq_dim = -1 if batch_first else 0
    T = arg.context_times.movedim(seq_dim, -1)
    C = arg.context_channels.movedim(seq_dim, -1)
    X = arg.context_values.movedim(seq_dim, -1)
    Q = arg.query_times.movedim(seq_dim, -1)
    M = arg.query_channels.movedim(seq_dim, -1)
    Y = (
        arg.target_values.movedim(seq_dim, -1)
        if arg.target_values is not None
        else None
    )

    # flatten the batch dimensions to simplify unpadding, then unpad each sample
    T_flat = T.unsqueeze(0).flatten(end_dim=-2)
    C_flat = C.unsqueeze(0).flatten(end_dim=-2)
    X_flat = X.unsqueeze(0).flatten(end_dim=-2)
    Q_flat = Q.unsqueeze(0).flatten(end_dim=-2)
    M_flat = M.unsqueeze(0).flatten(end_dim=-2)
    Y_flat = Y.unsqueeze(0).flatten(end_dim=-2) if Y is not None else None

    context_lengths = T_flat.isfinite().sum(dim=-1)
    query_lengths = Q_flat.isfinite().sum(dim=-1)

    return [
        TripletTimeData(
            context_times=c_time,
            context_channels=c_channel,
            context_values=c_value,
            query_times=q_time,
            query_channels=q_channel,
            target_values=q_value,
            static_covariates=static_arg,
            # metadata
            batch_first=batch_first,
            validate_args=False,  # skip validation since we trust the arguments
        )
        for c_time, c_channel, c_value, q_time, q_channel, q_value, static_arg in zip(
            unpad_sequence(T_flat, context_lengths, batch_first=True),
            unpad_sequence(C_flat, context_lengths, batch_first=True),
            unpad_sequence(X_flat, context_lengths, batch_first=True),
            unpad_sequence(Q_flat, query_lengths, batch_first=True),
            unpad_sequence(M_flat, query_lengths, batch_first=True),
            (
                unpad_sequence(Y_flat, query_lengths, batch_first=True)
                if Y_flat is not None
                else [None] * len(Q_flat)
            ),
            (
                arg.static_covariates.unsqueeze(0).flatten(end_dim=-2)
                if arg.static_covariates is not None
                else [None] * len(T_flat)
            ),
            strict=True,
        )
    ]
