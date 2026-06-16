r"""Utility classes for forecasting."""

__all__ = [
    "BatchedCombinedArgs",
    "BatchedDenseArgs",
    "BatchedTripletArgs",
    "TripletArg",
    "DenseArg",
    "CombinedArg",
    "all_or_none",
    "unique_count",
    "is_prefix_mask",
    "scatter_fill",
]


from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


def all_or_none[T](vals: Iterable[T | None], /) -> list[T] | None:
    result = []
    has_none = False
    for arg in vals:
        if arg is None:
            has_none = True
        else:
            result.append(arg)
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


def consecutive_group_indices(times: Tensor, valid: Tensor, /) -> tuple[Tensor, Tensor]:
    r"""Compute group indices for consecutive equal time stamps."""
    is_new = torch.ones_like(valid)
    is_new[..., 1:] = times[..., 1:] != times[..., :-1]
    is_new &= valid
    inverse = is_new.cumsum(dim=-1) - 1
    return inverse.masked_fill(~valid, -1), is_new.sum(dim=-1)


def scatter_fill(
    shape: Sequence[int],
    indices: tuple[Tensor, ...],
    reference: Tensor,
    fill_value: bool | float,
    /,
) -> Tensor:
    r"""Create a filled tensor and write reference values at the given indices."""
    result = reference.new_full(shape, fill_value)
    result[indices] = reference
    return result


@dataclass(frozen=True)
class DenseArg:
    r"""Dense representation of forecasting arguments.

    Assumptions:
        - context time stamps are finite and non-decreasing
        - query time stamps are finite and strictly increasing
        - if query mask is not given, it is assumed to be a full true tensor
        - if query values are given, they are finite at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one query value observed per time stamp
        - there is at least on query mask selected per time stamp
    """

    context_times: Tensor  # Float[(N)], finite
    context_values: Tensor  # Float[(N, D)], sparse

    query_times: Tensor  # Float[(K)], finite
    query_mask: Tensor | None = None  # Bool[(K, F)]
    query_values: Tensor | None = None  # Float[(K, F)], sparse

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    @classmethod
    def from_triplet(cls, arg: TripletArg, /) -> DenseArg:
        return arg.to_dense()

    @classmethod
    def from_combined(cls, arg: CombinedArg, /) -> DenseArg:
        return arg.to_dense()

    @classmethod
    def from_batched(cls, arg: BatchedDenseArgs, /) -> list[DenseArg]:
        return arg.unbatch()

    def __post_init__(self) -> None:
        T = self.context_times
        X = self.context_values
        Q = self.query_times

        *_, context_size, context_dim = X.shape
        assert T.shape == (context_size,)
        assert T.isfinite().all()
        assert (T.diff(dim=-1) >= 0.0).all()  # non-decreasing
        assert X.shape == (context_size, context_dim)
        assert X.isfinite().any(dim=-1).all()  # at least one value per step

        *_, query_size = Q.shape
        assert Q.shape == (query_size,)
        assert Q.isfinite().all()
        assert (Q.diff(dim=-1) > 0.0).all()  # strictly increasing

        if (M := self.query_mask) is not None:
            *_, query_dim = M.shape
            assert M.dtype == torch.bool
            assert M.shape == (query_size, query_dim)
            assert M.any(dim=-1).all()  # at least one value per step

        if (V := self.query_values) is not None:
            *_, query_dim = V.shape
            assert V.shape == (query_size, query_dim)
            assert V.isfinite().any(dim=-1).all()  # at least one value per step

            if (mask := self.query_mask) is None:
                assert V.isfinite().all()
            else:
                assert V.shape == mask.shape
                assert V[mask].isfinite().all()

    def to_triplet(self) -> TripletArg:
        T = self.context_times
        X = self.context_values
        Q = self.query_times
        M = self.query_mask
        Y = self.query_values

        time_indices, context_channels = X.isfinite().nonzero(as_tuple=True)  # O×2
        context_times = T[time_indices]
        context_values = X[time_indices, context_channels]

        query_mask = (
            M
            if M is not None
            else torch.ones(
                (Q.shape[0], Y.shape[-1] if Y is not None else X.shape[-1]),
                dtype=torch.bool,
                device=Q.device,
            )
        )
        query_indices, query_channels = query_mask.nonzero(as_tuple=True)
        query_times = Q[query_indices]
        query_values = Y[query_indices, query_channels] if Y is not None else None

        return TripletArg(
            context_times=context_times,
            context_channels=context_channels,
            context_values=context_values,
            query_times=query_times,
            query_channels=query_channels,
            query_values=query_values,
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> CombinedArg:
        raise NotImplementedError


@dataclass(frozen=True)
class BatchedDenseArgs:
    r"""Batched forecasting arguments.

    Shapes:
        K: max(Kᵢ) query size
        N: max(Nᵢ) context size
        D: input dimensionality
        F: output dimensionality
        M: static covariate dimensionality

    Assumptions: (up to tail padding)
        - context time stamps are finite and non-decreasing
        - query time stamps are finite and strictly increasing
        - if query mask is not given, it is assumed to be a full true tensor
        - if query values are given, they are finite at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one query value observed per time stamp
    """

    context_times: Tensor  # Float[(..., N)], padded NaN
    context_values: Tensor  # Float[(..., N, D)], padded NaN, sparse

    query_times: Tensor  # Float[(..., K)], padded NaN
    query_mask: Tensor | None = None  # Bool[(..., K, F)]  padded False
    query_values: Tensor | None = None  # Float[(..., K, F)]  padded NaN, sparse

    static_covariates: Tensor | None = None  # Float[(..., M)]  padded NaN, sparse

    @classmethod
    def from_combined(cls, arg: BatchedCombinedArgs, /) -> BatchedDenseArgs:
        return arg.to_dense()

    @classmethod
    def from_triplet(cls, arg: BatchedTripletArgs, /) -> BatchedDenseArgs:
        return arg.to_dense()

    def __post_init__(self) -> None:
        T = self.context_times
        X = self.context_values
        Q = self.query_times

        # check shapes
        *batch_shape, context_size, _ = X.shape
        *_, query_size = Q.shape
        assert T.shape == (*batch_shape, context_size)
        assert Q.shape == (*batch_shape, query_size)

        # check that non-valid values are at the tail
        T_valid = T.isfinite()
        Q_valid = Q.isfinite()
        X_valid = X.isfinite().any(dim=-1)  # at least one value per step
        assert is_prefix_mask(Q_valid).all()
        assert is_prefix_mask(T_valid).all()
        assert is_prefix_mask(X_valid).all()
        Q_increasing = Q.diff(dim=-1).gt(0.0)  # query times are strictly increasing
        T_increasing = T.diff(dim=-1).ge(0.0)  # context times are non-decreasing
        assert is_prefix_mask(Q_increasing).all()
        assert is_prefix_mask(T_increasing).all()

        # check padding
        context_lengths = T.isnan().sum(dim=-1)
        assert torch.equal(X.isnan().all(dim=-1).sum(dim=-1), context_lengths)

        if (M := self.query_mask) is not None:
            *_, query_dim = M.shape
            assert M.dtype == torch.bool
            assert M.shape == (*batch_shape, query_size, query_dim)
            M_valid = M.isfinite().all(dim=-1)  # at least one value per step
            assert is_prefix_mask(M_valid).all()

        if (V := self.query_values) is not None:
            *_, query_dim = V.shape
            assert V.shape == (*batch_shape, query_size, query_dim)
            V_valid = V.isfinite()

            if (mask := self.query_mask) is None:
                assert V_valid.all()
            else:
                assert mask.shape == V.shape
                assert V_valid[mask].all()

        if (S := self.static_covariates) is not None:
            *_, static_dim = S.shape
            assert S.shape == (*batch_shape, static_dim)

    @classmethod
    def from_unbatched(cls, args: Sequence[DenseArg], /) -> BatchedDenseArgs:
        if not args:
            raise ValueError("Expected at least one DenseArg.")

        query_mask = (
            None
            if (M := all_or_none(arg.query_mask for arg in args)) is None
            else pad_sequence(M, batch_first=True, padding_value=False)
        )

        query_values = (
            None
            if (V := all_or_none(arg.query_values for arg in args)) is None
            else pad_sequence(V, batch_first=True, padding_value=torch.nan)
        )

        static_covariates = (
            None
            if (S := all_or_none(arg.static_covariates for arg in args)) is None
            else torch.stack(S)
        )

        return cls(
            context_times=pad_sequence(
                [arg.context_times for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            context_values=pad_sequence(
                [arg.context_values for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            query_times=pad_sequence(
                [arg.query_times for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            query_mask=query_mask,
            query_values=query_values,
            static_covariates=static_covariates,
        )

    def unbatch(self) -> list[DenseArg]:
        T = self.context_times.unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-3)
        Q = self.query_times.unsqueeze(0).flatten(end_dim=-2)
        query_mask = (
            None
            if self.query_mask is None
            else self.query_mask.unsqueeze(0).flatten(end_dim=-3)
        )
        query_values = (
            None
            if self.query_values is None
            else self.query_values.unsqueeze(0).flatten(end_dim=-3)
        )
        static_covariates = (
            None
            if self.static_covariates is None
            else self.static_covariates.unsqueeze(0).flatten(end_dim=-2)
        )

        context_lengths = (~T.isnan()).sum(dim=-1)
        query_lengths = (~Q.isnan()).sum(dim=-1)
        num_samples = T.shape[0]

        context_times = unpad_sequence(T, context_lengths, batch_first=True)
        context_values = unpad_sequence(X, context_lengths, batch_first=True)
        query_times = unpad_sequence(Q, query_lengths, batch_first=True)
        query_masks = (
            [None] * num_samples
            if query_mask is None
            else unpad_sequence(query_mask, query_lengths, batch_first=True)
        )
        query_values = (
            [None] * num_samples
            if query_values is None
            else unpad_sequence(query_values, query_lengths, batch_first=True)
        )
        static_args = (
            [None] * num_samples
            if static_covariates is None
            else list(static_covariates.unbind(dim=0))
        )

        return [
            DenseArg(
                context_times=context_time,
                context_values=context_value,
                query_times=query_time,
                query_mask=query_mask,
                query_values=query_value,
                static_covariates=static_arg,
            )
            for context_time, context_value, query_time, query_mask, query_value, static_arg in zip(
                context_times,
                context_values,
                query_times,
                query_masks,
                query_values,
                static_args,
                strict=True,
            )
        ]

    def to_triplet(self) -> BatchedTripletArgs:
        T = self.context_times
        X = self.context_values
        Q = self.query_times
        Y = self.query_values
        batch_shape = T.shape[:-1]
        context_size, input_dim = X.shape[-2:]
        query_size = Q.shape[-1]

        T_flat = T.reshape(-1, context_size)
        X_flat = X.reshape(-1, context_size, input_dim)
        Q_flat = Q.reshape(-1, query_size)
        Y_flat = None if Y is None else Y.reshape(-1, query_size, Y.shape[-1])

        X_valid = X_flat.isfinite() & T_flat.isfinite().unsqueeze(-1)
        context_mask = X_valid.flatten(start_dim=1)
        context_counts = context_mask.sum(dim=-1)
        num_context = int(context_counts.max().item())
        context_positions = context_mask.cumsum(dim=-1) - 1

        context_times = T.new_full((T_flat.shape[0], num_context), torch.nan)
        context_channels = torch.full(
            (T_flat.shape[0], num_context),
            -1,
            dtype=torch.long,
            device=X.device,
        )
        context_values = X.new_full((T_flat.shape[0], num_context), torch.nan)

        batch_indices, flat_indices = context_mask.nonzero(as_tuple=True)
        target_indices = context_positions[batch_indices, flat_indices]
        time_indices = flat_indices.div(input_dim, rounding_mode="floor")
        channel_indices = flat_indices.remainder(input_dim)

        context_times[batch_indices, target_indices] = T_flat[
            batch_indices, time_indices
        ]
        context_channels[batch_indices, target_indices] = channel_indices
        context_values[batch_indices, target_indices] = X_flat[
            batch_indices, time_indices, channel_indices
        ]

        if self.query_mask is None:
            query_dim = input_dim if Y is None else Y.shape[-1]
            query_valid = Q_flat.isfinite().unsqueeze(-1).expand(-1, -1, query_dim)
        else:
            query_dim = self.query_mask.shape[-1]
            query_valid = self.query_mask.reshape(
                -1, query_size, query_dim
            ) & Q_flat.isfinite().unsqueeze(-1)

        query_mask = query_valid.flatten(start_dim=1)
        query_counts = query_mask.sum(dim=-1)
        num_query = int(query_counts.max().item())
        query_positions = query_mask.cumsum(dim=-1) - 1

        query_times = Q.new_full((Q_flat.shape[0], num_query), torch.nan)
        query_channels = torch.full(
            (Q_flat.shape[0], num_query),
            -1,
            dtype=torch.long,
            device=Q.device,
        )
        query_values = (
            None
            if Y_flat is None
            else Q.new_full((Q_flat.shape[0], num_query), torch.nan)
        )

        batch_indices, flat_indices = query_mask.nonzero(as_tuple=True)
        target_indices = query_positions[batch_indices, flat_indices]
        time_indices = flat_indices.div(query_dim, rounding_mode="floor")
        channel_indices = flat_indices.remainder(query_dim)

        query_times[batch_indices, target_indices] = Q_flat[batch_indices, time_indices]
        query_channels[batch_indices, target_indices] = channel_indices
        if query_values is not None and Y_flat is not None:
            query_values[batch_indices, target_indices] = Y_flat[
                batch_indices, time_indices, channel_indices
            ]

        return BatchedTripletArgs(
            context_times=context_times.reshape(*batch_shape, num_context),
            context_channels=context_channels.reshape(*batch_shape, num_context),
            context_values=context_values.reshape(*batch_shape, num_context),
            query_times=query_times.reshape(*batch_shape, num_query),
            query_channels=query_channels.reshape(*batch_shape, num_query),
            query_values=(
                None
                if query_values is None
                else query_values.reshape(*batch_shape, num_query)
            ),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> BatchedCombinedArgs:
        raise NotImplementedError


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
    query_values: Tensor | None = None  # Float[(Qᵢ)], finite

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    @classmethod
    def from_dense(cls, arg: DenseArg, /) -> TripletArg:
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

        if (V := self.query_values) is not None:
            assert V.shape == (num_query,)
            assert V.isfinite().all()

    def to_dense(
        self,
        /,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> DenseArg:
        C = self.context_channels
        M = self.query_channels

        context_dim = int(C.max().item()) + 1 if context_dim is None else context_dim
        query_dim = int(M.max().item()) + 1 if query_dim is None else query_dim

        if (self.context_channels >= context_dim).any():
            raise ValueError("Expected context channel indices below context_dim.")
        if (self.query_channels >= query_dim).any():
            raise ValueError("Expected query channel indices below query_dim.")

        context_times, context_inverse = torch.unique_consecutive(
            self.context_times,
            return_inverse=True,
        )
        context_values = scatter_fill(
            (context_times.shape[0], context_dim),
            (context_inverse, self.context_channels),
            self.context_values,
            torch.nan,
        )

        query_times, query_inverse = torch.unique_consecutive(
            self.query_times,
            return_inverse=True,
        )
        query_mask = scatter_fill(
            (query_times.shape[0], query_dim),
            (query_inverse, self.query_channels),
            torch.ones_like(self.query_channels, dtype=torch.bool),
            False,
        )

        query_values = (
            None
            if self.query_values is None
            else scatter_fill(
                (query_times.shape[0], query_dim),
                (query_inverse, self.query_channels),
                self.query_values,
                torch.nan,
            )
        )

        return DenseArg(
            context_times=context_times,
            context_values=context_values,
            query_times=query_times,
            query_mask=query_mask,
            query_values=query_values,
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> CombinedArg:
        raise NotImplementedError


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
    query_values: Tensor | None = None  # Float[(..., Q)], padded NaN

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

        *_, num_query = self.query_times.shape
        assert Q.shape == (*batch_shape, num_query)
        assert is_prefix_mask(Q.isfinite()).all()
        assert is_prefix_mask(Q.diff(dim=-1).ge(0.0)).all()

        M_valid = M >= 0
        assert M.shape == (*batch_shape, num_query)
        assert is_prefix_mask(M_valid).all()
        query_pairs = torch.stack([Q, M], dim=-1)
        query_pairs = query_pairs.masked_fill(~M_valid.unsqueeze(-1), torch.nan)
        assert torch.equal(unique_count(query_pairs), M_valid.sum(dim=-1))

        if (V := self.query_values) is not None:
            V_valid = V.isfinite()
            assert V.shape == (*batch_shape, num_query)
            assert torch.equal(V_valid, M_valid)

    @classmethod
    def from_combined(cls, arg: BatchedCombinedArgs, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_dense(cls, arg: BatchedDenseArgs, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_unbatched(cls, args: Sequence[TripletArg]) -> BatchedTripletArgs:
        if not args:
            raise ValueError("Expected at least one TripletArg.")

        query_channels = all_or_none(arg.query_channels for arg in args)
        if query_channels is None:
            raise ValueError("Expected query channels for batched triplet arguments.")

        query_values = (
            None
            if (V := all_or_none(arg.query_values for arg in args)) is None
            else pad_sequence(V, batch_first=True, padding_value=torch.nan)
        )

        static_covariates = (
            None
            if (S := all_or_none(arg.static_covariates for arg in args)) is None
            else torch.stack(S)
        )

        return cls(
            context_times=pad_sequence(
                [arg.context_times for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            context_channels=pad_sequence(
                [arg.context_channels for arg in args],
                batch_first=True,
                padding_value=-1,
            ),
            context_values=pad_sequence(
                [arg.context_values for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            query_times=pad_sequence(
                [arg.query_times for arg in args],
                batch_first=True,
                padding_value=torch.nan,
            ),
            query_channels=pad_sequence(
                query_channels,
                batch_first=True,
                padding_value=-1,
            ),
            query_values=query_values,
            static_covariates=static_covariates,
        )

    def unbatch(self) -> list[TripletArg]:
        T = self.context_times.unsqueeze(0).flatten(end_dim=-2)
        C = self.context_channels.unsqueeze(0).flatten(end_dim=-2)
        X = self.context_values.unsqueeze(0).flatten(end_dim=-2)
        Q = self.query_times.unsqueeze(0).flatten(end_dim=-2)
        M = self.query_channels.unsqueeze(0).flatten(end_dim=-2)
        query_values = (
            None
            if self.query_values is None
            else self.query_values.unsqueeze(0).flatten(end_dim=-2)
        )
        static_covariates = (
            None
            if self.static_covariates is None
            else self.static_covariates.unsqueeze(0).flatten(end_dim=-2)
        )

        context_lengths = T.isfinite().sum(dim=-1)
        query_lengths = M.ge(0).sum(dim=-1)
        num_samples = T.shape[0]

        context_times = unpad_sequence(T, context_lengths, batch_first=True)
        context_channels = unpad_sequence(C, context_lengths, batch_first=True)
        context_values = unpad_sequence(X, context_lengths, batch_first=True)
        query_times = unpad_sequence(Q, query_lengths, batch_first=True)
        query_channels = unpad_sequence(M, query_lengths, batch_first=True)
        query_values = (
            [None] * num_samples
            if query_values is None
            else unpad_sequence(query_values, query_lengths, batch_first=True)
        )
        static_args = (
            [None] * num_samples
            if static_covariates is None
            else list(static_covariates.unbind(dim=0))
        )

        return [
            TripletArg(
                context_times=context_time,
                context_channels=context_channel,
                context_values=context_value,
                query_times=query_time,
                query_channels=query_channel,
                query_values=query_value,
                static_covariates=static_arg,
            )
            for context_time, context_channel, context_value, query_time, query_channel, query_value, static_arg in zip(
                context_times,
                context_channels,
                context_values,
                query_times,
                query_channels,
                query_values,
                static_args,
                strict=True,
            )
        ]

    def to_dense(
        self,
        /,
        *,
        context_dim: int | None = None,
        query_dim: int | None = None,
    ) -> BatchedDenseArgs:
        T = self.context_times
        C = self.context_channels
        X = self.context_values
        Q = self.query_times
        M = self.query_channels
        Y = self.query_values

        batch_shape = T.shape[:-1]
        *_, num_context = T.shape
        *_, num_query = Q.shape

        T_flat = T.reshape(-1, num_context)
        C_flat = C.reshape(-1, num_context)
        X_flat = X.reshape(-1, num_context)
        Q_flat = Q.reshape(-1, num_query)
        M_flat = M.reshape(-1, num_query)
        Y_flat = None if Y is None else Y.reshape(-1, num_query)
        num_batches = T_flat.shape[0]
        query_valid = Q_flat.isfinite() & M_flat.ge(0)
        context_valid = T_flat.isfinite() & C_flat.ge(0) & X_flat.isfinite()

        context_dim = (
            int(C_flat[context_valid].max().item()) + 1
            if context_dim is None
            else context_dim
        )
        query_dim = (
            int(M_flat[query_valid].max().item()) + 1
            if query_dim is None
            else query_dim
        )
        if (self.context_channels >= context_dim).any():
            raise ValueError("Expected context channel indices below context_dim.")
        if (self.query_channels >= query_dim).any():
            raise ValueError("Expected query channel indices below query_dim.")

        context_inverse, context_lengths = consecutive_group_indices(
            T_flat,
            context_valid,
        )
        context_size = int(context_lengths.max().item())
        context_batch = torch.arange(num_batches, device=T.device)
        context_batch = context_batch.reshape(-1, 1).expand(-1, num_context)
        context_indices = (
            context_batch[context_valid],
            context_inverse[context_valid],
        )
        context_times = scatter_fill(
            (num_batches, context_size),
            context_indices,
            T_flat[context_valid],
            torch.nan,
        )
        context_values = scatter_fill(
            (num_batches, context_size, context_dim),
            (*context_indices, C_flat[context_valid]),
            X_flat[context_valid],
            torch.nan,
        )

        query_inverse, query_lengths = consecutive_group_indices(Q_flat, query_valid)
        query_size = int(query_lengths.max().item())
        query_batch = torch.arange(num_batches, device=Q.device)
        query_batch = query_batch.reshape(-1, 1).expand(-1, num_query)
        query_indices = (
            query_batch[query_valid],
            query_inverse[query_valid],
        )
        query_channels = M_flat[query_valid]
        query_times = scatter_fill(
            (num_batches, query_size),
            query_indices,
            Q_flat[query_valid],
            torch.nan,
        )
        query_mask = scatter_fill(
            (num_batches, query_size, query_dim),
            (*query_indices, query_channels),
            torch.ones_like(query_channels, dtype=torch.bool),
            False,
        )
        query_values = (
            None
            if Y_flat is None
            else scatter_fill(
                (num_batches, query_size, query_dim),
                (*query_indices, query_channels),
                Y_flat[query_valid],
                torch.nan,
            )
        )

        return BatchedDenseArgs(
            context_times=context_times.reshape(*batch_shape, context_size),
            context_values=context_values.reshape(
                *batch_shape, context_size, context_dim
            ),
            query_times=query_times.reshape(*batch_shape, query_size),
            query_mask=query_mask.reshape(*batch_shape, query_size, query_dim),
            query_values=(
                None
                if query_values is None
                else query_values.reshape(*batch_shape, query_size, query_dim)
            ),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> BatchedCombinedArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class CombinedArg:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        N: context size
        K: query size
        D: data dimensionality
    """

    times: Tensor  # Float[(N + K)], finite
    values: Tensor  # Float[(N + K, D)], sparse
    context_mask: Tensor  # Bool[(N + K, D)]
    query_mask: Tensor  # Bool[(N + K, D)]

    static_covariates: Tensor | None = None  # Float[(M)], sparse

    def __post_init__(self) -> None:
        T = self.times
        V = self.values
        M = self.context_mask
        Q = self.query_mask
        *_, num_combined, num_dim = V.shape
        assert T.shape == (num_combined,)
        assert T.isfinite().all()
        assert T.diff(dim=-1).ge(0.0).all()  # sorted in ascending order

        assert V.shape == (num_combined, num_dim)
        assert V.isfinite().any(dim=-1).all()  # at least one value per step

        assert M.dtype == torch.bool
        assert Q.dtype == torch.bool
        assert M.shape == (num_combined, num_dim)
        assert Q.shape == (num_combined, num_dim)
        assert (M.any(dim=-1) | Q.any(dim=-1)).all()  # at least one value per step

    @classmethod
    def from_dense(cls, arg: DenseArg, /) -> CombinedArg:
        return arg.to_combined()

    @classmethod
    def from_triplet(cls, arg: TripletArg, /) -> CombinedArg:
        return arg.to_combined()

    @classmethod
    def from_batched(cls, arg: BatchedCombinedArgs, /) -> list[CombinedArg]:
        return arg.unbatch()

    def to_dense(self) -> DenseArg:
        raise NotImplementedError

    def to_triplet(self) -> TripletArg:
        raise NotImplementedError


@dataclass(frozen=True)
class BatchedCombinedArgs:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        N: max(Nᵢ) context size
        K: max(Kᵢ) query size
        E: combined data dimensionality
        M: static covariate dimensionality
    """

    times: Tensor  # Float[(..., N + K)], padded NaN
    values: Tensor  # Float[(..., N + K, E)], padded NaN, sparse
    context_mask: Tensor  # Bool[(..., N + K, E)], padded False
    query_mask: Tensor  # Bool[(..., N + K, E)], padded False

    static_covariates: Tensor | None = None  # Float[(..., M)], padded NaN, sparse

    def __post_init__(self) -> None:
        T = self.times
        V = self.values
        M = self.context_mask
        Q = self.query_mask
        *batch_shape, num_combined, num_dim = V.shape
        T_valid = T.isfinite()
        T_ascending = T.diff(dim=-1).ge(0.0)
        assert T.shape == (*batch_shape, num_combined)
        assert is_prefix_mask(T_valid)
        assert is_prefix_mask(T_ascending).all()  # sorted in ascending order

        V_valid = V.isfinite().any(dim=-1)
        assert V.shape == (*batch_shape, num_combined, num_dim)
        assert is_prefix_mask(V_valid).all()  # at least one value per step

        mask_valid = M.any(dim=-1) | Q.any(dim=-1)
        assert M.dtype == torch.bool
        assert Q.dtype == torch.bool
        assert M.shape == (*batch_shape, num_combined, num_dim)
        assert Q.shape == (*batch_shape, num_combined, num_dim)
        assert is_prefix_mask(mask_valid).all()  # at least one value per step

    @classmethod
    def from_unbatched(cls, args: Sequence[CombinedArg], /) -> BatchedCombinedArgs:
        raise NotImplementedError

    @classmethod
    def from_dense(cls, arg: BatchedDenseArgs, /) -> BatchedCombinedArgs:
        return arg.to_combined()

    @classmethod
    def from_triplet(cls, arg: BatchedTripletArgs, /) -> BatchedCombinedArgs:
        return arg.to_combined()

    def unbatch(self) -> list[CombinedArg]:
        raise NotImplementedError

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> BatchedTripletArgs:
        raise NotImplementedError
