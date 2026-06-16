r"""Utility classes for forecasting."""

__all__ = [
    "BatchedCombinedArgs",
    "BatchedDenseArgs",
    "BatchedTripletArgs",
    "TripletArg",
    "DenseArg",
    "CombinedArg",
    "all_or_none",
    "is_prefix_mask",
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


@dataclass(frozen=True)
class DenseArg:
    r"""Dense representation of forecasting arguments.

    Assumptions:
        - context time stamps are finite and non-decreasing
        - query time stamps are finite and non-decreasing
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

    static_covariates: Tensor | None = None  # Float[(M)]

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
        assert (T.diff(dim=-1) >= 0.0).all()
        assert X.shape == (context_size, context_dim)
        assert X.isfinite().any(dim=-1).all()  # at least one value per step

        *_, query_size = Q.shape
        assert Q.shape == (query_size,)
        assert Q.isfinite().all()
        assert (Q.diff(dim=-1) >= 0.0).all()

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
        Y = self.query_values
        _, input_dim = X.shape

        # Flatten row-major, matching a DataFrame melt over time and channel.
        context_mask = X.isfinite().flatten()
        flat_indices = context_mask.nonzero(as_tuple=True)[0]
        time_indices = flat_indices.div(input_dim, rounding_mode="floor")
        context_channels = flat_indices.remainder(input_dim)

        if self.query_mask is None:
            query_dim = input_dim if Y is None else Y.shape[-1]
            query_mask = torch.ones(
                (Q.shape[0], query_dim),
                dtype=torch.bool,
                device=Q.device,
            )
        else:
            query_dim = self.query_mask.shape[-1]
            query_mask = self.query_mask

        query_flat = query_mask.flatten()
        flat_indices = query_flat.nonzero(as_tuple=True)[0]
        query_time_indices = flat_indices.div(query_dim, rounding_mode="floor")
        query_channels = flat_indices.remainder(query_dim)

        return TripletArg(
            context_times=T[time_indices],
            context_channels=context_channels,
            context_values=X.flatten()[context_mask],
            query_times=Q[query_time_indices],
            query_channels=query_channels,
            query_values=None if Y is None else Y[query_mask],
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
        - query time stamps are finite and non-decreasing
        - if query mask is not given, it is assumed to be a full true tensor
        - if query values are given, they are finite at entries selected by the query mask
        - there is at least one context value observed per time stamp
        - there is at least one query value observed per time stamp
    """

    context_times: Tensor  # Float[(..., N)], padded
    context_values: Tensor  # Float[(..., N, D)], padded, sparse

    query_times: Tensor  # Float[(..., K)], padded
    query_mask: Tensor | None = None  # Bool[(..., K, F)]  padded
    query_values: Tensor | None = None  # Float[(..., K, F)]  padded, sparse

    static_covariates: Tensor | None = None  # Float[(..., M)]  padded

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
        assert (Q_valid[..., :-1] | ~Q_valid[..., 1:]).all(dim=-1).all()
        assert (T_valid[..., :-1] | ~T_valid[..., 1:]).all(dim=-1).all()
        assert (X_valid[..., :-1] | ~X_valid[..., 1:]).all(dim=-1).all()
        # check that valid values are increasing
        ΔQ = Q.diff(dim=-1, prepend=torch.full_like(Q[..., [0]], -torch.inf))
        ΔT = T.diff(dim=-1, prepend=torch.full_like(T[..., [0]], -torch.inf))
        assert (~Q_valid | (ΔQ >= 0)).all(dim=-1).all()
        assert (~T_valid | (ΔT >= 0)).all(dim=-1).all()

        # check padding
        context_lengths = T.isnan().sum(dim=-1)
        assert torch.equal(X.isnan().all(dim=-1).sum(dim=-1), context_lengths)

        if (M := self.query_mask) is not None:
            *_, query_dim = M.shape
            assert M.dtype == torch.bool
            assert M.shape == (*batch_shape, query_size, query_dim)
            M_valid = M.isfinite().all(dim=-1)  # at least one value per step
            assert (M_valid[..., :-1] | ~M_valid[..., 1:]).all(dim=-1).all()

        if (V := self.query_values) is not None:
            *_, query_dim = V.shape
            assert V.shape == (*batch_shape, query_size, query_dim)
            V_valid = V.isfinite().any(dim=-1)  # at least one value per step
            assert (V_valid[..., :-1] | ~V_valid[..., 1:]).all(dim=-1).all()

            if (mask := self.query_mask) is not None:
                assert mask.shape == V.shape
            else:
                assert V[mask].isfinite().all()

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
        context_positions = context_mask.long().cumsum(dim=-1) - 1

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
        query_positions = query_mask.long().cumsum(dim=-1) - 1

        query_times = Q.new_full((Q_flat.shape[0], num_query), torch.nan)
        query_channels = torch.full(
            (Q_flat.shape[0], num_query),
            -1,
            dtype=torch.long,
            device=Q.device,
        )
        query_values = Q.new_full((Q_flat.shape[0], num_query), torch.nan)

        batch_indices, flat_indices = query_mask.nonzero(as_tuple=True)
        target_indices = query_positions[batch_indices, flat_indices]
        time_indices = flat_indices.div(query_dim, rounding_mode="floor")
        channel_indices = flat_indices.remainder(query_dim)

        query_times[batch_indices, target_indices] = Q_flat[batch_indices, time_indices]
        query_channels[batch_indices, target_indices] = channel_indices
        if Y_flat is not None:
            query_values[batch_indices, target_indices] = Y_flat[
                batch_indices, time_indices, channel_indices
            ]

        return BatchedTripletArgs(
            context_times=context_times.reshape(*batch_shape, num_context),
            context_channels=context_channels.reshape(*batch_shape, num_context),
            context_values=context_values.reshape(*batch_shape, num_context),
            query_times=query_times.reshape(*batch_shape, num_query),
            query_channels=query_channels.reshape(*batch_shape, num_query),
            query_values=query_values.reshape(*batch_shape, num_query),
            static_covariates=self.static_covariates,
        )

    def to_combined(self) -> BatchedCombinedArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class TripletArg:
    r"""Triplet representation of forecasting arguments."""

    context_times: Tensor  # Float[(Oᵢ)], finite
    context_channels: Tensor  # Long[(Oᵢ)]
    context_values: Tensor  # Float[(Oᵢ)], finite

    query_times: Tensor  # Float[(Qᵢ)], finite
    query_channels: Tensor | None = None  # Long[(Qᵢ)]
    query_values: Tensor | None = None  # Float[(Qᵢ)], finite

    static_covariates: Tensor | None = None  # Float[(M)]

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
        V = self.context_values

        *_, num_context = T.shape
        assert T.shape == (num_context,)
        assert C.shape == (num_context,)
        assert V.shape == (num_context,)
        assert T.isfinite().all()
        assert C.isfinite().all()
        assert V.isfinite().all()

        Q = self.query_times
        *_, num_query = Q.shape
        assert Q.shape == (num_query,)
        assert Q.isfinite().all()

        if self.query_channels is not None:
            assert self.query_channels.shape == (num_query,)
            assert self.query_channels.isfinite().all()

        if self.query_values is not None:
            assert self.query_values.shape == (num_query,)
            assert self.query_values.isfinite().all()

    def to_dense(self) -> DenseArg:
        if (self.context_channels < 0).any():
            raise ValueError("Expected non-negative context channel indices.")
        if self.context_channels.numel() == 0:
            raise ValueError(
                "Cannot infer dense context dimension from empty triplets."
            )

        context_dim = int(self.context_channels.max().item()) + 1
        context_times, context_inverse = torch.unique_consecutive(
            self.context_times,
            return_inverse=True,
        )
        context_values = self.context_values.new_full(
            (context_times.shape[0], context_dim),
            torch.nan,
        )
        # Triplets with the same consecutive time are written into one dense row.
        context_values[context_inverse, self.context_channels.long()] = (
            self.context_values
        )

        if self.query_channels is None:
            query_values = (
                None
                if self.query_values is None
                else self.query_values.unsqueeze(dim=-1)
            )
            return DenseArg(
                context_times=context_times,
                context_values=context_values,
                query_times=self.query_times,
                query_values=query_values,
                static_covariates=self.static_covariates,
            )

        if (self.query_channels < 0).any():
            raise ValueError("Expected non-negative query channel indices.")

        query_dim = (
            context_dim
            if self.query_channels.numel() == 0
            else int(self.query_channels.max().item()) + 1
        )
        query_times, query_inverse = torch.unique_consecutive(
            self.query_times,
            return_inverse=True,
        )
        query_mask = torch.zeros(
            (query_times.shape[0], query_dim),
            dtype=torch.bool,
            device=self.query_times.device,
        )
        query_mask[query_inverse, self.query_channels.long()] = True

        query_values = None
        if self.query_values is not None:
            query_values = self.query_values.new_full(
                (query_times.shape[0], query_dim),
                torch.nan,
            )
            query_values[query_inverse, self.query_channels.long()] = self.query_values

        return DenseArg(
            context_times=context_times,
            context_values=context_values,
            query_times=query_times,
            query_mask=None if bool(query_mask.all().item()) else query_mask,
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
    """

    context_times: Tensor  # Float[(..., O)], padded
    context_channels: Tensor  # Long[(..., O)], padded
    context_values: Tensor  # Float[(..., O)], padded

    query_times: Tensor  # Float[(..., Q)], padded
    query_channels: Tensor  # Long[(..., Q)], padded
    query_values: Tensor  # Float[(..., Q)], padded

    static_covariates: Tensor | None = None  # Float[(..., M)]

    def __post_init__(self) -> None:
        T = self.context_times
        C = self.context_channels
        V = self.context_values
        *batch_shape, num_context = T.shape
        assert T.shape == (*batch_shape, num_context)
        assert C.shape == (*batch_shape, num_context)
        assert V.shape == (*batch_shape, num_context)

        Q = self.query_times
        M = self.query_channels
        *_, num_query = self.query_times.shape
        assert Q.shape == (*batch_shape, num_query)
        assert M.shape == (*batch_shape, num_query)

    @classmethod
    def from_combined(cls, arg: BatchedCombinedArgs, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_dense(cls, arg: BatchedDenseArgs, /) -> BatchedTripletArgs:
        return arg.to_triplet()

    @classmethod
    def from_unbatched(cls, args: Sequence[TripletArg]) -> BatchedTripletArgs:
        raise NotImplementedError

    def unbatch(self) -> list[TripletArg]:
        raise NotImplementedError

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

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

    times: Tensor  # Float[(..., N + K)], padded
    values: Tensor  # Float[(..., N + K, E)], padded, sparse
    context_mask: Tensor  # Bool[(..., N + K, E)]
    query_mask: Tensor  # Bool[(..., N + K, E)]

    static_covariates: Tensor | None = None  # Float[(..., M)], padded, sparse

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
