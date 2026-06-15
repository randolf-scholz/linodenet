r"""Utility classes for forecasting."""

__all__ = [
    "BatchedCombinedArgs",
    "BatchedDenseArgs",
    "BatchedTripletArgs",
    "TripletArg",
    "DenseArg",
    "all_or_none",
]


from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


@dataclass(frozen=True)
class DenseArg:
    r"""Dense representation of forecasting arguments."""

    context_times: Tensor  # Float[(N)], finite
    context_values: Tensor  # Float[(N, D)], sparse

    query_times: Tensor  # Float[(K)], finite
    query_mask: Tensor | None = None  # Bool[(K, F)]
    query_values: Tensor | None = None  # Float[(K, F)], sparse

    static_covariates: Tensor | None = None  # Float[(M)]

    def __post_init__(self) -> None:
        T = self.context_times
        X = self.context_values

        *_, context_size, context_dim = X.shape
        assert T.shape == (context_size,)
        assert X.shape == (context_size, context_dim)
        assert T.isfinite().all()
        assert X.isfinite().any(dim=-1).all()  # at least one value observed per time

        Q = self.query_times
        *_, query_size = Q.shape
        assert Q.shape == (query_size,)
        assert Q.isfinite().all()

        if self.query_mask is not None:
            *_, query_dim = self.query_mask.shape
            assert self.query_mask.dtype == torch.bool
            assert self.query_mask.shape == (query_size, query_dim)

        if self.query_values is not None:
            *_, query_dim = self.query_values.shape
            assert self.query_values.shape == (query_size, query_dim)

            if self.query_mask is None:
                assert self.query_values.isfinite().all()
            else:
                assert self.query_values.shape == self.query_mask.shape
                assert self.query_values[self.query_mask].isfinite().all()


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


@dataclass(frozen=True)
class BatchedDenseArgs:
    r"""Batched forecasting arguments.

    Shapes:
        K: max(Kᵢ) query size
        N: max(Nᵢ) context size
        D: input dimensionality
        F: output dimensionality
        M: static covariate dimensionality
    """

    context_times: Tensor  # Float[(..., N)], padded
    context_values: Tensor  # Float[(..., N, D)], padded, sparse

    query_times: Tensor  # Float[(..., K)], padded
    query_mask: Tensor | None = None  # Bool[(..., K, F)]  padded
    query_values: Tensor | None = None  # Float[(..., K, F)]  padded, sparse

    static_covariates: Tensor | None = None  # Float[(..., M)]  padded

    def __post_init__(self) -> None:
        T = self.context_times
        Q = self.query_times
        X = self.context_values

        # check shapes
        *batch_shape, context_size, _ = X.shape
        *_, query_size = Q.shape
        assert T.shape == (*batch_shape, context_size)
        assert Q.shape == (*batch_shape, query_size)

        if self.query_mask is not None:
            *_, query_dim = self.query_mask.shape
            assert self.query_mask.dtype == torch.bool
            assert self.query_mask.shape == (*batch_shape, query_size, query_dim)

        if self.query_values is not None:
            *_, query_dim = self.query_values.shape
            assert self.query_values.shape == (*batch_shape, query_size, query_dim)
            if self.query_mask is not None:
                assert self.query_mask.shape == self.query_values.shape

        if self.static_covariates is not None:
            *_, static_dim = self.static_covariates.shape
            assert self.static_covariates.shape == (*batch_shape, static_dim)

        # check that non-valid values are at the tail
        T_valid = T.isfinite()
        Q_valid = Q.isfinite()
        X_valid = X.isfinite().any(dim=-1)  # at least one value observed
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

    @classmethod
    def from_unbatched(cls, args: Sequence[DenseArg]) -> BatchedDenseArgs:
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

    def unbatch(self) -> list[TripletArg]:
        raise NotImplementedError

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_combined(self) -> BatchedCombinedArgs:
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

    times: Tensor  # Float[(..., N + K)], finite
    values: Tensor  # Float[(..., N + K, E)], finite
    context_mask: Tensor  # Bool[(..., N + K, E)]
    query_mask: Tensor  # Bool[(..., N + K, E)]

    static_covariates: Tensor | None = None  # Float[(..., M)]

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> BatchedTripletArgs:
        raise NotImplementedError
