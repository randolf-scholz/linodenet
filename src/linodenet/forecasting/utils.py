r"""Utility classes for forecasting."""

__all__ = [
    "BatchedCombinedArgs",
    "BatchedDenseArgs",
    "BatchedTripletArgs",
    "UnbatchedCombinedArgs",
    "UnbatchedDenseArgs",
    "UnbatchedTripletArgs",
]


from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class UnbatchedDenseArgs:
    r"""Unbatched forecasting arguments.

    Shapes:
        Nᵢ: context size
        Kᵢ: query size
        D: input dimensionality
        F: output dimensionality
        M: static covariate dimensionality
    """

    context_times: Sequence[Tensor]  # Float[(Nᵢ)], finite
    context_values: Sequence[Tensor]  # Float[(Nᵢ, D)], sparse

    query_times: Sequence[Tensor]  # Float[(Kᵢ)], finite
    query_mask: Sequence[Tensor] | None = None  # Bool[(Kᵢ, F)]

    static_covariates: Sequence[Tensor] | None = None  # Float[(M)]

    def batch(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> UnbatchedTripletArgs:
        raise NotImplementedError

    def to_combined(self) -> UnbatchedCombinedArgs:
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
    """

    context_times: Tensor  # Float[(..., N)], padded
    context_values: Tensor  # Float[(..., N, D)], sparse

    query_times: Tensor  # Float[(..., K)], padded
    query_mask: Tensor | None = None  # Bool[(..., K, F)]

    static_covariates: Tensor | None = None  # Float[(..., M)]

    def unbatch(self) -> UnbatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> BatchedTripletArgs:
        raise NotImplementedError

    def to_combined(self) -> BatchedCombinedArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class UnbatchedTripletArgs:
    r"""Batched forecasting arguments with covariates.

    Shapes:
        Qᵢ: number of query values
        Oᵢ: number of observed values
        M: static covariate dimensionality
    """

    context_times: Sequence[Tensor]  # Float[(Oᵢ)], finite
    context_channels: Sequence[Tensor]  # Long[(Oᵢ)]
    context_values: Sequence[Tensor]  # Float[(Oᵢ)], finite

    query_times: Sequence[Tensor]  # Float[(Qᵢ)], finite
    query_channels: Sequence[Tensor] | None = None  # Long[(Qᵢ)]
    query_values: Sequence[Tensor] | None = None  # Float[(Qᵢ)], finite

    static_covariates: Sequence[Tensor] | None = None  # Float[(M)]

    def batch(self) -> BatchedTripletArgs:
        raise NotImplementedError

    def to_dense(self) -> UnbatchedDenseArgs:
        raise NotImplementedError

    def to_combined(self) -> UnbatchedCombinedArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class BatchedTripletArgs:
    r"""Triplet representation of forecasting arguments.

    Shapes:
        Q = max(Qᵢ): number of query values
        O = max(Oᵢ): number of observed values
        M: static covariate dimensionality
    """

    context_times: Tensor  # Float[(..., O)], padded
    context_channels: Tensor  # Long[(..., O)], padded
    context_values: Tensor  # Float[(..., O)], padded

    query_times: Tensor  # Float[(..., Q)], padded
    query_channels: Tensor  # Long[(..., Q)], padded
    query_values: Tensor  # Float[(..., Q)], padded

    static_covariates: Tensor | None = None  # Float[(..., M)]

    def unbatch(self) -> UnbatchedTripletArgs:
        raise NotImplementedError

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_combined(self) -> BatchedCombinedArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class UnbatchedCombinedArgs:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        Nᵢ: context size
        Kᵢ: query size
        E: combined data dimensionality
        M: static covariate dimensionality
    """

    times: Sequence[Tensor]  # Float[(Nᵢ + Kᵢ)], finite
    values: Sequence[Tensor]  # Float[(Nᵢ + Kᵢ, E)], finite
    context_mask: Sequence[Tensor]  # Bool[(Nᵢ + Kᵢ, E)]
    query_mask: Sequence[Tensor]  # Bool[(Nᵢ + Kᵢ, E)]

    static_covariates: Sequence[Tensor] | None = None  # Float[(M)]

    def batch(self) -> BatchedCombinedArgs:
        raise NotImplementedError

    def to_dense(self) -> UnbatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> UnbatchedTripletArgs:
        raise NotImplementedError


@dataclass(frozen=True)
class BatchedCombinedArgs:
    r"""Representation with concatenated context and query tensors.

    Shapes:
        N = max(Nᵢ): context size
        K = max(Kᵢ): query size
        E: combined data dimensionality
        M: static covariate dimensionality
    """

    times: Tensor  # Float[(..., N + K)], finite
    values: Tensor  # Float[(..., N + K, E)], finite
    context_mask: Tensor  # Bool[(..., N + K, E)]
    query_mask: Tensor  # Bool[(..., N + K, E)]

    static_covariates: Tensor | None = None  # Float[(..., M)]

    def unbatch(self) -> UnbatchedCombinedArgs:
        raise NotImplementedError

    def to_dense(self) -> BatchedDenseArgs:
        raise NotImplementedError

    def to_triplet(self) -> BatchedTripletArgs:
        raise NotImplementedError
