r"""Public Interfaces."""

__all__ = [
    # Protocols
    "BimodalToGaussian",
    "GaussianToBimodal",
    "GaussianToMixture",
    "HardBend",
    "MixtureToGaussian",
    "NdtriExp",
    "SingularTriplet",
    "SpectralNorm",
    # Classes
    "KnownFunctions",
]


from typing import Optional, Protocol, ReadOnly, TypedDict

from torch import Tensor

from signatures import signature


class GaussianToBimodal(Protocol):
    r"""Protocol for Gaussian-to-bimodal transport implementations."""

    def __call__(
        self,
        y: Tensor,
        /,
        mu: Tensor = ...,
        sigma: Tensor = ...,
        *,
        maxiter: int = ...,
    ) -> Tensor: ...


class BimodalToGaussian(Protocol):
    r"""Protocol for bimodal-to-Gaussian transport implementations."""

    def __call__(
        self, x: Tensor, /, mu: Tensor = ..., sigma: Tensor = ...
    ) -> Tensor: ...


class GaussianToMixture(Protocol):
    r"""Protocol for Gaussian-to-mixture transport implementations."""

    def __call__(
        self,
        y: Tensor,
        /,
        weights: Tensor,
        mus: Tensor,
        sigmas: Tensor,
        *,
        maxiter: int = ...,
    ) -> Tensor: ...


class MixtureToGaussian(Protocol):
    r"""Protocol for mixture-to-Gaussian transport implementations."""

    def __call__(
        self, x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
    ) -> Tensor: ...


class HardBend(Protocol):
    r"""Protocol for hard bend activation function."""

    def __call__(
        self,
        x: Tensor,
        /,
        a: Tensor | float = ...,
        c: Tensor | float = ...,
        m: Tensor | float = ...,
    ) -> Tensor: ...


class NdtriExp(Protocol):
    r"""Protocol of the ndtri_exp class."""

    def __call__(self, log_p: Tensor, /) -> Tensor: ...


class SingularTriplet(Protocol):
    r"""Protocol for singular triplet implementations."""

    @signature("(m, n) -> [(), (m), (n)]")
    def __call__(
        self,
        A: Tensor,
        /,
        *,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: int = ...,
        atol: float = ...,
        rtol: float = ...,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Computes the singular triplet.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: O(M+N))
            atol: The absolute tolerance. (Default: 1e-6)
            rtol: The relative tolerance. (Default: 1e-6)

        Returns:
            sigma: The singular value (scaler).
            u: The left singular vector (shape: M).
            v: The right singular vector (shape: N).
        """
        ...


class SpectralNorm(Protocol):
    r"""Protocol for spectral norm implementations."""

    @signature("(m, n) -> ()")
    def __call__(
        self,
        A: Tensor,
        /,
        *,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: int = ...,
        atol: float = ...,
        rtol: float = ...,
    ) -> Tensor:
        r"""Computes the spectral norm.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: O(M+N))
            atol: The absolute tolerance. (Default: 1e-6)
            rtol: The relative tolerance. (Default: 1e-6)

        Returns:
            sigma: The singular value (scaler).
        """
        ...


class KnownFunctions(TypedDict):
    r"""The known functions in the custom library."""

    # fmt: off
    singular_triplet:    ReadOnly[ SingularTriplet | None ]
    spectral_norm:       ReadOnly[ SpectralNorm | None ]
    ndtri_exp:           ReadOnly[ NdtriExp | None ]
    hard_bend:           ReadOnly[ HardBend | None ]
    bimodal_to_gaussian: ReadOnly[ BimodalToGaussian | None ]
    gaussian_to_bimodal: ReadOnly[ GaussianToBimodal | None ]
    gaussian_to_mixture: ReadOnly[ GaussianToMixture | None ]
    mixture_to_gaussian: ReadOnly[ MixtureToGaussian | None ]
    # fmt: on
