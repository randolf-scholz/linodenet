r"""Public Interfaces."""

__all__ = [
    # Constants
    "DEFAULT_NEWTON_MAXITER",
    # Protocols
    "BimodalToGaussian",
    "BimodalToGaussianValueAndGrad",
    "GaussianToBimodal",
    "GaussianToBimodalValueAndGrad",
    "GaussianToMixture",
    "GaussianToMixtureValueAndGrad",
    "HardBend",
    "MixtureToGaussian",
    "MixtureToGaussianValueAndGrad",
    "NdtriExp",
    "SingularTriplet",
    "SpectralNorm",
    # Classes
    "KnownFunctions",
    "Kernels",
    "IncompleteKernels",
]

from dataclasses import dataclass
from typing import Final, Optional, Protocol, ReadOnly, TypedDict

import torch
from torch import Tensor

from signatures import signature

DEFAULT_NEWTON_MAXITER: Final[dict[torch.dtype, int]] = {
    torch.float16: 10,
    torch.bfloat16: 10,
    torch.float32: 10,
    torch.float64: 15,
}


class GaussianToBimodal(Protocol):
    r"""Protocol for Gaussian-to-bimodal transport implementations."""

    def __call__(
        self,
        y: Tensor,
        /,
        mu: Tensor = ...,
        sigma: Tensor = ...,
        *,
        maxiter: Optional[int] = ...,
    ) -> Tensor: ...


class GaussianToBimodalValueAndGrad(Protocol):
    r"""Protocol for Gaussian-to-bimodal transport with elementwise derivative."""

    def __call__(
        self,
        y: Tensor,
        /,
        mu: Tensor = ...,
        sigma: Tensor = ...,
        *,
        maxiter: Optional[int] = ...,
    ) -> tuple[Tensor, Tensor]: ...


class BimodalToGaussian(Protocol):
    r"""Protocol for bimodal-to-Gaussian transport implementations."""

    def __call__(
        self, x: Tensor, /, mu: Tensor = ..., sigma: Tensor = ...
    ) -> Tensor: ...


class BimodalToGaussianValueAndGrad(Protocol):
    r"""Protocol for bimodal-to-Gaussian transport with elementwise derivative."""

    def __call__(
        self, x: Tensor, /, mu: Tensor = ..., sigma: Tensor = ...
    ) -> tuple[Tensor, Tensor]: ...


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
        maxiter: Optional[int] = ...,
    ) -> Tensor: ...


class GaussianToMixtureValueAndGrad(Protocol):
    r"""Protocol for Gaussian-to-mixture transport with elementwise derivative."""

    def __call__(
        self,
        y: Tensor,
        /,
        weights: Tensor,
        mus: Tensor,
        sigmas: Tensor,
        *,
        maxiter: Optional[int] = ...,
    ) -> tuple[Tensor, Tensor]: ...


class MixtureToGaussian(Protocol):
    r"""Protocol for mixture-to-Gaussian transport implementations."""

    def __call__(
        self, x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
    ) -> Tensor: ...


class MixtureToGaussianValueAndGrad(Protocol):
    r"""Protocol for mixture-to-Gaussian transport with elementwise derivative."""

    def __call__(
        self, x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
    ) -> tuple[Tensor, Tensor]: ...


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
    singular_triplet:                    ReadOnly[ SingularTriplet | None ]
    spectral_norm:                       ReadOnly[ SpectralNorm | None ]
    ndtri_exp:                           ReadOnly[ NdtriExp | None ]
    hard_bend:                           ReadOnly[ HardBend | None ]
    bimodal_to_gaussian:                 ReadOnly[ BimodalToGaussian | None ]
    bimodal_to_gaussian_value_and_grad:  ReadOnly[ BimodalToGaussianValueAndGrad | None ]
    gaussian_to_bimodal:                 ReadOnly[ GaussianToBimodal | None ]
    gaussian_to_bimodal_value_and_grad:  ReadOnly[ GaussianToBimodalValueAndGrad | None ]
    gaussian_to_mixture:                 ReadOnly[ GaussianToMixture | None ]
    gaussian_to_mixture_value_and_grad:  ReadOnly[ GaussianToMixtureValueAndGrad | None ]
    mixture_to_gaussian:                 ReadOnly[ MixtureToGaussian | None ]
    mixture_to_gaussian_value_and_grad:  ReadOnly[ MixtureToGaussianValueAndGrad | None ]
    # fmt: on


@dataclass(frozen=True)
class Kernels:
    r"""The selected kernels exposed as attributes."""

    # fmt: off
    singular_triplet:                    SingularTriplet
    spectral_norm:                       SpectralNorm
    ndtri_exp:                           NdtriExp
    hard_bend:                           HardBend
    bimodal_to_gaussian:                 BimodalToGaussian
    bimodal_to_gaussian_value_and_grad:  BimodalToGaussianValueAndGrad
    gaussian_to_bimodal:                 GaussianToBimodal
    gaussian_to_bimodal_value_and_grad:  GaussianToBimodalValueAndGrad
    gaussian_to_mixture:                 GaussianToMixture
    gaussian_to_mixture_value_and_grad:  GaussianToMixtureValueAndGrad
    mixture_to_gaussian:                 MixtureToGaussian
    mixture_to_gaussian_value_and_grad:  MixtureToGaussianValueAndGrad
    # fmt: on


@dataclass(frozen=True)
class IncompleteKernels:
    r"""The selected kernels exposed as attributes."""

    # fmt: off
    singular_triplet:                    SingularTriplet | None
    spectral_norm:                       SpectralNorm | None
    ndtri_exp:                           NdtriExp | None
    hard_bend:                           HardBend | None
    bimodal_to_gaussian:                 BimodalToGaussian | None
    bimodal_to_gaussian_value_and_grad:  BimodalToGaussianValueAndGrad | None
    gaussian_to_bimodal:                 GaussianToBimodal | None
    gaussian_to_bimodal_value_and_grad:  GaussianToBimodalValueAndGrad | None
    gaussian_to_mixture:                 GaussianToMixture | None
    gaussian_to_mixture_value_and_grad:  GaussianToMixtureValueAndGrad | None
    mixture_to_gaussian:                 MixtureToGaussian | None
    mixture_to_gaussian_value_and_grad:  MixtureToGaussianValueAndGrad | None
    # fmt: on
