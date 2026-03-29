r"""Reimplementation of `scipy.special.ndtri_exp` for PyTorch.

SCIPY LICENSE
=============
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = [
    # functions
    "ndtri_exp",
    "ndtri_exp_naive",
]

import math
from typing import Final

import torch
from torch import Tensor
from torch.special import ndtri

# constants
_UPPER_CUTOFF: Final[float] = -0.14541345786885906  # log(1-e⁻²)
_LOWER_CUTOFF: Final[float] = -2.0
_SQRT_2: Final[float] = math.sqrt(2.0)
_P1: Final[list[float]] = [
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
]
_Q1: Final[list[float]] = [
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
]
_P2: Final[list[float]] = [
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
]
_Q2: Final[list[float]] = [
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
]

# create a small cache for the coefficients so they do not need to
# be re-initialized constantly per device.
_COEFF_CACHE: Final[
    dict[
        tuple[torch.device, torch.dtype],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]
] = {}


def _get_coeffs(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key = (device, dtype)
    coeffs = _COEFF_CACHE.get(key)
    if coeffs is None:
        coeffs = (
            torch.tensor(_P1, device=device, dtype=dtype),
            torch.tensor(_Q1, device=device, dtype=dtype),
            torch.tensor(_P2, device=device, dtype=dtype),
            torch.tensor(_Q2, device=device, dtype=dtype),
        )
        _COEFF_CACHE[key] = coeffs
    return coeffs


def _polyeval8(x: Tensor, coeffs: Tensor, /) -> Tensor:
    # Horner, coeffs in descending order
    assert coeffs.shape == (9,)
    y = torch.zeros_like(x)
    # y = y * x + c, unrolled for better performance
    y = torch.addcmul(coeffs[0], x, y)
    y = torch.addcmul(coeffs[1], x, y)
    y = torch.addcmul(coeffs[2], x, y)
    y = torch.addcmul(coeffs[3], x, y)
    y = torch.addcmul(coeffs[4], x, y)
    y = torch.addcmul(coeffs[5], x, y)
    y = torch.addcmul(coeffs[6], x, y)
    y = torch.addcmul(coeffs[7], x, y)
    y = torch.addcmul(coeffs[8], x, y)
    return y


def _poly1eval8(x: Tensor, coeffs: Tensor, /) -> Tensor:
    # Horner with leading 1: evaluates x^n + c0*x^(n-1)+...+c_{n-1}
    assert coeffs.shape == (8,)
    y = torch.ones_like(x)
    # y = y * x + c, unrolled for better performance
    y = torch.addcmul(coeffs[0], x, y)
    y = torch.addcmul(coeffs[1], x, y)
    y = torch.addcmul(coeffs[2], x, y)
    y = torch.addcmul(coeffs[3], x, y)
    y = torch.addcmul(coeffs[4], x, y)
    y = torch.addcmul(coeffs[5], x, y)
    y = torch.addcmul(coeffs[6], x, y)
    y = torch.addcmul(coeffs[7], x, y)
    return y


def _ndtri_exp_small(log_p: Tensor, /) -> Tensor:
    r"""Rational approximation of Φ⁻¹√(-2 log p) when log_p < -2."""
    # cast the coefficients to the same dtype and device as log_p
    p1, q1, p2, q2 = _get_coeffs(device=log_p.device, dtype=log_p.dtype)
    x = torch.sqrt(-2 * log_p)
    z = x.reciprocal()  # 1/x
    x0 = x - z * x.log()  # x - log(x)/x
    x1 = torch.where(
        x < 8.0,
        z * _polyeval8(z, p1) / _poly1eval8(z, q1),
        z * _polyeval8(z, p2) / _poly1eval8(z, q2),
    )
    return x1 - x0


def ndtri_exp_naive(log_p: Tensor) -> Tensor:
    r"""Computes the inverse of `log_ndtr` using the naive implementation."""
    return ndtri(log_p.exp())


def ndtri_exp(log_p: Tensor) -> Tensor:
    r"""Inverse of `log_ndtr`, i.e. the log-quantile function of the standard normal distribution.

    domain: (-∞, 0]   (image of [0,1] under log)
    codomain: [-∞, +∞]

    torch currently does not implement the inverse of `log_ndtr`,
    this is simply a placeholder using the naive implementation.

    References:
        - scipy.special.ndtri_exp
    """
    invalid_mask = log_p.isnan() | (log_p > 0)
    neginf_mask = log_p.isneginf()
    small_mask = (log_p < _LOWER_CUTOFF) & ~(invalid_mask | neginf_mask)
    medium_mask = (log_p >= _LOWER_CUTOFF) & (log_p <= _UPPER_CUTOFF)
    large_mask = (log_p > _UPPER_CUTOFF) & ~invalid_mask

    # mask the unused part of the test with constant dummy value.
    # this prevents propagation of spurious NANs.
    # XREF: https://github.com/pytorch/pytorch/issues/89543
    dummy = log_p.new_full((), -1.0)
    small_input = torch.where(small_mask, log_p, dummy)
    medium_input = torch.where(medium_mask, log_p, dummy)
    large_input = torch.where(large_mask, log_p, dummy)

    neginf = log_p.new_full((), -math.inf)
    invalid = log_p.new_full((), math.nan)
    small = _ndtri_exp_small(small_input)
    medium = ndtri(medium_input.exp())
    large = -ndtri(-large_input.expm1())

    return torch.where(
        invalid_mask,
        invalid,
        torch.where(
            neginf_mask,
            neginf,
            torch.where(
                small_mask,
                small,
                torch.where(medium_mask, medium, large),
            ),
        ),
    )
