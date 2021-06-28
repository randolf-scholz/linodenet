from typing import Callable, Union

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
import torch
from torch import nn, Tensor


def _torch_is_float_dtype(x: Tensor) -> bool:
    return x.dtype in (torch.half, torch.float, torch.double, torch.bfloat16,
                       torch.complex32, torch.complex64, torch.complex128)


def _torch_scaled_norm(x: Tensor,  p: float = 2, dim: Union[int, tuple[int, ...]] = None,
                       keepdim: bool = False) -> Tensor:
    dim = () if dim is None else dim

    if not _torch_is_float_dtype(x):
        x = x.to(dtype=torch.float)
    x = torch.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return torch.exp(torch.mean(torch.log(x), dim=dim, keepdim=keepdim))
    if p == 1:
        return torch.mean(x, dim=dim, keepdim=keepdim)
    if p == 2:
        return torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=keepdim))
    if p == float('inf'):
        return torch.amax(x, dim=dim, keepdim=keepdim)
    # other p
    return torch.mean(x**p, dim=dim, keepdim=keepdim) ** (1 / p)


def _numpy_scaled_norm(x: ndarray, p: float = 2, axis: Union[int, tuple[int, ...]] = None,
                       keepdims: bool = False) -> ndarray:
    x = np.abs(x)

    if p == 0:
        # https://math.stackexchange.com/q/282271/99220
        return np.exp(np.mean(np.log(x), axis=axis, keepdims=keepdims))
    if p == 1:
        return np.mean(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))
    if p == float('inf'):
        return np.max(x, axis=axis, keepdims=keepdims)
    # other p
    return np.mean(x**p, axis=axis, keepdims=keepdims) ** (1 / p)


def scaled_norm(x: Union[ArrayLike, Tensor], p=2, axis=None, keepdims=False) -> Union[ArrayLike, Tensor]:
    r"""Scaled $\ell^p$-norm, works with both :class:`torch.Tensor` and :class:`numpy.ndarray`

    .. math::
        \|x\|_p = \big(\tfrac{1}{n}\sum_{i=1}^n x_i^p \big)^{1/p}

    Parameters
    ----------
    x: ArrayLike
    p: int, default=2
    axis: tuple[int], default=None
    keepdims: bool, default=False

    Returns
    -------
    ArrayLike
    """
    if isinstance(x, Tensor):
        return _torch_scaled_norm(x, p=p, dim=axis, keepdim=keepdims)

    x = np.asanyarray(x)
    return _numpy_scaled_norm(x, p=p, axis=axis, keepdims=keepdims)