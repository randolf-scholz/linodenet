r"""Utility functions."""
import logging
from collections.abc import Mapping
from typing import Final, Type

from torch import nn

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "ACTIVATIONS",
    "deep_dict_update",
    "deep_keyval_update",
]

ACTIVATIONS: Final[dict[str, Type[nn.Module]]] = {
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "LogSoftmax": nn.LogSoftmax,
    "MultiheadAttention": nn.MultiheadAttention,
    "PReLU": nn.PReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "Softsign": nn.Softsign,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink,
    "Threshold": nn.Threshold,
}


def deep_dict_update(d: dict, new: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new: Mapping
    """
    for key, value in new.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            d[key] = new[key]
    return d


def deep_keyval_update(d: dict, **new_kv: dict) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kv: Mapping
    """
    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_keyval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d
