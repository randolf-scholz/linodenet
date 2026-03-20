r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    # Constants
    "ACTIVATIONS",
    "ACTIVATION_FNS",
    "ACTIVATION_FNS_WITH_ARGS",
    # ABCs & Protocols
    "Activation",
    "ActivationBase",
    "GenericActivation",
    # Classes
    "CReLU",
    "GEGLU",
    "ReGLU",
    # Functions
    "crelu",
    "geglu",
    "reglu",
    "gaussian_to_bimodal",
    "bimodal_to_gaussian",
    "hard_bend",
    # utils
    "get_activation",
]

from torch import nn as _nn

from linodenet.nn.activations import base
from linodenet.nn.activations.base import (
    Activation,
    ActivationBase,
    GenericActivation,
    get_activation,
)
from linodenet.nn.activations.crelu import CReLU, crelu
from linodenet.nn.activations.geglu import GEGLU, geglu
from linodenet.nn.activations.reglu import ReGLU, reglu
from linodenet_special import bimodal_to_gaussian, gaussian_to_bimodal, hard_bend

ACTIVATIONS: dict[str, type[_nn.Module]] = {
    # torch imports
    "CELU"        : _nn.CELU,
    "ELU"         : _nn.ELU,
    "GELU"        : _nn.GELU,
    "GLU"         : _nn.GLU,
    "Hardshrink"  : _nn.Hardshrink,
    "Hardsigmoid" : _nn.Hardsigmoid,
    "Hardswish"   : _nn.Hardswish,
    "Hardtanh"    : _nn.Hardtanh,
    "Identity"    : _nn.Identity,
    "LeakyReLU"   : _nn.LeakyReLU,
    "LogSigmoid"  : _nn.LogSigmoid,
    "Mish"        : _nn.Mish,
    "PReLU"       : _nn.PReLU,
    "RReLU"       : _nn.RReLU,
    "ReLU"        : _nn.ReLU,
    "ReLU6"       : _nn.ReLU6,
    "SELU"        : _nn.SELU,
    "SiLU"        : _nn.SiLU,
    "Sigmoid"     : _nn.Sigmoid,
    "Softplus"    : _nn.Softplus,
    "Softshrink"  : _nn.Softshrink,
    "Softsign"    : _nn.Softsign,
    "Tanh"        : _nn.Tanh,
    "Tanhshrink"  : _nn.Tanhshrink,
}  # fmt: skip
r"""Dictionary containing all available activation classes."""


ACTIVATION_FNS: dict[str, Activation] = {
    "hard_bend": hard_bend,
    # torch imports
    "celu"        : _nn.functional.celu,
    "elu"         : _nn.functional.elu,
    "gelu"        : _nn.functional.gelu,
    "hardshrink"  : _nn.functional.hardshrink,
    "hardsigmoid" : _nn.functional.hardsigmoid,
    "hardswish"   : _nn.functional.hardswish,
    "hardtanh"    : _nn.functional.hardtanh,
    "leaky_relu"  : _nn.functional.leaky_relu,
    "log_sigmoid" : _nn.functional.logsigmoid,
    "mish"        : _nn.functional.mish,
    "relu"        : _nn.functional.relu,
    "relu6"       : _nn.functional.relu6,
    "rrelu"       : _nn.functional.rrelu,
    "selu"        : _nn.functional.selu,
    "sigmoid"     : _nn.functional.sigmoid,
    "silu"        : _nn.functional.silu,
    "softplus"    : _nn.functional.softplus,
    "softshrink"  : _nn.functional.softshrink,
    "softsign"    : _nn.functional.softsign,
    "tanh"        : _nn.functional.tanh,
    "tanhshrink"  : _nn.functional.tanhshrink,
}  # fmt: skip
r"""Dictionary containing all available activation functions."""


ACTIVATION_FNS_WITH_ARGS: dict[str, GenericActivation] = {
    "reglu": reglu,
    "geglu": geglu,
    "crelu": crelu,
}  # fmt: skip
r"""Activations that do not match the usual signature of activations."""
