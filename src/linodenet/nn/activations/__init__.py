r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    # Constants
    "ALL_ACTIVATIONS",
    "ACTIVATION_FUNCTIONS",
    "ACTIVATION_CLASSES",
    "TORCH_ACTIVATION_FUNCTIONS",
    "TORCH_INPLACE_ACTIVATIONS",
    "TORCH_SPECIAL_ACTIVATIONS",
    "TORCH_ACTIVATION_CLASSES",
    # ABCs & Protocols
    "Activation",
    "ActivationBase",
    "GenericActivation",
    # Classes
    "HardBend",
    "GEGLU",
    "ReGLU",
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
    # utils
    "get_activation",
]

from torch import nn

from linodenet.nn.activations import base
from linodenet.nn.activations.base import (
    Activation,
    ActivationBase,
    GenericActivation,
    get_activation,
)
from linodenet.nn.activations.geglu import GEGLU, geglu
from linodenet.nn.activations.hard_bend import HardBend, hard_bend
from linodenet.nn.activations.reglu import ReGLU, reglu

TORCH_ACTIVATION_FUNCTIONS: dict[str, Activation] = {
    "relu": nn.functional.relu,
    # Applies the rectified linear unit function element-wise.
    "hardtanh": nn.functional.hardtanh,
    # Applies the HardTanh function element-wise.
    "hardswish": nn.functional.hardswish,
    # Applies the hardswish function, element-wise, as described in the paper:
    "relu6": nn.functional.relu6,
    # Applies the element-wise function `ReLU6(x)=\min(\max(0,x),6)`.
    "elu": nn.functional.elu,
    # Applies element-wise, `ELU(x)=\max(0,x)+\min(0,α⋅(\exp(x)−1))`.
    "selu": nn.functional.selu,
    # Applies element-wise, `SELU(x)=β⋅(\max(0,x)+\min(0,α⋅(eˣ−1)))` with `α≈1.677` and `β≈1.05`.
    "celu": nn.functional.celu,
    # Applies element-wise, `CELU(x)= \max(0,x)+\min(0,α⋅(\exp(x/α)−1)`.
    "leaky_relu": nn.functional.leaky_relu,
    # Applies element-wise, `LeakyReLU(x)=\max(0,x)+negative_slope⋅\min(0,x)`.
    "rrelu": nn.functional.rrelu,
    # Randomized leaky ReLU.
    "glu": nn.functional.glu,
    # The gated linear unit.
    "gelu": nn.functional.gelu,
    # Applies element-wise the function `GELU(x)=x⋅Φ(x)`.
    "log_sigmoid": nn.functional.logsigmoid,  # FIXME: name is different for some reason.
    # Applies element-wise `LogSigmoid(x_i)=\log(1/(1+\exp(−x_i)))`.
    "hardshrink": nn.functional.hardshrink,
    # Applies the hard shrinkage function element-wise.
    "tanhshrink": nn.functional.tanhshrink,
    # Applies element-wise, `Tanhshrink(x)=x−\tanh(x)`.
    "softsign": nn.functional.softsign,
    # Applies element-wise, the function `SoftSign(x)=x/(1+∣x∣)`.
    "softplus": nn.functional.softplus,
    # Applies element-wise, the function `Softplus(x)=1/β⋅\log(1+\exp(β⋅x))`.
    "softmin": nn.functional.softmin,
    # Applies a softmin function.
    "softmax": nn.functional.softmax,
    # Applies a softmax function.
    "softshrink": nn.functional.softshrink,
    # Applies the soft shrinkage function elementwise
    "gumbel_softmax": nn.functional.gumbel_softmax,
    # Samples from the Gumbel-Softmax distribution and optionally discretizes.
    "log_softmax": nn.functional.log_softmax,
    # Applies a softmax followed by a logarithm.
    "tanh": nn.functional.tanh,
    # Applies element-wise, `\tanh(x)=(\exp(x)−\exp(−x))/(\exp(x)+\exp(−x))`.
    "sigmoid": nn.functional.sigmoid,
    # Applies the element-wise function `Sigmoid(x)=1/(1+\exp(−x))`.
    "hardsigmoid": nn.functional.hardsigmoid,
    # Applies the hardsigmoid function element-wise.
    "silu": nn.functional.silu,
    # Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    "mish": nn.functional.mish,
    # Applies the Mish function, element-wise.
    "normalize": nn.functional.normalize,
    # Performs Lp normalization of inputs over specified dimension.
}
r"""Dictionary containing all available functional activations in torch."""


TORCH_INPLACE_ACTIVATIONS: dict[str, Activation] = {
    "relu_": nn.functional.relu_,
    # In-place version of relu().
    "hardtanh_": nn.functional.hardtanh_,
    # In-place version of hardtanh().
    "elu_": nn.functional.elu_,
    # In-place version of elu().
    "leaky_relu_": nn.functional.leaky_relu_,
    # In-place version of leaky_relu().
    "rrelu_": nn.functional.rrelu_,
    # In-place version of rrelu().
}


TORCH_SPECIAL_ACTIVATIONS: dict[str, GenericActivation] = {
    "threshold": nn.functional.threshold,
    # Thresholds each element of the input Tensor.
    "prelu": nn.functional.prelu,
    # `PReLU(x)=\max(0,x)+ω⋅\min(0,x)` where ω is a learnable parameter.
    "batch_norm": nn.functional.batch_norm,
    # Applies Batch Normalization for each channel across a batch of data.
    "group_norm": nn.functional.group_norm,
    # Applies Group Normalization for last certain number of dimensions.
    "layer_norm": nn.functional.layer_norm,
    # Applies Layer Normalization for last certain number of dimensions.
    "local_response_norm": nn.functional.local_response_norm,
    # Applies local response normalization over an input signal composed of several input planes.
}
r"""Special activations that do not represent usual activation functions."""


TORCH_ACTIVATION_CLASSES: dict[str, type[Activation]] = {
    "AdaptiveLogSoftmaxWithLoss" : nn.AdaptiveLogSoftmaxWithLoss,
    "ELU"                        : nn.ELU,
    "Hardshrink"                 : nn.Hardshrink,
    "Hardsigmoid"                : nn.Hardsigmoid,
    "Hardtanh"                   : nn.Hardtanh,
    "Hardswish"                  : nn.Hardswish,
    "Identity"                   : nn.Identity,
    "LeakyReLU"                  : nn.LeakyReLU,
    "LogSigmoid"                 : nn.LogSigmoid,
    "LogSoftmax"                 : nn.LogSoftmax,
    "MultiheadAttention"         : nn.MultiheadAttention,
    "PReLU"                      : nn.PReLU,
    "ReLU"                       : nn.ReLU,
    "ReLU6"                      : nn.ReLU6,
    "RReLU"                      : nn.RReLU,
    "SELU"                       : nn.SELU,
    "CELU"                       : nn.CELU,
    "GELU"                       : nn.GELU,
    "Sigmoid"                    : nn.Sigmoid,
    "SiLU"                       : nn.SiLU,
    "Softmax"                    : nn.Softmax,
    "Softmax2d"                  : nn.Softmax2d,
    "Softplus"                   : nn.Softplus,
    "Softshrink"                 : nn.Softshrink,
    "Softsign"                   : nn.Softsign,
    "Tanh"                       : nn.Tanh,
    "Tanhshrink"                 : nn.Tanhshrink,
    "Threshold"                  : nn.Threshold,
}  # fmt: skip
r"""Dictionary containing all available activations in torch."""


ACTIVATION_FUNCTIONS: dict[str, Activation] = {
    **TORCH_ACTIVATION_FUNCTIONS,
    "reglu": reglu,
    "geglu": geglu,
    "hard_bend": hard_bend,
}
r"""Dictionary containing all available activation functions."""


ACTIVATION_CLASSES: dict[str, type[Activation]] = {
    **TORCH_ACTIVATION_CLASSES,
    "HardBend": HardBend,
    "GeGLU": GEGLU,
    "ReGLU": ReGLU,
}
r"""Dictionary containing all available activation classes."""


ALL_ACTIVATIONS: dict[str, Activation | type[Activation]] = {
    **ACTIVATION_FUNCTIONS,
    **ACTIVATION_CLASSES,
    **TORCH_ACTIVATION_FUNCTIONS,
    **TORCH_ACTIVATION_CLASSES,
}
r"""Dictionary containing all available activations."""


# cleanup namespace
del nn
