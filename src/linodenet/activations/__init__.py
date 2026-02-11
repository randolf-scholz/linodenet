r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    "functional",
    "modules",
    # Constants
    "ACTIVATIONS",
    "FUNCTIONAL_ACTIVATIONS",
    "MODULAR_ACTIVATIONS",
    "TORCH_ACTIVATIONS",
    "TORCH_FUNCTIONAL_ACTIVATIONS",
    "TORCH_MODULAR_ACTIVATIONS",
    # ABCs & Protocols
    "Activation",
    "ActivationBase",
    # Classes
    "HardBend",
    "GeGLU",
    "ReGLU",
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
    # utils
    "get_activation",
]

from linodenet.activations import base, functional, modules
from linodenet.activations._torch_imports import (
    TORCH_ACTIVATIONS,
    TORCH_FUNCTIONAL_ACTIVATIONS,
    TORCH_MODULAR_ACTIVATIONS,
)
from linodenet.activations.base import Activation, ActivationBase
from linodenet.activations.functional import geglu, hard_bend, reglu
from linodenet.activations.modules import GeGLU, HardBend, ReGLU

FUNCTIONAL_ACTIVATIONS: dict[str, Activation] = {
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    "reglu": reglu,
    "geglu": geglu,
    "hard_bend": hard_bend,
}
r"""Dictionary containing all available functional activations."""

MODULAR_ACTIVATIONS: dict[str, type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    "HardBend": HardBend,
    "GeGLU": GeGLU,
    "ReGLU": ReGLU,
}
r"""Dictionary containing all available activations."""

ACTIVATIONS: dict[str, Activation | type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    **MODULAR_ACTIVATIONS,
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **FUNCTIONAL_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""


def get_activation(kind: object = None, /, **cfg: object) -> Activation:
    r"""Get an activation function by name."""
    match kind:
        # if an instance, return as-is
        case Activation() as instance:
            if cfg:
                raise ValueError(f"Cannot pass arguments to an instance: {cfg!r}")
            return instance
        # if a name, look up in the dictionary
        case str(name):
            try:
                obj = ACTIVATIONS[name]
            except KeyError as exc:
                exc.add_note(f"Activation {name!r} not found in {list(ACTIVATIONS)=}")
                raise
            return get_activation(obj, **cfg)
        # if a class, try to instantiate it with the given configuration
        case type() as cls:
            try:
                return cls(**cfg)
            except TypeError as exc:
                exc.add_note(f"Failed to instantiate {cls} with arguments {cfg!r}")
                raise
        # if a config, extract the name and instantiate
        case None:
            if "__module__" in cfg:
                from blueprint import initialize_from_dict

                result = initialize_from_dict(cfg)
                assert isinstance(result, Activation)
                return result
            try:
                return get_activation(cfg.pop("__name__"), **cfg)
            except KeyError as exc:
                exc.add_note(f"Expected {cfg=} to contain '__name__'")
                raise
        case _:
            raise TypeError(f"Invalid argument: {kind!r}")
