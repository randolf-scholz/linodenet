r"""Generic layers."""

__all__ = [
    # Classes
    "Multiply",
    "Parallel",
    "Repeat",
    "Series",
    "Sum",
]

from collections.abc import Callable, Mapping
from typing import Any, Final, Optional, Self

import torch
from torch import Tensor, jit, nn

from linodenet.utils import initialize_from_dict


class Series(nn.Sequential):
    r"""An augmentation of nn.Sequential."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "modules": [],
    }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /, **kwargs: Any) -> Self:
        r"""Create a new instance from a configuration dictionary."""
        cfg = dict(config, **kwargs)
        modules = [initialize_from_dict(module_cfg) for module_cfg in cfg["modules"]]
        return cls(*modules)

    def __matmul__(self, other: nn.Module) -> Self:
        r"""Chain with other module."""
        cls = type(self)
        if isinstance(other, Series):
            return cls(*(*self, *other))
        return cls(*(*self, other))

    def __rmatmul__(self, other: nn.Module) -> "Series":
        r"""Chain with other module."""
        if isinstance(other, Series):
            other_type = type(other)
            return other_type(*(*other, *self))

        cls = type(self)
        return cls(*(other, *self))

    def __imatmul__(self, other: nn.Module) -> Self:
        r"""Chain with other module."""
        raise NotImplementedError(
            "`@=` not possible because `nn.Sequential` does not implement an append"
            " function."
        )

    def simplify(self) -> "Series":
        r"""Simplify the series by removing nesting."""
        modules: list[nn.Module] = []
        for module in self:
            if isinstance(module, Series):
                modules.extend(module.simplify())
            else:
                modules.append(module)
        return Series(*modules)


class Parallel(nn.ModuleList):
    r"""Modules in parallel."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "modules": [],
    }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /, **kwargs: Any) -> Self:
        r"""Create a new instance from a configuration dictionary."""
        cfg = dict(config, **kwargs)
        modules = [initialize_from_dict(module_cfg) for module_cfg in cfg["modules"]]
        return cls(modules)

    @jit.export
    def forward(self, x: Tensor) -> list[Tensor]:
        r""".. Signature:: ``(..., n) -> [..., (..., n)]``."""
        return [module(x) for module in self]

    def __matmul__(self, other: nn.Module) -> Self:
        r"""Chain with other module."""
        cls = type(self)
        if isinstance(other, Parallel):
            return cls((*self, *other))
        return cls((*self, other))

    def __rmatmul__(self, other: nn.Module) -> "Parallel":
        r"""Chain with other module."""
        if isinstance(other, Parallel):
            other_type = type(other)
            return other_type((*other, *self))

        cls = type(self)
        return cls((other, *self))

    def __imatmul__(self, other: nn.Module) -> Self:
        r"""Chain with other module."""
        raise NotImplementedError(
            "`@=` not possible because `nn.Sequential` does not implement an append"
            " function."
        )


class Repeat(nn.Sequential):
    r"""An copies of a module multiple times."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "module": None,
        "copies": 1,
        "independent": True,
    }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /, **kwargs: Any) -> Self:
        r"""Create a new instance from a configuration dictionary."""
        cfg = dict(config, **kwargs)
        num = cfg["copies"]

        if cfg["independent"]:
            modules = [initialize_from_dict(cfg["module"]) for _ in range(num)]
        else:
            module = initialize_from_dict(cfg["module"])
            modules = [module] * num

        return cls(*modules)


class Multiply(nn.Module):
    r"""Multiply inputs with a learnable parameter.

    By default multiply with a scalar.
    """

    # CONSTANTS
    signature: Final[str]
    r"""CONST: The signature"""
    learnable: Final[bool]
    r"""CONST: Whether the parameter is learnable."""

    # PARAMETERS
    kernel: Tensor
    r"""PARAM: The kernel"""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "shape": (),
        "signature": "..., -> ...",
        "learnable": True,
        "initialization": None,
    }

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        *,
        signature: str = "..., -> ...",
        learnable: bool = True,
        initialization: Optional[Callable[[tuple[int, ...]], Tensor]] = None,
    ) -> None:
        super().__init__()

        self.signature = signature
        self.learnable = learnable
        self.initialization = initialization
        initial_value = torch.randn(shape)
        self.kernel = nn.Parameter(initial_value, requires_grad=learnable)

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``... -> ...``."""
        return torch.einsum(self.signature, x, self.kernel)


class Sum(nn.ModuleList):
    r"""Add Module Outputs for same inputs."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "modules": [],
    }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], /, **kwargs: Any) -> Self:
        r"""Create a new instance from a configuration dictionary."""
        cfg = dict(config, **kwargs)
        modules = [initialize_from_dict(module_cfg) for module_cfg in cfg["modules"]]
        return cls(modules)

    def forward(self, *args, **kwargs):
        r""".. Signature:: ``... -> ...``."""
        return sum(module(*args, **kwargs) for module in self)
