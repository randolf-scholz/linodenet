r"""Generic layers."""

__all__ = [
    # Classes
    "Multiply",
    "Parallel",
    "Repeat",
    "Series",
    "Sum",
]

from collections.abc import Callable, Iterable
from typing import Any, Final, Optional, Self

import torch
from torch import Tensor, jit, nn

from linodenet.utils._utils import deep_dict_update, initialize_from_dict


class Series(nn.Sequential):
    r"""An augmentation of nn.Sequential."""

    HP = {
        "__name__": __qualname__,
        "__module__": __module__,
        "modules": [None],
    }

    def __init__(self, *modules: nn.Module, **cfg: Any) -> None:
        config = deep_dict_update(self.HP, cfg)

        layers: list[nn.Module] = list(modules)

        if config["modules"] != [None]:
            del config["modules"][0]
            for _, layer in enumerate(config["modules"]):
                module = initialize_from_dict(layer)
                layers.append(module)

        super().__init__(*layers)

    def __matmul__(self, other: nn.Module) -> "Series":
        r"""Chain with other module."""
        cls = type(self)
        if isinstance(other, Series):
            return cls(*(*self, *other))
        return cls(*(*self, other))

    def __rmatmul__(self, other: nn.Module) -> "Series":
        r"""Chain with other module."""
        cls = type(self)
        if isinstance(other, Series):
            return cls(*(*other, *self))
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
        "__module__": __module__,
        "modules": [None],
    }

    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = None, /, **cfg: Any
    ) -> None:
        config = deep_dict_update(self.HP, cfg)

        layers: list[nn.Module] = [] if modules is None else list(modules)

        if config["modules"] != [None]:
            del config["modules"][0]
            for _, layer in enumerate(config["modules"]):
                module = initialize_from_dict(layer)
                layers.append(module)

        super().__init__(layers)

    @jit.export
    def forward(self, x: Tensor) -> list[Tensor]:
        r""".. Signature:: ``(..., n) -> [..., (..., n)]``."""
        return [module(x) for module in self]

    def __matmul__(self, other: nn.Module) -> "Parallel":
        r"""Chain with other module."""
        if isinstance(other, Parallel):
            return Parallel((*self, *other))
        return Parallel((*self, other))

    def __rmatmul__(self, other: nn.Module) -> "Parallel":
        r"""Chain with other module."""
        return Parallel((other, *self))

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
        "__module__": __module__,
        "module": None,
        "copies": 1,
        "independent": True,
    }

    def __init__(self, *modules: nn.Module, **cfg: Any) -> None:
        config = deep_dict_update(self.HP, cfg)

        copies: list[nn.Module] = list(modules)

        for _ in range(config["copies"]):
            if isinstance(config["module"], nn.Module):
                module = config["module"]
            else:
                module = initialize_from_dict(config["module"])

            if config["independent"]:
                copies.append(module)
            else:
                copies = [module] * config["copies"]
                break

        config["module"] = str(config["module"])
        super().__init__(*copies)


class Multiply(nn.Module):
    r"""Multiply inputs with a learnable parameter.

    By default multiply with a scalar.
    """

    signature: Final[str]
    r"""CONST: The signature"""
    learnable: Final[bool]
    r"""CONST: Whether the parameter is learnable."""

    kernel: Tensor
    r"""PARAM: The kernel"""

    def __init__(
        self,
        shape: tuple[int, ...] = (),
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
        "__module__": __module__,
        "modules": [],
    }

    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = None, **cfg: Any
    ) -> None:
        config = deep_dict_update(self.HP, cfg)

        layers: list[nn.Module] = [] if modules is None else list(modules)

        for layer in config["modules"]:
            module = initialize_from_dict(layer)
            layers.append(module)

        super().__init__(layers)

    def forward(self, *args, **kwargs):
        r""".. Signature:: ``... -> ...``."""
        return sum(module(*args, **kwargs) for module in self)
