r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "Repeat",
    "Series",
    "Parallel",
    "Multiply",
]

from collections.abc import Callable
from typing import Any, Final, List, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.util._util import deep_dict_update, initialize_from_config


class Series(nn.Sequential):
    r"""An augmentation of nn.Sequential."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "modules": [None],
    }

    def __init__(self, *args: Any, **HP: Any) -> None:
        self.CFG = HP = deep_dict_update(self.HP, HP)

        modules: list[nn.Module] = []

        if HP["modules"] != [None]:
            del HP["modules"][0]
            for _, layer in enumerate(HP["modules"]):
                module = initialize_from_config(layer)
                modules.append(module)

        modules = list(args) + modules
        super().__init__(*modules)

    def __matmul__(self, other: nn.Module) -> Series:
        r"""Chain with other module."""
        if isinstance(other, Series):
            return Series(*(*self, *other))
        return Series(*(*self, other))

    def __rmatmul__(self, other: nn.Module) -> Series:
        r"""Chain with other module."""
        if isinstance(other, Series):
            return Series(*(*other, *self))
        return Series(*(other, *self))

    def __imatmul__(self, other: nn.Module) -> Series:
        r"""Chain with other module."""
        raise NotImplementedError(
            "`@=` not possible because `nn.Sequential` does not implement an append function."
        )

    def simplify(self: Series) -> Series:
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
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "modules": [None],
    }

    def __init__(self, *args: Any, **HP: Any) -> None:
        self.CFG = HP = deep_dict_update(self.HP, HP)

        modules: list[nn.Module] = []

        if HP["modules"] != [None]:
            del HP["modules"][0]
            for _, layer in enumerate(HP["modules"]):
                module = initialize_from_config(layer)
                modules.append(module)

        modules = list(args) + modules

        super().__init__(modules)

    @jit.export
    def forward(self, x: Tensor) -> list[Any]:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        result: List[Any] = []

        for module in self:
            result.append(module(x))

        return result

    def __matmul__(self, other: nn.Module) -> Parallel:
        r"""Chain with other module."""
        if isinstance(other, Parallel):
            return Parallel(*(*self, *other))
        return Parallel(*(*self, other))

    def __rmatmul__(self, other: nn.Module) -> Parallel:
        r"""Chain with other module."""
        return Parallel(*(other, *self))

    def __imatmul__(self, other: nn.Module) -> Parallel:
        r"""Chain with other module."""
        raise NotImplementedError(
            "`@=` not possible because `nn.Sequential` does not implement an append function."
        )


class Repeat(nn.Sequential):
    r"""An copies of a module multiple times."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "module": None,
        "copies": 1,
        "independent": True,
    }

    def __init__(self, **HP: Any) -> None:
        self.CFG = HP = deep_dict_update(self.HP, HP)

        copies: list[nn.Module] = []

        for _ in range(HP["copies"]):
            if isinstance(HP["module"], nn.Module):
                module = HP["module"]
            else:
                module = initialize_from_config(HP["module"])

            if HP["independent"]:
                copies.append(module)
            else:
                copies = [module] * HP["copies"]
                break

        HP["module"] = str(HP["module"])
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
        r"""Forward pass."""
        return torch.einsum(self.signature, x, self.kernel)
