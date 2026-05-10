r"""Generic types for PyTorch modules."""

__all__ = [
    "Constant",
    "ModuleMapping",
    "ModuleSequence",
]

from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import TYPE_CHECKING, Never, Self, overload

import torch
from torch import Tensor, nn
from torch.nn import Module, ModuleDict, ModuleList


class Constant(Module):
    r"""Module that returns a learned constant tensor."""

    value: Tensor
    r"""PARAM: Constant tensor returned by the module."""

    def __init__(
        self,
        shape_or_tensor: tuple[int, ...] | Tensor,
        /,
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        match shape_or_tensor:
            case tuple(shape):
                self.value = nn.Parameter(torch.randn(shape), requires_grad=learnable)
                nn.init.kaiming_uniform_(self.value)
            case Tensor() as tensor:
                self.value = nn.Parameter(tensor, requires_grad=learnable)
            case _:
                raise TypeError(
                    f"Expected shape or tensor, got {type(shape_or_tensor)!r}"
                )

    def forward(self, _: Tensor) -> Tensor:
        return self.value


class ModuleSequence[M: Module](ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    @classmethod
    def from_iterable(cls, modules: Iterable[M], /) -> ModuleSequence[M]:
        r"""Initialize from an iterable of modules."""
        return ModuleSequence(modules)

    if TYPE_CHECKING:
        # We add these at type-checking time to help mypy and pyright
        # Since they are skipped at runtime, they won't interfere with JIT compilation

        def __init__(self, modules: Iterable[M] = (), /) -> None: ...
        def __iter__(self) -> Iterator[M]: ...

        @overload  # type: ignore[override]
        def __getitem__(self, index: int, /) -> M: ...  # pyrefly: ignore[bad-override]
        @overload
        def __getitem__(self, index: slice, /) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]


class ModuleMapping[M: Module](ModuleDict, Mapping[str, M]):
    r"""Wrapper for ModuleDict to make it a generic Mapping type."""

    def __hash__(self) -> int:
        # NOTE: fixes https://github.com/pytorch/pytorch/issues/110959
        return hash(tuple(self.items()))

    if TYPE_CHECKING:
        # We add these at type-checking time to help mypy and pyright
        # Since they are skipped at runtime, they won't interfere with JIT compilation

        @overload
        def __init__(self: ModuleMapping[Never], /) -> None: ...
        @overload
        def __init__(self, modules: Mapping[str, M], /) -> None: ...

        def __iter__(self) -> Iterator[str]: ...
        def __getitem__(self, key: str) -> M: ...
        def __contains__(self, key: str) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        def keys(self) -> KeysView[str]: ...
        def values(self) -> ValuesView[M]: ...
        def items(self) -> ItemsView[str, M]: ...
