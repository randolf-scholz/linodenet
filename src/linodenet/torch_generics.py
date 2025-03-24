r"""Generic types for PyTorch modules."""

__all__ = ["ModuleSequence", "ModuleMapping"]


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

from torch import nn


class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    if TYPE_CHECKING:

        @overload
        def __init__(self: "ModuleSequence[Never]", /) -> None: ...
        @overload
        def __init__(self, modules: Iterable[M], /) -> None: ...

        def __iter__(self) -> Iterator[M]: ...
        @overload  # type: ignore[override]
        def __getitem__(self, index: int) -> M: ...
        @overload
        def __getitem__(self, index: slice) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]


class ModuleMapping[M: nn.Module](nn.ModuleDict, Mapping[str, M]):
    r"""Wrapper for ModuleDict to make it a generic Mapping type."""

    def __hash__(self) -> int:
        # NOTE: fixes https://github.com/pytorch/pytorch/issues/110959
        return hash(tuple(self.items()))

    if TYPE_CHECKING:

        @overload
        def __init__(self: "ModuleMapping[Never]", /) -> None: ...
        @overload
        def __init__(self, modules: Mapping[str, M], /) -> None: ...

        def __iter__(self) -> Iterator[str]: ...
        def __getitem__(self, key: str) -> M: ...
        def keys(self) -> KeysView[str]: ...
        def values(self) -> ValuesView[M]: ...
        def items(self) -> ItemsView[str, M]: ...
