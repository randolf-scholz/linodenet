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
from typing import TYPE_CHECKING, overload, reveal_type

from torch import nn

from linodenet.constants import EMPTY_MAP


class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    def __init__(self, modules: Iterable[M] = (), /) -> None:
        super().__init__(modules)

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[M]: ...
        @overload  # type: ignore[override]
        def __getitem__(self, index: int) -> M: ...
        @overload
        def __getitem__(self, index: slice) -> Sequence[M]: ...


class ModuleMapping[M: nn.Module](nn.ModuleDict, Mapping[str, M]):
    r"""Wrapper for ModuleDict to make it a generic Mapping type."""

    def __init__(self, modules: Mapping[str, M] = EMPTY_MAP, /) -> None:
        super().__init__(modules)

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[str]: ...
        def __getitem__(self, key: str) -> M: ...
        def keys(self) -> KeysView[str]: ...
        def values(self) -> ValuesView[M]: ...
        def items(self) -> ItemsView[str, M]: ...


seq: Sequence[nn.Linear] = ModuleSequence([nn.Linear(3, 3)])
mapping: Mapping[str, nn.Linear] = ModuleMapping({"linear": nn.Linear(3, 3)})


reveal_type(ModuleSequence[nn.Linear].__iter__)
