r"""Test initialize torch.Size from `Sequence[int]`."""

from collections.abc import Sequence
from typing import SupportsIndex, overload

import torch


def test_size_from_int_sequence() -> None:
    class MyRange(Sequence[int]):
        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        @overload
        def __getitem__(self, index: int, /) -> int: ...
        @overload
        def __getitem__(self, index: slice, /) -> MyRange: ...
        def __getitem__(self, index: int | slice, /) -> int | MyRange:  # pyright: ignore[reportIncompatibleMethodOverride]
            if isinstance(index, slice):
                if index.start not in {None, 0}:
                    raise ValueError
                return MyRange(index.stop - index.start)
            if 0 <= index < self.size:
                return index
            raise IndexError("Index out of range")

    seq = MyRange(5)

    # Test with a sequence
    torch.Size(seq)


def test_size_from_index_sequence() -> None:
    class MyInt:
        def __init__(self, value: int) -> None:
            self.value = value

        def __index__(self) -> int:
            return self.value

    class MyRange(Sequence[SupportsIndex]):
        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        @overload
        def __getitem__(self, index: int, /) -> MyInt: ...
        @overload
        def __getitem__(self, index: slice, /) -> MyRange: ...
        def __getitem__(self, index: int | slice, /) -> MyInt | MyRange:  # pyright: ignore[reportIncompatibleMethodOverride]
            if isinstance(index, slice):
                if index.start not in {None, 0}:
                    raise ValueError
                return MyRange(index.stop - index.start)
            if 0 <= index < self.size:
                return MyInt(index)
            raise IndexError("Index out of range")

    seq = MyRange(5)

    # Test with a sequence
    torch.Size(seq)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
