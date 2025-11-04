r"""Test type hints for algebra module."""

from collections.abc import Sequence
from typing import assert_type

from linodenet.algebra import Fn, Parallel, Seq, parallel


def check_upcast_sequence[T](s: Sequence[T]) -> Seq[T]:
    return s


def check_typing_parallel[X: Fn, Y: Fn](
    *,
    f1: X,
    l1: list[X],
    s1: Parallel[X],
    f2: Y,
    l2: list[Y],
    s2: Parallel[Y],
) -> None:
    # foo + foo
    assert_type(parallel(f1, f1), Parallel[X])
    assert_type(parallel(f1, f2), Parallel[X | Y])
    # foo + list
    assert_type(parallel(f1, l1), Parallel[X])
    assert_type(parallel(f1, l2), Parallel[X | Y])
    # list + list
    assert_type(parallel(l1, l1), Parallel[X])
    assert_type(parallel(l1, l2), Parallel[X | Y])
    # foo + seq
    assert_type(parallel(f1, s1), Parallel[X | Parallel[X]])
    assert_type(parallel(f1, s2), Parallel[X | Parallel[Y]])
    # list + seq
    assert_type(parallel(l1, s1), Parallel[X | Parallel[X]])
    assert_type(parallel(l1, s2), Parallel[X | Parallel[Y]])
    # seq + seq
    assert_type(parallel(s1, s1), Parallel[Parallel[X]])
    assert_type(parallel(s1, s2), Parallel[Parallel[X] | Parallel[Y]])
