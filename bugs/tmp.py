from typing import Sequence, TypeVar, cast

X = TypeVar("X")
Y = TypeVar("Y")


x: Sequence[int] | Sequence[str] = cast(Sequence[int] | Sequence[str], [])
y: Sequence[int | str] = cast(Sequence[int | str], [])

reveal_type(x)
reveal_type(y)


z: Sequence[int | str] = x
w: Sequence[int] | Sequence[str] = y  # ✘


reveal_type(z)
reveal_type(w)
