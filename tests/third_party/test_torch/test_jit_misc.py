r"""Miscellaneous tests for TorchScript JIT."""

from torch import Tensor, jit


def test_jit_ternary() -> None:
    def ternary_test(x: Tensor, y: bool) -> Tensor:
        r = x if y else -x
        return 2 * r

    jit.script(ternary_test)
