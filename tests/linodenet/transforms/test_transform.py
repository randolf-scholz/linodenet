r"""Base test class for transforms."""

from torch import Tensor

from linodenet.mappings.transforms import Transform
from tests.testing import TestSuite


class TestTransform(TestSuite):
    def assert_reversible(
        self,
        transform: Transform,
        arg: Tensor,
        atol: float,
        rtol: float,
    ) -> None:
        encoded = transform.encode(arg)
        decoded = transform.decode(encoded)
        self.assert_close(decoded, arg, atol=atol, rtol=rtol)

        decoded = transform.decode(arg)
        encoded = transform.encode(decoded)
        self.assert_close(encoded, arg, atol=atol, rtol=rtol)
