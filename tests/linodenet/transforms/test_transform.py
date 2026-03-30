r"""Base test class for transforms."""

from torch import Tensor

from linodenet.mappings.transforms import Transform
from tests.testing import TestSuite


class TestTransform(TestSuite):
    def assert_right_invertible(
        self,
        transform: Transform,
        arg: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        encoded = transform.encode(arg)
        decoded = transform.decode(encoded)
        self.assert_close(decoded, arg, atol=atol, rtol=rtol)

    def assert_left_invertible(
        self,
        transform: Transform,
        arg: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        decoded = transform.decode(arg)
        encoded = transform.encode(decoded)
        self.assert_close(encoded, arg, atol=atol, rtol=rtol)

    def assert_right_invertible_with_logabsdet(
        self,
        transform: Transform,
        arg: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        encoded, forward_logabsdet = transform.encode_and_logabsdet(arg)
        decoded, inverse_logabsdet = transform.decode_and_logabsdet(encoded)
        self.assert_close(decoded, arg, atol=atol, rtol=rtol)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            forward_logabsdet.new_zeros(forward_logabsdet.shape),
            atol=atol,
            rtol=rtol,
        )

    def assert_left_invertible_with_logabsdet(
        self,
        transform: Transform,
        arg: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        decoded, inverse_logabsdet = transform.decode_and_logabsdet(arg)
        encoded, forward_logabsdet = transform.encode_and_logabsdet(decoded)
        self.assert_close(encoded, arg, atol=atol, rtol=rtol)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            forward_logabsdet.new_zeros(forward_logabsdet.shape),
            atol=atol,
            rtol=rtol,
        )

    def assert_invertible(
        self,
        transform: Transform,
        x: Tensor,
        y: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        self.assert_right_invertible(transform, x, atol=atol, rtol=rtol)
        self.assert_left_invertible(transform, y, atol=atol, rtol=rtol)
        self.assert_right_invertible_with_logabsdet(transform, x, atol=atol, rtol=rtol)
        self.assert_left_invertible_with_logabsdet(transform, y, atol=atol, rtol=rtol)
