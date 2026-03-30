r"""Base test class for transforms."""

__all__ = ["TestTransform"]

import torch
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

    def assert_dual(
        self,
        transform: Transform,
        inverse: Transform,
        x: Tensor,
        y: Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        encoded_x = transform.encode(x)
        decoded_x = inverse.decode(x)
        decoded_y = transform.decode(y)
        encoded_y = inverse.encode(y)
        recovered_x = inverse.encode(encoded_x)
        recovered_y = transform.encode(encoded_y)
        self.assert_close(encoded_x, decoded_x, atol=atol, rtol=rtol)
        self.assert_close(decoded_y, encoded_y, atol=atol, rtol=rtol)
        self.assert_close(recovered_x, x, atol=atol, rtol=rtol)
        self.assert_close(recovered_y, y, atol=atol, rtol=rtol)

        encoded_x, encoded_x_logabsdet = transform.encode_and_logabsdet(x)
        decoded_x, decoded_x_logabsdet = inverse.decode_and_logabsdet(x)
        decoded_y, decoded_y_logabsdet = transform.decode_and_logabsdet(y)
        encoded_y, encoded_y_logabsdet = inverse.encode_and_logabsdet(y)
        recovered_x, recovered_x_logabsdet = inverse.encode_and_logabsdet(encoded_x)
        recovered_y, recovered_y_logabsdet = transform.encode_and_logabsdet(encoded_y)
        self.assert_close(encoded_x, decoded_x, atol=atol, rtol=rtol)
        self.assert_close(decoded_y, encoded_y, atol=atol, rtol=rtol)
        self.assert_close(recovered_x, x, atol=atol, rtol=rtol)
        self.assert_close(recovered_y, y, atol=atol, rtol=rtol)
        self.assert_close(
            encoded_x_logabsdet + decoded_x_logabsdet,
            torch.zeros_like(encoded_x_logabsdet),
            atol=atol,
            rtol=rtol,
        )
        self.assert_close(
            decoded_y_logabsdet + encoded_y_logabsdet,
            torch.zeros_like(decoded_y_logabsdet),
            atol=atol,
            rtol=rtol,
        )
        self.assert_close(
            encoded_x_logabsdet + recovered_x_logabsdet,
            torch.zeros_like(encoded_x_logabsdet),
            atol=atol,
            rtol=rtol,
        )
        self.assert_close(
            encoded_y_logabsdet + recovered_y_logabsdet,
            torch.zeros_like(encoded_y_logabsdet),
            atol=atol,
            rtol=rtol,
        )
