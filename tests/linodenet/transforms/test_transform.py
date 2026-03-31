r"""Base test class for transforms."""

__all__ = ["TestTransform"]

import torch
from torch import Tensor

from linodenet.mappings.transforms import Transform
from tests.testing import TestSuite


class TestTransform(TestSuite):
    def assert_logabsdet_matches_finite_difference_volume_change(
        self,
        transform: Transform,
        arg: Tensor,
        *,
        step: float,
        atol: float,
        rtol: float,
    ) -> None:
        _, logabsdet = transform.encode_and_logabsdet(arg)
        estimates = torch.stack(
            [
                self._finite_difference_logabsdet(
                    transform,
                    point,
                    step=step,
                )
                for point in arg
            ]
        )
        self.assert_close(logabsdet, estimates, atol=atol, rtol=rtol)

    def _finite_difference_logabsdet(
        self,
        transform: Transform,
        x: Tensor,
        *,
        step: float,
    ) -> Tensor:
        frame = torch.randn(x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype)
        orthogonal_frame, _ = torch.linalg.qr(frame)
        columns = []

        for direction in orthogonal_frame.mT:
            y_plus = transform.encode(x + step * direction)
            y_minus = transform.encode(x - step * direction)
            columns.append((y_plus - y_minus) / (2 * step))

        jacobian = torch.stack(columns, dim=-1)
        _, logabsdet = torch.linalg.slogdet(jacobian)
        return logabsdet

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
            torch.zeros_like(forward_logabsdet),
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
            torch.zeros_like(forward_logabsdet),
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
        y_primal = transform.encode(x)
        x_primal = transform.decode(y)
        y_dual = inverse.decode(x)
        x_dual = inverse.encode(y)
        x_recovered = inverse.encode(y_primal)
        y_recovered = transform.encode(x_dual)
        self.assert_close(y_primal, y_dual, atol=atol, rtol=rtol)
        self.assert_close(x_primal, x_dual, atol=atol, rtol=rtol)
        self.assert_close(x_recovered, x, atol=atol, rtol=rtol)
        self.assert_close(y_recovered, y, atol=atol, rtol=rtol)

        y_primal, y_primal_logabsdet = transform.encode_and_logabsdet(x)
        x_primal, x_primal_logabsdet = transform.decode_and_logabsdet(y)
        y_dual, y_dual_logabsdet = inverse.decode_and_logabsdet(x)
        x_dual, x_dual_logabsdet = inverse.encode_and_logabsdet(y)
        x_recovered, x_recovered_logabsdet = inverse.encode_and_logabsdet(y_primal)
        y_recovered, y_recovered_logabsdet = transform.encode_and_logabsdet(x_dual)
        self.assert_close(y_primal, y_dual, atol=atol, rtol=rtol)
        self.assert_close(x_primal, x_dual, atol=atol, rtol=rtol)
        self.assert_close(x_recovered, x, atol=atol, rtol=rtol)
        self.assert_close(y_recovered, y, atol=atol, rtol=rtol)
        self.assert_close(y_primal_logabsdet, y_dual_logabsdet, atol=atol, rtol=rtol)
        self.assert_close(x_primal_logabsdet, x_dual_logabsdet, atol=atol, rtol=rtol)
        self.assert_close(
            y_primal_logabsdet + x_recovered_logabsdet,
            torch.zeros_like(y_primal_logabsdet),
            atol=atol,
            rtol=rtol,
        )
        self.assert_close(
            x_dual_logabsdet + y_recovered_logabsdet,
            torch.zeros_like(x_dual_logabsdet),
            atol=atol,
            rtol=rtol,
        )
