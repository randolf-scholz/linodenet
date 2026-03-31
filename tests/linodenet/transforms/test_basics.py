import pytest
import torch
from torch import Tensor


def _relu_derivative(x: Tensor, /) -> Tensor:
    return (x > 0).to(dtype=x.dtype)


def _identity_derivative(x: Tensor, /) -> Tensor:
    return torch.ones_like(x)


def _tanh_derivative(x: Tensor, /) -> Tensor:
    return 1 - torch.tanh(x).square()


def _make_scaled_matrix(input_size: int, /) -> Tensor:
    matrix = torch.randn(input_size, input_size)
    matrix = 0.5 * matrix / torch.linalg.matrix_norm(matrix)
    return matrix.requires_grad_()


ACTIVATION_DERIVATIVES = {
    "relu": _relu_derivative,
    "identity": _identity_derivative,
    "tanh": _tanh_derivative,
}


@pytest.mark.parametrize("activation", ["relu", "identity", "tanh"])
def test_preactivation(activation: str) -> None:
    r"""Check $∇ₐ \log |\det Dₓg(x, A)|$ for $g(x, A)=x+f(Ax)$.

    Remarks:
        Identity does not kill the log-det gradient in the preactivation form:
        it still yields nonzero gradients with respect to $A$.
        ReLU also yields nonzero gradients, but only through active
        preactivation coordinates, so inactive rows become exactly zero.
        Tanh yields nonzero gradients without that hard masking behavior.
    """
    torch.manual_seed(0)
    input_size = 7
    x = torch.randn(input_size)
    A = _make_scaled_matrix(input_size)

    slopes = ACTIVATION_DERIVATIVES[activation](A @ x)
    jacobian = torch.eye(input_size, dtype=A.dtype) + torch.diag(slopes) @ A
    loss = torch.linalg.slogdet(jacobian).logabsdet
    loss.backward()
    assert A.grad is not None
    grad = A.grad

    assert torch.linalg.norm(grad) > 0

    match activation:
        case "identity":
            assert torch.all(torch.linalg.vector_norm(grad, dim=1) > 0)
        case "relu":
            inactive = slopes == 0
            assert torch.any(inactive)
            assert torch.allclose(
                grad[inactive],
                torch.zeros_like(grad[inactive]),
                atol=0.0,
                rtol=0.0,
            )
            assert torch.linalg.norm(grad[~inactive]) > 0
        case "tanh":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(grad, dim=1) > 0)
        case _:
            raise AssertionError(f"Unhandled activation: {activation}")


@pytest.mark.parametrize("activation", ["relu", "identity", "tanh"])
def test_postactivation(activation: str) -> None:
    r"""Check $∇ₐ \log |\det Dₓg(x, A)|$ for $g(x, A)=x+Af(x)$.

    Remarks:
        Identity does not kill the log-det gradient in the postactivation form:
        it still yields nonzero gradients with respect to $A$.
        ReLU also yields nonzero gradients, but only through active input
        coordinates, so inactive columns become exactly zero.
        Tanh yields nonzero gradients without that hard masking behavior.
    """
    torch.manual_seed(0)
    input_size = 7
    x = torch.randn(input_size)
    A = _make_scaled_matrix(input_size)

    slopes = ACTIVATION_DERIVATIVES[activation](x)
    jacobian = torch.eye(input_size, dtype=A.dtype) + A @ torch.diag(slopes)
    loss = torch.linalg.slogdet(jacobian).logabsdet
    loss.backward()
    assert A.grad is not None
    grad = A.grad

    assert torch.linalg.norm(grad) > 0

    match activation:
        case "identity":
            assert torch.all(torch.linalg.vector_norm(grad, dim=0) > 0)
        case "relu":
            inactive = slopes == 0
            assert torch.any(inactive)
            assert torch.allclose(
                grad[:, inactive],
                torch.zeros_like(grad[:, inactive]),
                atol=0.0,
                rtol=0.0,
            )
            assert torch.linalg.norm(grad[:, ~inactive]) > 0
        case "tanh":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(grad, dim=0) > 0)
        case _:
            raise AssertionError(f"Unhandled activation: {activation}")
