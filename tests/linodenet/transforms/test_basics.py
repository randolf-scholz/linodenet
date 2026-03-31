r"""Basic gradient-structure checks for residual transform Jacobians."""

import pytest
import torch
from torch import Tensor
from torch.func import grad, vmap


def _make_scaled_matrix(input_size: int, /) -> Tensor:
    matrix = torch.randn(input_size, input_size)
    matrix = 0.5 * matrix / torch.linalg.matrix_norm(matrix)
    return matrix.requires_grad_()


ACTIVATIONS: dict = {
    "relu": torch.relu,
    "identity": lambda x: x,
    "tanh": torch.tanh,
    "elu": torch.nn.functional.elu,
}


@pytest.mark.parametrize("activation", ["relu", "identity", "tanh", "elu"])
def test_preactivation(activation: str) -> None:
    r"""Check $∇ₐ \log |\det Dₓg(x, A)|$ for $g(x, A)=x+f(Ax)$.

    Remarks:
        - Identity does not kill the log-det gradient in the preactivation
          form: it still yields nonzero gradients with respect to $A$.
        - ReLU also yields nonzero gradients, but only through active
          preactivation coordinates, so inactive rows become exactly zero.
        - Tanh yields nonzero gradients without that hard masking behavior.
        - ELU behaves similarly to tanh here: its derivative stays strictly
          positive, so every row can contribute to the gradient.
    """
    torch.manual_seed(0)
    input_size = 7
    x = torch.randn(input_size)
    A = _make_scaled_matrix(input_size)
    activation_fn = ACTIVATIONS[activation]

    slopes = vmap(grad(activation_fn))(A @ x)
    jacobian = torch.eye(input_size, dtype=A.dtype) + torch.diag(slopes) @ A
    loss = torch.linalg.slogdet(jacobian).logabsdet
    loss.backward()
    assert A.grad is not None

    assert torch.linalg.norm(A.grad) > 0

    match activation:
        case "identity":
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=1) > 0)
        case "relu":
            inactive = slopes == 0
            assert torch.any(inactive)
            assert torch.allclose(
                A.grad[inactive],
                torch.zeros_like(A.grad[inactive]),
                atol=0.0,
                rtol=0.0,
            )
            assert torch.linalg.norm(A.grad[~inactive]) > 0
        case "tanh":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=1) > 0)
        case "elu":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=1) > 0)
        case _:
            raise AssertionError(f"Unhandled activation: {activation}")


@pytest.mark.parametrize("activation", ["relu", "identity", "tanh", "elu"])
def test_postactivation(activation: str) -> None:
    r"""Check $∇ₐ \log |\det Dₓg(x, A)|$ for $g(x, A)=x+Af(x)$.

    Remarks:
        - Identity does not kill the log-det gradient in the postactivation
          form: it still yields nonzero gradients with respect to $A$.
        - ReLU also yields nonzero gradients, but only through active input
          coordinates, so inactive columns become exactly zero.
        - Tanh yields nonzero gradients without that hard masking behavior.
        - ELU behaves similarly to tanh here: its derivative stays strictly
          positive, so every column can contribute to the gradient.
    """
    torch.manual_seed(0)
    input_size = 7
    x = torch.randn(input_size)
    A = _make_scaled_matrix(input_size)
    activation_fn = ACTIVATIONS[activation]

    slopes = vmap(grad(activation_fn))(x)
    jacobian = torch.eye(input_size, dtype=A.dtype) + A @ torch.diag(slopes)
    loss = torch.linalg.slogdet(jacobian).logabsdet
    loss.backward()
    assert A.grad is not None

    assert torch.linalg.norm(A.grad) > 0

    match activation:
        case "identity":
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=0) > 0)
        case "relu":
            inactive = slopes == 0
            assert torch.any(inactive)
            assert torch.allclose(
                A.grad[:, inactive],
                torch.zeros_like(A.grad[:, inactive]),
                atol=0.0,
                rtol=0.0,
            )
            assert torch.linalg.norm(A.grad[:, ~inactive]) > 0
        case "tanh":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=0) > 0)
        case "elu":
            assert torch.all(slopes > 0)
            assert torch.all(torch.linalg.vector_norm(A.grad, dim=0) > 0)
        case _:
            raise AssertionError(f"Unhandled activation: {activation}")
