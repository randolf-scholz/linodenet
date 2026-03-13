import pytest
import torch

from linodenet.bijections import TriangularFlow
from tests.linodenet.bijections.fixtures import SEEDS


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
def test_invertibility(seed: int, input_size: int) -> None:
    r"""Check round trips and zero logabsdet for a unit lower-triangular flow."""
    torch.manual_seed(seed)
    value_atol = 1e-5
    value_rtol = 1e-5

    batch_size = 128
    flow = TriangularFlow(input_size)
    with torch.no_grad():
        flow.lower.copy_(0.1 * torch.randn_like(flow.lower).tril(diagonal=-1))

    x = torch.randn(batch_size, input_size)
    y, forward_logabsdet = flow.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

    assert y.shape == x.shape
    assert xhat.shape == x.shape
    assert forward_logabsdet.shape == (batch_size,)
    assert inverse_logabsdet.shape == (batch_size,)
    assert torch.allclose(xhat, x, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(forward_logabsdet, torch.zeros_like(forward_logabsdet))
    assert torch.allclose(inverse_logabsdet, torch.zeros_like(inverse_logabsdet))

    y = torch.randn(batch_size, input_size)
    x, inverse_logabsdet = flow.decode_and_logabsdet(y)
    yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

    assert torch.allclose(yhat, y, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(forward_logabsdet, torch.zeros_like(forward_logabsdet))
    assert torch.allclose(inverse_logabsdet, torch.zeros_like(inverse_logabsdet))


def test_weight_is_unit_lower_triangular() -> None:
    r"""Check the constructed weight matrix has unit diagonal."""
    flow = TriangularFlow(8)
    with torch.no_grad():
        flow.lower.copy_(torch.randn_like(flow.lower))

    weight = flow.weight

    assert torch.allclose(weight.diag(), torch.ones(8, dtype=weight.dtype))
    assert torch.allclose(weight, weight.tril())


def test_default_permutation_is_identity() -> None:
    r"""Check the default permutation is the identity."""
    flow = TriangularFlow(8)

    expected = torch.arange(8, dtype=torch.int64)

    assert torch.equal(flow.perm, expected)
    assert torch.equal(flow.invperm, expected)
    assert flow.state_dict()["perm"].dtype == torch.int64
    assert flow.state_dict()["invperm"].dtype == torch.int64


def test_permuted_flow_matches_manual_change_of_basis() -> None:
    r"""Check the permutation is applied before and after the triangular map."""
    permutation = torch.tensor([2, 0, 3, 1])
    flow = TriangularFlow(4, permutation=permutation)
    with torch.no_grad():
        flow.lower.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0],
                    [-0.1, 0.2, 0.0, 0.0],
                    [0.4, -0.2, 0.5, 0.0],
                ]
            )
        )

    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, 2.0]])
    lower = flow.lower.tril(diagonal=-1)
    x_perm = x[..., flow.perm]
    y_perm = x_perm + torch.einsum("mn, ...n -> ...m", lower, x_perm)
    y_expected = y_perm[..., flow.invperm]
    x_expected = torch.linalg.solve_triangular(
        flow.weight,
        y_expected[..., flow.perm, None],
        upper=False,
        unitriangular=True,
    ).squeeze(-1)[..., flow.invperm]

    y = flow.encode(x)
    xhat = flow.decode(y)

    assert torch.allclose(y, y_expected)
    assert torch.allclose(xhat, x_expected)
    assert torch.allclose(xhat, x)


@pytest.mark.parametrize(
    ("permutation", "error"),
    [
        pytest.param(torch.tensor([[0, 1], [2, 3]]), AssertionError, id="not-1d"),
        pytest.param(torch.tensor([0, 1, 1, 3]), AssertionError, id="duplicate"),
        pytest.param(torch.tensor([0, 1, 2]), AssertionError, id="wrong-length"),
        pytest.param(torch.tensor([0, 1, 2, 4]), AssertionError, id="out-of-range"),
        pytest.param(torch.tensor([0.0, 1.0, 2.0, 3.0]), IndexError, id="non-integer"),
    ],
)
def test_invalid_permutations_raise(
    permutation: torch.Tensor,
    error: type[BaseException],
) -> None:
    r"""Check invalid permutations are rejected."""
    with pytest.raises(error):
        TriangularFlow(4, permutation=permutation)
