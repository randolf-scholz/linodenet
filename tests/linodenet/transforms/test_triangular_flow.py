import pytest
import torch

from linodenet.mappings.transforms import TriangularTransform
from tests.testing import SEEDS_10

from .test_transform import TestTransform


class TestTriangularFlow(TestTransform):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5
    LOGABSDET_ATOL = 1e-6
    LOGABSDET_RTOL = 1e-6
    BATCH_SIZE = 128

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
    def test_invertibility(self, seed: int, input_size: int) -> None:
        r"""Check round trips and zero logabsdet for a unit lower-triangular flow."""
        torch.manual_seed(seed)
        flow = TriangularTransform(input_size)
        with torch.no_grad():
            flow.lower.copy_(0.1 * torch.randn_like(flow.lower).tril(diagonal=-1))

        x = torch.randn(self.BATCH_SIZE, input_size)
        y = torch.randn(self.BATCH_SIZE, input_size)
        self.assert_invertible(
            flow,
            x,
            y,
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
            logdet_atol=self.LOGABSDET_ATOL,
            logdet_rtol=self.LOGABSDET_RTOL,
        )

    def test_weight_is_unit_lower_triangular(self) -> None:
        r"""Check the constructed weight matrix has unit diagonal."""
        flow = TriangularTransform(8)
        with torch.no_grad():
            flow.lower.copy_(torch.randn_like(flow.lower))

        weight = flow.weight

        self.assert_close(weight.diag(), torch.ones(8, dtype=weight.dtype))
        self.assert_close(weight, weight.tril())

    def test_default_permutation_is_identity(self) -> None:
        r"""Check the default permutation is the identity."""
        flow = TriangularTransform(8)

        expected = torch.arange(8, dtype=torch.int64)

        assert torch.equal(flow.permutation, expected)
        assert torch.equal(flow.inverse_permutation, expected)
        assert flow.state_dict()["permutation"].dtype == torch.int64
        assert flow.state_dict()["inverse_permutation"].dtype == torch.int64

    def test_permuted_flow_matches_manual_change_of_basis(self) -> None:
        r"""Check the permutation is applied before and after the triangular map."""
        permutation = torch.tensor([2, 0, 3, 1])
        flow = TriangularTransform(4, permutation=permutation)
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
        x_perm = x[..., flow.permutation]
        y_perm = x_perm + torch.einsum("mn, ...n -> ...m", lower, x_perm)
        y_expected = y_perm[..., flow.inverse_permutation]
        x_expected = torch.linalg.solve_triangular(
            flow.weight,
            y_expected[..., flow.permutation, None],
            upper=False,
            unitriangular=True,
        ).squeeze(-1)[..., flow.inverse_permutation]

        y = flow.encode(x)
        xhat = flow.decode(y)

        self.assert_close(y, y_expected)
        self.assert_close(xhat, x_expected)
        self.assert_close(xhat, x)

    @pytest.mark.parametrize(
        ("permutation", "error"),
        [
            pytest.param(torch.tensor([[0, 1], [2, 3]]), AssertionError, id="not-1d"),
            pytest.param(torch.tensor([0, 1, 1, 3]), AssertionError, id="duplicate"),
            pytest.param(torch.tensor([0, 1, 2]), AssertionError, id="wrong-length"),
            pytest.param(torch.tensor([0, 1, 2, 4]), AssertionError, id="out-of-range"),
            pytest.param(
                torch.tensor([0.0, 1.0, 2.0, 3.0]), IndexError, id="non-integer"
            ),
        ],
    )
    def test_invalid_permutations_raise(
        self,
        permutation: torch.Tensor,
        error: type[BaseException],
    ) -> None:
        r"""Check invalid permutations are rejected."""
        with pytest.raises(error):
            TriangularTransform(4, permutation=permutation)
