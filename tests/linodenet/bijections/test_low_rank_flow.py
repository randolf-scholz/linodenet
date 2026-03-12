import pytest
import torch

from linodenet.bijections import LowRankFlow
from tests.linodenet.bijections.fixtures import SEEDS


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
@pytest.mark.parametrize("rank", [1, 2, 4], ids="rank={}".format)
def test_invertibility(seed: int, input_size: int, rank: int) -> None:
    r"""Check forward/inverse round trips and logabsdet cancellation."""
    torch.manual_seed(seed)
    value_atol = 1e-4
    value_rtol = 1e-5
    logabsdet_atol = 1e-5
    logabsdet_rtol = 1e-5

    batch_size = 128
    flow = LowRankFlow(input_size, rank=min(rank, input_size))

    print(f"Test Case {seed=}, {input_size=}, {rank=}")

    def check_forward() -> None:
        x = torch.randn(batch_size, input_size)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

        assert y.shape == x.shape
        assert forward_logabsdet.shape == (batch_size,)
        assert xhat.shape == x.shape
        assert inverse_logabsdet.shape == (batch_size,)

        forward_inverse_abs_error = (xhat - x).abs()
        forward_inverse_rel_error = forward_inverse_abs_error / torch.maximum(
            torch.maximum(xhat.abs(), x.abs()),
            torch.full_like(xhat, torch.finfo(xhat.dtype).eps),
        )
        forward_inverse_logabsdet_error = (forward_logabsdet + inverse_logabsdet).abs()

        print(
            "forward -> inverse "
            f"\n\tvalue     max_abs_error={forward_inverse_abs_error.max():.6e}"
            f"   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tvalue     max_rel_error={forward_inverse_rel_error.max():.6e}"
            f"   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tlogabsdet max_abs_error={forward_inverse_logabsdet_error.max():.6e}"
            f"   (atol={logabsdet_atol}, rtol={logabsdet_rtol})"
        )

        assert torch.allclose(xhat, x, atol=value_atol, rtol=value_rtol), (
            f"forward_inverse max_abs_error={forward_inverse_abs_error.max():.6e}, "
            f"max_rel_error={forward_inverse_rel_error.max():.6e}, "
        )
        assert torch.allclose(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=logabsdet_atol,
            rtol=logabsdet_rtol,
        ), (
            f"forward_inverse_logabsdet max_abs_error="
            f"{forward_inverse_logabsdet_error.max():.6e}, "
        )

    def check_inverse() -> None:
        y = torch.randn(batch_size, input_size)
        x, inverse_logabsdet = flow.decode_and_logabsdet(y)
        yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

        assert x.shape == y.shape
        assert inverse_logabsdet.shape == (batch_size,)
        assert yhat.shape == y.shape
        assert forward_logabsdet.shape == (batch_size,)

        inverse_forward_abs_error = (yhat - y).abs()
        inverse_forward_rel_error = inverse_forward_abs_error / torch.maximum(
            torch.maximum(yhat.abs(), y.abs()),
            torch.full_like(yhat, torch.finfo(yhat.dtype).eps),
        )
        inverse_forward_logabsdet_error = (inverse_logabsdet + forward_logabsdet).abs()

        print(
            "inverse -> forward "
            f"\n\tvalue     max_abs_error={inverse_forward_abs_error.max():.6e}"
            f"   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tvalue     max_rel_error={inverse_forward_rel_error.max():.6e}"
            f"   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tlogabsdet max_abs_error={inverse_forward_logabsdet_error.max():.6e}"
            f"   (atol={logabsdet_atol}, rtol={logabsdet_rtol})"
        )

        assert torch.allclose(yhat, y, atol=value_atol, rtol=value_rtol), (
            f"inverse_forward max_abs_error={inverse_forward_abs_error.max():.6e}, "
            f"max_rel_error={inverse_forward_rel_error.max():.6e}, "
            f"{value_atol=}, {value_rtol=}"
        )
        assert torch.allclose(
            inverse_logabsdet + forward_logabsdet,
            torch.zeros_like(inverse_logabsdet),
            atol=logabsdet_atol,
            rtol=logabsdet_rtol,
        ), (
            f"inverse_forward_logabsdet max_abs_error="
            f"{inverse_forward_logabsdet_error.max().item():.6e}, "
        )

    check_forward()
    check_inverse()
