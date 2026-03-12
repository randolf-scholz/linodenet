import matplotlib.pyplot as plt
import pytest
import torch

from linodenet.bijections import SplineFlow
from tests.linodenet.bijections.fixtures import SEEDS
from tests.utils.project import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("layers", [1, 2, 3, 4], ids="layers={}".format)
@pytest.mark.parametrize("bins", [1, 2, 4, 8], ids="bins={}".format)
def test_invertibility(seed: int, layers, bins) -> None:
    torch.manual_seed(seed)
    value_atol = 1.5e-2 * layers
    value_rtol = 1e-3
    logabsdet_atol = 5e-3 * layers
    logabsdet_rtol = 1e-3

    batch_size = 128
    num_heads = 4
    flow = SplineFlow(
        num_heads,
        num_flow_layers=layers,
        num_bins=bins,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.0, 3.0),
    )

    print(f"Test Case {seed=}, {layers=}, {bins=}")

    def check_forward() -> None:
        x = torch.randn(batch_size, num_heads)
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
            f"\n\tvalue     max_abs_error={forward_inverse_abs_error.max():.6e}   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tvalue     max_rel_error={forward_inverse_rel_error.max():.6e}   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tlogabsdet max_abs_error={forward_inverse_logabsdet_error.max():.6e}   (atol={logabsdet_atol}, rtol={logabsdet_rtol})"
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
        y = torch.randn(batch_size, num_heads)
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
            f"\n\tvalue     max_abs_error={inverse_forward_abs_error.max():.6e}   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tvalue     max_rel_error={inverse_forward_rel_error.max():.6e}   (atol={value_atol}, rtol={value_rtol})"
            f"\n\tlogabsdet max_abs_error={inverse_forward_logabsdet_error.max():.6e}   (atol={logabsdet_atol}, rtol={logabsdet_rtol})"
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


def test_single_spline_can_learn_monotone_function() -> None:
    torch.manual_seed(0)

    model = SplineFlow(
        1,
        num_flow_layers=1,
        num_bins=8,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.5, 3.5),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.linspace(-4.0, 4.0, steps=256).unsqueeze(-1)
    y = x + 3 + 0.5 * torch.sin(x + 3)

    with torch.no_grad():
        initial_prediction = model.encode(x)
        initial_loss = torch.mean((initial_prediction - y) ** 2)

    for _ in range(400):
        optimizer.zero_grad()
        prediction = model.encode(x)
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_prediction = model.encode(x)
        final_loss = torch.mean((final_prediction - y) ** 2)
        max_abs_error = (final_prediction - y).abs().max()

    fig, ax = plt.subplots(figsize=(6.5, 4.0), tight_layout=True)
    ax.plot(x.squeeze(-1), y.squeeze(-1), label="target", linewidth=2.0)
    ax.plot(
        x.squeeze(-1),
        initial_prediction.squeeze(-1),
        label="initial",
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(x.squeeze(-1), final_prediction.squeeze(-1), label="trained", linewidth=1.8)
    ax.set_title("Single Spline Training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(RESULT_DIR / "single_spline_training.pdf")
    fig.savefig(RESULT_DIR / "single_spline_training.png", dpi=300)
    plt.close(fig)

    print(
        "single_spline_training "
        f"initial_loss={initial_loss.item():.6e}, "
        f"final_loss={final_loss.item():.6e}, "
        f"max_abs_error={max_abs_error.item():.6e}"
    )

    assert final_loss < initial_loss * 1e-1
