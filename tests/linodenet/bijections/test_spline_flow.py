import matplotlib.pyplot as plt
import pytest
import torch

from linodenet.bijections import SplineFlow
from tests.linodenet.bijections.fixtures import SEEDS
from tests.utils.project import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.parametrize("n_heads", [1, 4, (), (1,), (2, 3), (2, 2, 3)], ids=str)
def test_num_heads(n_heads: int | tuple[int, ...]) -> None:
    r"""Verify head-shaped inputs preserve event and logdet shapes in both directions."""
    batch_size = 8
    flow = SplineFlow(
        n_heads,
        num_flow_layers=2,
        num_bins=4,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.0, 3.0),
    )

    head_shape = (n_heads,) if isinstance(n_heads, int) else n_heads
    x = torch.randn(batch_size, *head_shape, requires_grad=True)
    y, forward_logabsdet = flow.encode_and_logabsdet(x)
    forward_loss = y.square().mean() + forward_logabsdet.square().mean()
    forward_loss.backward()

    assert y.shape == x.shape
    assert forward_logabsdet.shape == (batch_size, *head_shape[:-1])

    z = torch.randn(batch_size, *head_shape, requires_grad=True)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(z)
    inverse_loss = xhat.square().mean() + inverse_logabsdet.square().mean()
    inverse_loss.backward()

    assert xhat.shape == z.shape
    assert inverse_logabsdet.shape == (batch_size, *head_shape[:-1])


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("layers", [1, 2, 3, 4], ids="layers={}".format)
@pytest.mark.parametrize("bins", [1, 2, 4, 8], ids="bins={}".format)
def test_invertibility(seed: int, layers, bins) -> None:
    """Check encode/decode round trips within tolerances that scale with layer depth.

    The spline stack is only approximately invertible in finite precision, so the
    admissible absolute error grows slightly with the number of composed layers.
    """
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
            f"\n\tvalue     max_abs_error={forward_inverse_abs_error.max():.6e}   "
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


TEST_FNS = {
    "sinusoid": lambda x: x + 3 + 0.5 * torch.sin(x + 3),
    "small_slope": lambda x: 0.2 * x,
    "large_slope": lambda x: 5 * x,
    "offset": lambda x: x + 2,
}


@pytest.mark.parametrize("case", TEST_FNS)
@pytest.mark.parametrize("bins", [7, 8])
def test_single_spline_can_learn_monotone_function(case: str, bins: int) -> None:
    r"""Verify one spline layer can fit simple monotone targets from its initialization."""
    test_fn = TEST_FNS[case]
    torch.manual_seed(0)

    model = SplineFlow(
        num_flow_layers=1,
        num_bins=bins,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.5, 3.5),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.linspace(-4.0, 4.0, steps=256)
    y = test_fn(x)

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
        layer = model[0]
        params = layer.spline_parameters(torch.Size())
        knots = layer.spline.get_spline_parameters(
            widths=params.w,
            heights=params.h,
            lambdas=params.lambdas,
            derivatives=params.derivatives,
            x_center=layer.x_center,
            y_center=layer.y_center,
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.0), tight_layout=True)
    ax.plot(x, y, label="target", linewidth=2.0)
    ax.plot(x, final_prediction, label="trained", linewidth=2)
    ax.plot(x, initial_prediction, label="initial", lw=2, linestyle="--")
    ax.scatter(knots.x, knots.y, label="knots", s=28, zorder=3)
    dx = torch.full_like(knots.derivatives, 0.25)
    dy = knots.derivatives * dx
    ax.quiver(
        knots.x,
        knots.y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="black",
        alpha=0.8,
        zorder=4,
        label="knot derivatives",
    )
    ax.set_title("Single Spline Training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(RESULT_DIR / f"{case}-{bins=}.pdf")
    fig.savefig(RESULT_DIR / f"{case}-{bins=}.png", dpi=300)
    plt.close(fig)

    print(
        "single_spline_training "
        f"initial_loss={initial_loss.item():.6e}, "
        f"final_loss={final_loss.item():.6e}, "
        f"max_abs_error={max_abs_error.item():.6e}"
    )

    assert final_loss < initial_loss * 1e-1


def test_spline_initialization_matches_requested_linear_map() -> None:
    r"""Ensure the default initialization realizes the affine map implied by the bounds."""
    model = SplineFlow(
        num_flow_layers=1,
        num_bins=4,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-2.0, 4.0),
        use_fp64=False,
    )
    layer = model[0]
    params = layer.spline_parameters(torch.Size())
    knots = layer.spline.get_spline_parameters(
        widths=params.w,
        heights=params.h,
        lambdas=params.lambdas,
        derivatives=params.derivatives,
        x_center=layer.x_center,
        y_center=torch.zeros_like(layer.x_center),
    )

    assert torch.allclose(knots.x, torch.linspace(-3.0, 3.0, steps=5))
    assert torch.allclose(knots.y + layer.y_center, torch.linspace(-2.0, 4.0, steps=5))
    assert torch.allclose(knots.derivatives, torch.ones_like(knots.derivatives))

    x = torch.linspace(-5.0, 5.0, steps=17)
    y, forward_logabsdet = model.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = model.decode_and_logabsdet(y)

    expected = x + 1.0
    assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        forward_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(xhat, x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
    )


def test_spline_centers_shift_effective_support() -> None:
    r"""Confirm center offsets move the spline support while preserving linear tails."""
    model = SplineFlow(
        num_flow_layers=1,
        num_bins=4,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-2.0, 4.0),
        use_fp64=False,
    )
    layer = model[0]

    with torch.no_grad():
        layer.x_center.fill_(1.5)
        layer.y_center.fill_(1.25)

    params = layer.spline_parameters(torch.Size())
    knots = layer.spline.get_spline_parameters(
        widths=params.w,
        heights=params.h,
        lambdas=params.lambdas,
        derivatives=params.derivatives,
        x_center=layer.x_center,
        y_center=torch.zeros_like(layer.x_center),
    )

    x = torch.tensor([-3.0, -2.0, -1.5, 4.5, 5.0, 6.0])
    y, forward_logabsdet = model.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = model.decode_and_logabsdet(y)

    shifted_knots = knots.y + layer.y_center
    left_expected = shifted_knots[0] + knots.derivatives[0] * (x[:2] - knots.x[0])
    right_expected = shifted_knots[-1] + knots.derivatives[-1] * (x[-2:] - knots.x[-1])
    center_idx = len(knots.x) // 2

    assert knots.x[center_idx].detach() == pytest.approx(layer.x_center.item())
    assert shifted_knots[center_idx].detach() == pytest.approx(layer.y_center.item())
    assert torch.allclose(y[:2], left_expected)
    assert torch.allclose(y[-2:], right_expected)
    assert torch.allclose(xhat, x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
    )
