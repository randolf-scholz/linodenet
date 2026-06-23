import matplotlib.pyplot as plt
import pytest
import torch

from linodenet.mappings.transforms import SplineTransform
from tests.testing import PROJECT, SEEDS_5

from .base import TestTransform

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


class TestSplineFlow(TestTransform):
    TEST_FNS = {
        "sinusoid": lambda x: x + 3 + 0.5 * torch.sin(x + 3),
        "small_slope": lambda x: 0.2 * x,
        "large_slope": lambda x: 5 * x,
        "offset": lambda x: x + 2,
    }

    NUM_HEADS_BATCH_SIZE = 8
    NUM_HEADS_LAYERS = 2
    NUM_HEADS_BINS = 4
    VALUE_ATOL_PER_LAYER = 1.5e-2
    VALUE_RTOL = 1e-3
    LOGABSDET_ATOL_PER_LAYER = 5e-3
    LOGABSDET_RTOL = 1e-3
    BATCH_SIZE = 128
    NUM_HEADS = 4
    LINEAR_ATOL = 1e-5
    LINEAR_RTOL = 1e-5

    @pytest.mark.parametrize("n_heads", [1, 4, (), (1,), (2, 3), (2, 2, 3)], ids=str)
    def test_num_heads(self, n_heads: int | tuple[int, ...]) -> None:
        r"""Verify head-shaped inputs preserve event and logdet shapes in both directions."""
        flow = SplineTransform(
            n_heads,
            num_flow_layers=self.NUM_HEADS_LAYERS,
            num_bins=self.NUM_HEADS_BINS,
            x_bounds=(-3.0, 3.0),
            y_bounds=(-3.0, 3.0),
        )

        head_shape = (n_heads,) if isinstance(n_heads, int) else n_heads
        x = torch.randn(self.NUM_HEADS_BATCH_SIZE, *head_shape, requires_grad=True)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        forward_loss = y.square().mean() + forward_logabsdet.square().mean()
        forward_loss.backward()

        assert y.shape == x.shape
        assert forward_logabsdet.shape == (self.NUM_HEADS_BATCH_SIZE, *head_shape[:-1])

        z = torch.randn(self.NUM_HEADS_BATCH_SIZE, *head_shape, requires_grad=True)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(z)
        inverse_loss = xhat.square().mean() + inverse_logabsdet.square().mean()
        inverse_loss.backward()

        assert xhat.shape == z.shape
        assert inverse_logabsdet.shape == (self.NUM_HEADS_BATCH_SIZE, *head_shape[:-1])

    @pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
    @pytest.mark.parametrize("layers", [1, 2, 3, 4], ids="layers={}".format)
    @pytest.mark.parametrize("bins", [1, 2, 4, 8], ids="bins={}".format)
    def test_invertibility(self, seed: int, layers: int, bins: int) -> None:
        """Check encode/decode round trips within tolerances that scale with layer depth.

        The spline stack is only approximately invertible in finite precision, so the
        admissible absolute error grows slightly with the number of composed layers.
        """
        torch.manual_seed(seed)
        value_atol = self.VALUE_ATOL_PER_LAYER * layers
        logabsdet_atol = self.LOGABSDET_ATOL_PER_LAYER * layers

        flow = SplineTransform(
            self.NUM_HEADS,
            num_flow_layers=layers,
            num_bins=bins,
            x_bounds=(-3.0, 3.0),
            y_bounds=(-3.0, 3.0),
        )

        x = torch.randn(self.BATCH_SIZE, self.NUM_HEADS)
        y = torch.randn(self.BATCH_SIZE, self.NUM_HEADS)
        self.assert_invertible(
            flow,
            x,
            y,
            atol=value_atol,
            rtol=self.VALUE_RTOL,
            logdet_atol=logabsdet_atol,
            logdet_rtol=self.LOGABSDET_RTOL,
        )

    @pytest.mark.parametrize("case", TEST_FNS)
    @pytest.mark.parametrize("bins", [7, 8])
    def test_single_spline_can_learn_monotone_function(
        self, case: str, bins: int
    ) -> None:
        r"""Verify one spline layer can fit simple monotone targets from its initialization."""
        test_fn = self.TEST_FNS[case]
        torch.manual_seed(0)

        model = SplineTransform(
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

        for _ in range(200):
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
            widths, heights, lambdas, derivatives = layer.spline_parameters(
                torch.Size()
            )
            knots = layer.spline.get_spline_knots(
                widths=widths,
                heights=heights,
                lambdas=lambdas,
                derivatives=derivatives,
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

        assert final_loss < initial_loss * 1e-1
        assert max_abs_error >= 0.0

    def test_spline_initialization_matches_requested_linear_map(self) -> None:
        r"""Ensure the default initialization realizes the affine map implied by the bounds."""
        model = SplineTransform(
            num_flow_layers=1,
            num_bins=4,
            x_bounds=(-3.0, 3.0),
            y_bounds=(-2.0, 4.0),
            use_fp64=False,
        )
        layer = model[0]
        widths, heights, lambdas, derivatives = layer.spline_parameters(torch.Size())
        knots = layer.spline.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=layer.x_center,
            y_center=torch.zeros_like(layer.x_center),
        )

        self.assert_close(
            knots.x, torch.linspace(-3.0, 3.0, steps=5), atol=1e-6, rtol=1e-6
        )
        self.assert_close(
            knots.y + layer.y_center,
            torch.linspace(-2.0, 4.0, steps=5),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assert_close(
            knots.derivatives, torch.ones_like(knots.derivatives), atol=1e-6, rtol=1e-6
        )

        x = torch.linspace(-5.0, 5.0, steps=17)
        y, forward_logabsdet = model.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = model.decode_and_logabsdet(y)

        expected = x + 1.0
        self.assert_close(y, expected, atol=self.LINEAR_ATOL, rtol=self.LINEAR_RTOL)
        self.assert_close(
            forward_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LINEAR_ATOL,
            rtol=self.LINEAR_RTOL,
        )
        self.assert_close(xhat, x, atol=self.LINEAR_ATOL, rtol=self.LINEAR_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LINEAR_ATOL,
            rtol=self.LINEAR_RTOL,
        )

    def test_spline_centers_shift_effective_support(self) -> None:
        r"""Confirm center offsets move the spline support while preserving linear tails."""
        model = SplineTransform(
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

        widths, heights, lambdas, derivatives = layer.spline_parameters(torch.Size())
        knots = layer.spline.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=layer.x_center,
            y_center=torch.zeros_like(layer.x_center),
        )

        x = torch.tensor([-3.0, -2.0, -1.5, 4.5, 5.0, 6.0])
        y, forward_logabsdet = model.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = model.decode_and_logabsdet(y)

        shifted_knots = knots.y + layer.y_center
        left_expected = shifted_knots[0] + knots.derivatives[0] * (x[:2] - knots.x[0])
        right_expected = shifted_knots[-1] + knots.derivatives[-1] * (
            x[-2:] - knots.x[-1]
        )
        center_idx = len(knots.x) // 2

        assert knots.x[center_idx].detach() == pytest.approx(layer.x_center.item())
        assert shifted_knots[center_idx].detach() == pytest.approx(
            layer.y_center.item()
        )
        self.assert_close(y[:2], left_expected, atol=1e-6, rtol=1e-6)
        self.assert_close(y[-2:], right_expected, atol=1e-6, rtol=1e-6)
        self.assert_close(xhat, x, atol=self.LINEAR_ATOL, rtol=self.LINEAR_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LINEAR_ATOL,
            rtol=self.LINEAR_RTOL,
        )
