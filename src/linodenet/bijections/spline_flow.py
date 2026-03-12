"""Rational linear spline transforms and utilities.

Defines spline parameter structures and flow modules.
"""

__all__ = [
    # Classes
    "BinKnots",
    "SplineCoefficients",
    # Models
    "StaticLinearRationalSpline",
    "StaticUnconstrainedLRS",
    "LRS",
    "SplineFlow",
]


from collections.abc import Iterable
from typing import Final, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet.bijections.base import TransformBase, TransformSequence

DEFAULT_MIN_BIN_WIDTH: Final[float] = 1e-3
DEFAULT_MIN_BIN_HEIGHT: Final[float] = 1e-3
DEFAULT_MIN_DERIVATIVE: Final[float] = 1e-3


class BinWidths(NamedTuple):
    r"""Bin parameters (widths/heights) that specify a rational linear spline."""

    # widths and heights of the bins as well as derivatives and lambda-parameters
    w: Tensor  # (..., K)
    h: Tensor  # (..., K)
    lambdas: Tensor  # (..., K)
    derivatives: Tensor  # (..., K+1)

    def to_coefficients(self) -> SplineCoefficients:
        knots = BinKnots.from_widths(self)
        return SplineCoefficients.from_knots(knots)

    def to_knots(self) -> BinKnots:
        return BinKnots.from_widths(self)

    @staticmethod
    def from_knots(bins: BinKnots) -> BinWidths:
        x = bins.x
        y = bins.y
        lambdas = bins.lambdas
        derivatives = bins.derivatives
        widths = x.diff(dim=-1)  # (..., K)
        heights = y.diff(dim=-1)  # (..., K)

        return BinWidths(w=widths, h=heights, lambdas=lambdas, derivatives=derivatives)


class BinKnots(NamedTuple):
    r"""Bin parameters (knot x/y) that specify a rational linear spline."""

    # position of knots as well as derivatives and lambda-parameters
    x: Tensor  # (..., K+1)
    y: Tensor  # (..., K+1)
    lambdas: Tensor  # (..., K)
    derivatives: Tensor  # (..., K+1)

    def to_coefficients(self) -> SplineCoefficients:
        return SplineCoefficients.from_knots(self)

    def to_widths(self) -> BinWidths:
        return BinWidths.from_knots(self)

    @staticmethod
    def from_widths(
        bins: BinWidths,
        *,  # optional arguments
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ) -> BinKnots:
        """Determine the spline parameters from the raw inputs.

        Note:
            Instead of x and y, we expect widths and height, which are rescaled to
            the interval [LEFT, RIGHT] and [BOTTOM, TOP] respectively.

        Note:
            SplineBinWidths:
                w: The raw widths of the bins. (w∈∆ᴷ⁻¹)
                h: The raw heights of the bins. (h∈∆ᴷ⁻¹)
                λ: The raw lambdas of the bins. (λ∈(0,1)ᴷ⁻¹)
                d: The raw derivatives of the knots. (d>0)
        """
        widths = bins.w
        heights = bins.h
        derivatives = bins.derivatives
        lambdas = bins.lambdas

        num_bins = widths.shape[-1]
        one = torch.tensor(1.0, device=widths.device)
        assert min_bin_width * num_bins < 1.0, "bin width too small"
        assert min_bin_height * num_bins < 1.0, "bin height too small"
        assert (widths > 0.0).all() & widths.sum(-1).isclose(one).all()
        assert (heights > 0.0).all() & heights.sum(-1).isclose(one).all()
        assert ((lambdas > 0.0) & (lambdas < 1.0)).all()
        assert (derivatives > 0.0).all()

        # ensure >= MIN_BIN_WIDTH, by scaling down to 1-nε, and adding ε.
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

        # scale cumulative widths to the interval [LEFT, RIGHT]
        cumwidths = widths.cumsum(dim=-1).clip(0.0, 1.0)  # note: last value == 1.0
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        # get the actual knots in [LEFT, RIGHT]
        x = (right - left) * cumwidths + left  # (..., K)

        # calculate heights
        # ensure >= MIN_BIN_HEIGHT, by scaling down to 1-nε, and adding ε.
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        # scale cumulative heights to the interval [BOTTOM, TOP]
        cumheights = heights.cumsum(dim=-1).clip(0.0, 1.0)  # note: last value == 1.0
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        y = (top - bottom) * cumheights + bottom  # (..., K)

        # calculate lambdas and derivatives
        derivatives = derivatives.clip(min_derivative)

        return BinKnots(x=x, y=y, lambdas=lambdas, derivatives=derivatives)


class SplineCoefficients(NamedTuple):
    """Tuple of coefficients for a rational linear spline."""

    lam: Tensor  # (..., K)
    wa: Tensor  # (..., K)
    wb: Tensor  # (..., K)
    wc: Tensor  # (..., K)
    ya: Tensor  # (..., K)
    yb: Tensor  # (..., K)
    yc: Tensor  # (..., K)
    xa: Tensor  # (..., K)
    xb: Tensor  # (..., K)
    xc: Tensor  # (..., K)

    @staticmethod
    def from_knots(knots: BinKnots) -> SplineCoefficients:
        r"""Get the spline coefficients for the given knots shape: (..., K)."""
        x = knots.x  # (..., K)
        y = knots.y  # (..., K)
        λ = knots.lambdas  # (..., K)
        d = knots.derivatives  # (..., K)
        widths = x.diff(dim=-1)  # (..., K)
        heights = y.diff(dim=-1)  # (..., K)
        deltas = heights / widths  # (..., K)
        d_now = d[..., :-1]  # (..., K)
        d_next = d[..., 1:]  # (..., K)

        wa = torch.ones_like(d_now)
        wb = torch.sqrt(d_now / d_next) * wa
        wc = ((1 - λ) * wb * d_next + λ * wa * d_now) / deltas

        ya = y[..., :-1]  # (..., K)
        yb = y[..., 1:]  # (..., K)
        yc = ((1 - λ) * wa * ya + λ * wb * yb) / ((1 - λ) * wa + λ * wb)

        xa = x[..., :-1]  # (..., K)
        xb = x[..., 1:]
        xc = (1 - λ) * xa + λ * xb

        return SplineCoefficients(λ, wa, wb, wc, ya, yb, yc, xa, xb, xc)

    @staticmethod
    def from_selected_knots(knots: BinKnots, bin_idx: Tensor) -> SplineCoefficients:
        r"""Get the spline coefficients for the selected bins.

        Args:
            knots: The bin parameters (shape: (..., K)).
            bin_idx: The selected bin indices (shape: (...) with entries in {0, ..., K-1}).

        Returns:
            SplineCoefficients: The coefficients for the selected bins (shape: (...)).
        """
        # bin_idx: LongTensor with entries 0...K-1
        xa = knots.x.gather(-1, bin_idx).squeeze(-1)  # (...)
        xb = knots.x.gather(-1, bin_idx + 1).squeeze(-1)  # (...)

        ya = knots.y.gather(-1, bin_idx).squeeze(-1)  # (...)
        yb = knots.y.gather(-1, bin_idx + 1).squeeze(-1)  # (...)

        da = knots.derivatives.gather(-1, bin_idx).squeeze(-1)  # (...)
        db = knots.derivatives.gather(-1, bin_idx + 1).squeeze(-1)  # (...)

        λ = knots.lambdas.gather(-1, bin_idx).squeeze(-1)  # (...)

        wa = torch.ones_like(da)
        wb = torch.sqrt(da / db) * wa
        wc = ((1 - λ) * wb * db + λ * wa * da) * (xb - xa) / (yb - ya)
        xc = (1 - λ) * xa + λ * xb
        # TODO: check, it was yc = (1 - λ) * ya + λ * yb before.
        yc = ((1 - λ) * wa * ya + λ * wb * yb) / ((1 - λ) * wa + λ * wb)

        return SplineCoefficients(λ, wa, wb, wc, ya, yb, yc, xa, xb, xc)


class StaticLinearRationalSpline(nn.Module):
    r"""Non-trainable Linear Rational Spline."""

    # BUFFERS
    LEFT: Tensor
    RIGHT: Tensor
    BOTTOM: Tensor
    TOP: Tensor
    WIDTH: Tensor
    HEIGHT: Tensor
    MIN_BIN_WIDTH: Tensor
    MIN_BIN_HEIGHT: Tensor
    MIN_DERIVATIVE: Tensor

    ONE: Tensor

    def __init__(
        self,
        *,
        left: float | Tensor = 0.0,
        right: float | Tensor = 1.0,
        bottom: float | Tensor = 0.0,
        top: float | Tensor = 1.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ) -> None:
        super().__init__()
        self.register_buffer("ONE", torch.tensor(1.0))
        self.register_buffer("LEFT", torch.as_tensor(left))
        self.register_buffer("RIGHT", torch.as_tensor(right))
        self.register_buffer("BOTTOM", torch.as_tensor(bottom))
        self.register_buffer("TOP", torch.as_tensor(top))
        self.register_buffer("WIDTH", self.RIGHT - self.LEFT)
        self.register_buffer("HEIGHT", self.TOP - self.BOTTOM)
        self.register_buffer("MIN_DERIVATIVE", torch.tensor(float(min_derivative)))
        self.register_buffer("MIN_BIN_WIDTH", torch.tensor(float(min_bin_width)))
        self.register_buffer("MIN_BIN_HEIGHT", torch.tensor(float(min_bin_height)))
        assert (self.LEFT < self.RIGHT).all()
        assert (self.BOTTOM < self.TOP).all()

    def get_spline_parameters(
        self,
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K)
    ) -> BinKnots:
        r"""Determine the spline parameters from the raw inputs.

        Note:
            Instead of x and y, we expect widths and height, which are rescaled to
            the interval [LEFT, RIGHT] and [BOTTOM, TOP] respectively.

        Args:
            widths: The raw widths of the bins. (w∈∆ᴷ⁻¹)
            heights: The raw heights of the bins. (h∈∆ᴷ⁻¹)
            lambdas: The raw lambdas of the bins. (λ∈(0,1)ᴷ⁻¹)
            derivatives: The raw derivatives of the knots. (d>0)
        """
        num_bins = widths.shape[-1]
        assert self.MIN_BIN_WIDTH * num_bins < 1.0, "bin width too small"
        assert self.MIN_BIN_HEIGHT * num_bins < 1.0, "bin height too small"
        assert (widths > 0.0).all()
        assert widths.sum(-1).isclose(self.ONE).all()
        assert (heights > 0.0).all()
        assert heights.sum(-1).isclose(self.ONE).all()
        assert (lambdas > 0.0).all()
        assert (lambdas < 1.0).all()
        assert (derivatives > 0.0).all()

        # ensure >= MIN_BIN_WIDTH, by scaling down to 1-nε, and adding ε.
        widths = self.MIN_BIN_WIDTH + (1 - self.MIN_BIN_WIDTH * num_bins) * widths

        # scale cumulative widths to the interval [LEFT, RIGHT]
        cumwidths = widths.cumsum(dim=-1).clip(0.0, 1.0)  # note: last value == 1.0
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        # get the actual knots in [LEFT, RIGHT]
        x = self.WIDTH * cumwidths + self.LEFT  # (..., K)

        # calculate heights
        # ensure >= MIN_BIN_HEIGHT, by scaling down to 1-nε, and adding ε.
        heights = self.MIN_BIN_HEIGHT + (1 - self.MIN_BIN_HEIGHT * num_bins) * heights
        # scale cumulative heights to the interval [BOTTOM, TOP]
        cumheights = heights.cumsum(dim=-1).clip(0.0, 1.0)  # note: last value == 1.0
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        y = self.HEIGHT * cumheights + self.BOTTOM  # (..., K)

        # calculate lambdas and derivatives
        derivatives = derivatives.clip(self.MIN_DERIVATIVE)

        return BinKnots(x=x, y=y, lambdas=lambdas, derivatives=derivatives)

    def encode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        spline_params: BinKnots = self.get_spline_parameters(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
        )
        # select the bins
        num_bins = widths.shape[-1]
        bin_mask = inputs.unsqueeze(-1) >= spline_params.x
        # NOTE: subtract 1 to get the correct bin index, clip to avoid out-of-bounds
        bin_idx = (bin_mask.sum(dim=-1, keepdim=True) - 1).clip(0, num_bins - 1)

        # get the parameters/coefficients for the selected bins
        coef = SplineCoefficients.from_selected_knots(spline_params, bin_idx)
        lam, wa, wb, wc, ya, yb, yc, xa, xb, _ = coef

        # calculate return values
        phi = (inputs - xa) / (xb - xa)
        numerator = torch.where(
            phi <= lam,
            (lam - phi) * wa * ya + phi * wc * yc,
            (1 - phi) * wc * yc + (phi - lam) * wb * yb,
        )
        denominator = torch.where(
            phi <= lam,
            (lam - phi) * wa + phi * wc,
            (1 - phi) * wc + (phi - lam) * wb,
        )
        outputs = numerator / denominator

        derivative_numerator = torch.where(
            phi <= lam,
            lam * wa * wc * (yc - ya),
            (1 - lam)  * wb * wc* (yb - yc),
        ) / (xb - xa)  # fmt: skip
        logabsdet = derivative_numerator.log() - 2 * denominator.abs().log()

        return outputs, logabsdet  # (...), (...)

    def decode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        spline_params: BinKnots = self.get_spline_parameters(
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            lambdas=lambdas,
        )
        # select the bins
        bin_mask = inputs.unsqueeze(-1) >= spline_params.y  # 0...K
        # NOTE: subtract 1 to get the correct bin index, clip to avoid out-of-bounds
        num_knots = heights.shape[-1]
        bin_idx = (bin_mask.sum(dim=-1, keepdim=True) - 1).clip(0, num_knots - 1)

        # get the parameters/coefficients for the selected bins
        coef = SplineCoefficients.from_selected_knots(spline_params, bin_idx)
        lam, wa, wb, wc, ya, yb, yc, xa, xb, _ = coef

        # calculate return values
        numerator = torch.where(  # (...)
            inputs <= yc,
            lam * wa * (ya - inputs),
            lam * wb * (yb - inputs) + wc * (inputs - yc),
        )
        denominator = torch.where(  # (...)
            inputs <= yc,
            (wc - wa) * inputs + wa * ya - wc * yc,
            (wc - wb) * inputs + wb * yb - wc * yc,
        )
        outputs = (xb - xa) * (numerator / denominator) + xa

        derivative_numerator = (xb - xa) * torch.where(  # (...)
            inputs <= yc,
            lam * wa * wc * (yc - ya),
            (1 - lam) * wb * wc * (yb - yc),
        )

        logabsdet = derivative_numerator.log() - 2 * denominator.abs().log()

        return outputs, logabsdet  # (...), (...)


class StaticUnconstrainedLRS(StaticLinearRationalSpline):
    r"""Non-trainable unconstrained LRS with linear tails."""

    def encode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        """Identity mapping for inputs outside the interval."""
        mask = (inputs >= self.LEFT) & (inputs <= self.RIGHT)

        outputs, logabsdet = super().encode_and_logabsdet(
            inputs,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            lambdas=lambdas,
        )
        outputs = torch.where(mask, outputs, inputs)
        logabsdet = torch.where(mask, logabsdet, 0.0)

        return outputs, logabsdet

    def decode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        """Identity mapping for inputs outside the interval."""
        mask = (inputs >= self.BOTTOM) & (inputs <= self.TOP)

        outputs, logabsdet = super().decode_and_logabsdet(
            inputs,
            widths=widths,
            heights=heights,
            derivatives=derivatives,
            lambdas=lambdas,
        )
        outputs = torch.where(mask, outputs, inputs)
        logabsdet = torch.where(mask, logabsdet, 0.0)

        return outputs, logabsdet


class LRS(TransformBase):
    r"""Trainable Linear Rational Spline."""

    n_heads: Final[torch.Size]  # tuple[*H]
    r"""Number of mixture components."""
    num_bins: Final[int]
    r"""Number of rational linear spline components."""

    # Parameters
    widths: Tensor  # (*H, K)
    heights: Tensor  # (*H, K)
    lambdas: Tensor  # (*H, K)
    derivatives: Tensor  # (*H, K+1)

    spline: StaticUnconstrainedLRS

    def __init__(
        self,
        n_heads: int | tuple[int, ...],
        *,
        num_bins: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        super().__init__()
        #  Constants
        self.num_bins = int(num_bins)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.n_heads = torch.Size((n_heads,) if isinstance(n_heads, int) else n_heads)
        # Parameters
        self.widths = nn.Parameter(torch.randn(*self.n_heads, num_bins))
        self.heights = nn.Parameter(torch.randn(*self.n_heads, num_bins))
        self.lambdas = nn.Parameter(torch.randn(*self.n_heads, num_bins))
        self.derivatives = nn.Parameter(torch.randn(*self.n_heads, num_bins + 1))
        # Submodules
        left, right = x_bounds
        bottom, top = y_bounds
        assert left < right
        assert bottom < top
        self.spline = StaticUnconstrainedLRS(
            left=left, right=right, bottom=bottom, top=top
        )

    def _normalized_parameters(self, batch_shape: torch.Size, /) -> BinWidths:
        r"""Expand normalized spline parameters to match the batch shape."""
        widths = torch.softmax(self.widths, dim=-1)
        heights = torch.softmax(self.heights, dim=-1)
        lambdas = torch.sigmoid(self.lambdas)
        derivatives = F.softplus(self.derivatives)

        return BinWidths(
            w=widths.expand(*batch_shape, *widths.shape),
            h=heights.expand(*batch_shape, *heights.shape),
            lambdas=lambdas.expand(*batch_shape, *lambdas.shape),
            derivatives=derivatives.expand(*batch_shape, *derivatives.shape),
        )

    @torch.no_grad()
    def marginalize(self, kept: list[int] | Tensor) -> LRS:
        """Marginalize out the specified variables.

        Assumes that n_heads = (*heads, dims),
        and that the last dimension corresponds to the features.
        """
        device = self.widths.device
        kept = torch.as_tensor(kept, device=device)
        if kept.dtype == torch.bool:
            assert kept.shape == (self.n_heads[-1],)
            num_kept = int(kept.sum().item())
            kept = kept.nonzero(as_tuple=False).squeeze(-1)
        else:
            assert kept.min() >= 0
            assert kept.max() < self.n_heads[-1]
            num_kept = len(kept)

        new_heads = (*self.n_heads[:-1], num_kept)
        new = LRS(
            n_heads=new_heads,
            num_bins=self.num_bins,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
        ).to(device=device)

        # remove the specified variables from the parameters
        marg_widths = self.widths.index_select(dim=-2, index=kept)
        marg_heights = self.heights.index_select(dim=-2, index=kept)
        marg_lambdas = self.lambdas.index_select(dim=-2, index=kept)
        marg_derivatives = self.derivatives.index_select(dim=-2, index=kept)

        new.widths.copy_(marg_widths)
        new.heights.copy_(marg_heights)
        new.lambdas.copy_(marg_lambdas)
        new.derivatives.copy_(marg_derivatives)
        return new

    def encode(self, z: Tensor) -> Tensor:

        z, _ = self.encode_and_logabsdet(z)
        return z

    def decode(self, z: Tensor, /) -> Tensor:
        z, _ = self.decode_and_logabsdet(z)
        return z

    def encode_and_logabsdet(self, z: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Forward pass of the flow.

        Args:
            z (..., *H, D): input tensor

        Returns:
            z (..., *H, D): transformed tensor
            ldj (..., *H): log determinant of the Jacobian
        """
        batch_shape = z.shape[: -len(self.n_heads)]
        params = self._normalized_parameters(batch_shape)
        z, logabsdet = self.spline.encode_and_logabsdet(
            z,
            widths=params.w,
            heights=params.h,
            lambdas=params.lambdas,
            derivatives=params.derivatives,
        )
        return z, logabsdet.sum(dim=-1)

    def decode_and_logabsdet(self, z: Tensor) -> tuple[Tensor, Tensor]:
        r"""Inverse pass of the flow.

        Args:
            z (..., *H, D): input tensor

        Returns:
            z (..., *H, D): transformed tensor
            ldj (..., *H): log determinant of the Jacobian
        """
        batch_shape = z.shape[: -len(self.n_heads)]
        params = self._normalized_parameters(batch_shape)
        z, logabsdet = self.spline.decode_and_logabsdet(
            z,
            widths=params.w,
            heights=params.h,
            lambdas=params.lambdas,
            derivatives=params.derivatives,
        )
        return z, logabsdet.sum(dim=-1)


class SplineFlow(TransformSequence[LRS]):
    r"""Implements a sequence of rational linear spline layers."""

    @classmethod
    def from_iterable(cls, layers: Iterable[LRS], /) -> SplineFlow:
        """Create a SplineFlow from an iterable of LRS layers."""
        new = SplineFlow.__new__(SplineFlow)
        super(SplineFlow, new).__init__(layers)
        return new

    def __init__(
        self,
        n_heads: int | tuple[int, ...],
        *,
        num_flow_layers: int,
        num_bins: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        layers = [
            LRS(n_heads, num_bins=num_bins, x_bounds=x_bounds, y_bounds=y_bounds)
            for _ in range(num_flow_layers)
        ]
        super().__init__(layers)

    def marginalize(self, variables: list[int] | Tensor) -> SplineFlow:
        """Marginalize out the specified variables."""
        return SplineFlow.from_iterable(layer.marginalize(variables) for layer in self)
