r"""Separable flows."""

__all__ = [
    # constants
    "DEFAULT_MIN_BIN_HEIGHT",
    "DEFAULT_MIN_BIN_WIDTH",
    "DEFAULT_MIN_DERIVATIVE",
    # classes
    "MarginalizableNormalizingFlow",
    "Moses",
    "BinKnots",
    "LearnableLRS",
    "LinearRationalSpline",
    "MixtureWeightsModel",
    "ModuleSequence",
    "MultiHeadGaussian",
    "SplineCoefficients",
    "ConditionalSplineFlow",
    "SplineFlow",
    "UnconstrainedLinearRationalSpline",
    "PositionalEmbedding",
    "SeparableEncoder",
    "MultiHeadAttention",
    "ChannelEmbedding",
    # functions
    "inverse_softplus",
]


import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final, NamedTuple, Optional, overload

import torch
from torch import Tensor, nan, nn
from torch.linalg import cholesky, solve_triangular
from torch.nn import functional as F

from .grafiti import Grafiti
from .utils import EventBatch

DEFAULT_MIN_BIN_WIDTH: Final[float] = 1e-3
DEFAULT_MIN_BIN_HEIGHT: Final[float] = 1e-3
DEFAULT_MIN_DERIVATIVE: Final[float] = 1e-3
_LOG2PI: Final[float] = math.log(2.0 * math.pi)


class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    if TYPE_CHECKING:
        _modules: Mapping[str, M]  # type: ignore[override]

        # noinspection PyMissingConstructor
        def __init__(self, _: Iterable[M] = (), /) -> None: ...
        def __iter__(self) -> Iterator[M]: ...

    @overload
    def __getitem__(self, index: int, /) -> M: ...  # pyrefly: ignore[bad-override]
    @overload
    def __getitem__(self, index: slice, /) -> ModuleSequence[M]: ...
    def __getitem__(self, index: int | slice, /) -> M | ModuleSequence[M]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, slice):
            modules = list(self._modules.values())
            selection = modules[index]
            return ModuleSequence(selection)
        return self._modules[self._get_abs_string_index(index)]


def _centered_cumulative_knots(widths: Tensor, center: Tensor, /) -> Tensor:
    r"""Convert positive increments to centered knot coordinates."""
    cumwidths = torch.cat(
        [
            widths.new_zeros(*widths.shape[:-1], 1),
            widths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    num_knots = cumwidths.shape[-1]
    if num_knots % 2 == 1:
        center_idx = num_knots // 2
        center_offset = cumwidths[..., center_idx : center_idx + 1]
    else:
        right_center_idx = num_knots // 2
        left_center_idx = right_center_idx - 1
        center_offset = 0.5 * (
            cumwidths[..., left_center_idx : left_center_idx + 1]
            + cumwidths[..., right_center_idx : right_center_idx + 1]
        )

    return cumwidths - center_offset + center.unsqueeze(-1)


class BinKnots(NamedTuple):
    r"""Knot parameters that specify a rational linear spline."""

    # position of knots as well as derivatives and lambda-parameters
    x: Tensor  # (..., K+1)
    y: Tensor  # (..., K+1)
    lambdas: Tensor  # (..., K)
    derivatives: Tensor  # (..., K+1)

    def to(self, dtype: torch.dtype | None, device: torch.device | None) -> BinKnots:
        return BinKnots(
            x=self.x.to(dtype=dtype, device=device),
            y=self.y.to(dtype=dtype, device=device),
            lambdas=self.lambdas.to(dtype=dtype, device=device),
            derivatives=self.derivatives.to(dtype=dtype, device=device),
        )

    def to_coefficients(self) -> SplineCoefficients:
        return SplineCoefficients.from_knots(self)


class SplineCoefficients(NamedTuple):
    r"""Tuple of coefficients for a rational linear spline."""

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
    def from_selected_knots(knots: BinKnots, bin_idx: Tensor, /) -> SplineCoefficients:
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
        yc = ((1 - λ) * wa * ya + λ * wb * yb) / ((1 - λ) * wa + λ * wb)

        return SplineCoefficients(λ, wa, wb, wc, ya, yb, yc, xa, xb, xc)


# (...), (..., K+1) -> (...), (...)
def _lrs_encode(inputs: Tensor, knots: BinKnots) -> tuple[Tensor, Tensor]:
    r"""Evaluate the bounded LRS forward map and log|det J| at inputs."""
    num_bins = knots.x.shape[-1] - 1
    bin_idx = torch.searchsorted(
        knots.x[..., 1:-1], inputs[..., None], side="right"
    ).clip(0, num_bins - 1)

    lam, wa, wb, wc, ya, yb, yc, xa, xb, _ = SplineCoefficients.from_selected_knots(
        knots, bin_idx
    )

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
    derivative_numerator = torch.where(
        phi <= lam,
        lam * wa * wc * (yc - ya),
        (1 - lam) * wb * wc * (yb - yc),
    ) / (xb - xa)  # fmt: skip
    logabsdet = derivative_numerator.log() - 2 * denominator.abs().log()
    return numerator / denominator, logabsdet


# (...), (..., K+1) -> (...), (...)
def _lrs_decode(inputs: Tensor, knots: BinKnots) -> tuple[Tensor, Tensor]:
    r"""Evaluate the bounded LRS inverse map and log|det J| at inputs."""
    num_bins = knots.y.shape[-1] - 1
    bin_idx = torch.searchsorted(
        knots.y[..., 1:-1], inputs[..., None], side="right"
    ).clip(0, num_bins - 1)

    lam, wa, wb, wc, ya, yb, yc, xa, xb, _ = SplineCoefficients.from_selected_knots(
        knots, bin_idx
    )

    numerator = torch.where(
        inputs <= yc,
        lam * wa * (ya - inputs),
        lam * wb * (yb - inputs) + wc * (inputs - yc),
    )
    denominator = torch.where(
        inputs <= yc,
        (wc - wa) * inputs + wa * ya - wc * yc,
        (wc - wb) * inputs + wb * yb - wc * yc,
    )
    derivative_numerator = (xb - xa) * torch.where(
        inputs <= yc,
        lam * wa * wc * (yc - ya),
        (1 - lam) * wb * wc * (yb - yc),
    )
    logabsdet = derivative_numerator.log() - 2 * denominator.abs().log()
    return (xb - xa) * (numerator / denominator) + xa, logabsdet


class LinearRationalSpline(nn.Module):
    r"""Non-trainable Linear Rational Spline."""

    # BUFFERS
    MIN_BIN_WIDTH: Tensor
    MIN_BIN_HEIGHT: Tensor
    MIN_DERIVATIVE: Tensor

    use_fp64: Final[bool]

    def __init__(
        self,
        *,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
        use_fp64: bool = True,
    ) -> None:
        super().__init__()
        self.use_fp64 = use_fp64
        dtype = torch.float64 if use_fp64 else torch.float32
        self.register_buffer(
            "MIN_DERIVATIVE", torch.tensor(float(min_derivative), dtype=dtype)
        )
        self.register_buffer(
            "MIN_BIN_WIDTH", torch.tensor(float(min_bin_width), dtype=dtype)
        )
        self.register_buffer(
            "MIN_BIN_HEIGHT", torch.tensor(float(min_bin_height), dtype=dtype)
        )

    def get_spline_knots(
        self,
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K)
        x_center: Tensor,  # (...,)
        y_center: Tensor,  # (...,)
    ) -> BinKnots:
        r"""Determine the spline parameters from the raw inputs.

        Note:
            Instead of x and y, we expect positive widths and heights together with
            the x/y center coordinates.

        Args:
            widths: The positive widths of the bins.
            heights: The positive heights of the bins.
            lambdas: The raw lambdas of the bins. (λ∈(0,1)ᴷ⁻¹)
            derivatives: The raw derivatives of the knots. (d>0)
            x_center: The x center of the bins. (d>0)
            y_center: The y center of the bins. (d>0)
        """
        work_dtype = self.MIN_DERIVATIVE.dtype
        widths = widths.to(dtype=work_dtype)
        heights = heights.to(dtype=work_dtype)
        lambdas = lambdas.to(dtype=work_dtype)
        derivatives = derivatives.to(dtype=work_dtype)
        x_center = x_center.to(dtype=work_dtype)
        y_center = y_center.to(dtype=work_dtype)

        num_bins = widths.shape[-1]
        assert num_bins > 0
        assert (widths >= self.MIN_BIN_WIDTH).all()
        assert (heights >= self.MIN_BIN_HEIGHT).all()
        assert (lambdas > 0.0).all()
        assert (lambdas < 1.0).all()
        assert (derivatives > 0.0).all()

        x = _centered_cumulative_knots(widths, x_center)
        y = _centered_cumulative_knots(heights, y_center)

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
        original_dtype = inputs.dtype
        inputs = inputs.to(dtype=self.MIN_DERIVATIVE.dtype)
        knots = self.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=torch.zeros_like(inputs),
            y_center=torch.zeros_like(inputs),
        )
        outputs, logabsdet = _lrs_encode(inputs, knots)
        return outputs.to(dtype=original_dtype), logabsdet.to(dtype=original_dtype)

    def decode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        original_dtype = inputs.dtype
        inputs = inputs.to(dtype=self.MIN_DERIVATIVE.dtype)
        knots = self.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=torch.zeros_like(inputs),
            y_center=torch.zeros_like(inputs),
        )
        outputs, logabsdet = _lrs_decode(inputs, knots)
        return outputs.to(dtype=original_dtype), logabsdet.to(dtype=original_dtype)


class UnconstrainedLinearRationalSpline(LinearRationalSpline):
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
        r"""Use linear tails anchored at the learned spline endpoints."""
        original_dtype = inputs.dtype
        inputs = inputs.to(dtype=self.MIN_DERIVATIVE.dtype)
        knots = self.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=torch.zeros_like(inputs),
            y_center=torch.zeros_like(inputs),
        )
        outputs, logabsdet = _lrs_encode(inputs, knots)

        left_x, right_x = knots.x[..., 0], knots.x[..., -1]
        left_d, right_d = knots.derivatives[..., 0], knots.derivatives[..., -1]
        left_mask, right_mask = inputs < left_x, inputs > right_x
        outputs = torch.where(
            left_mask, knots.y[..., 0] + left_d * (inputs - left_x), outputs
        )
        outputs = torch.where(
            right_mask, knots.y[..., -1] + right_d * (inputs - right_x), outputs
        )
        logabsdet = torch.where(left_mask, left_d.log(), logabsdet)
        logabsdet = torch.where(right_mask, right_d.log(), logabsdet)

        return outputs.to(dtype=original_dtype), logabsdet.to(dtype=original_dtype)

    def decode_and_logabsdet(
        self,
        inputs: Tensor,  # (...)
        *,
        widths: Tensor,  # (..., K)
        heights: Tensor,  # (..., K)
        lambdas: Tensor,  # (..., K)
        derivatives: Tensor,  # (..., K+1)
    ) -> tuple[Tensor, Tensor]:  # (...), (...)
        r"""Invert the linear tails anchored at the learned spline endpoints."""
        original_dtype = inputs.dtype
        inputs = inputs.to(dtype=self.MIN_DERIVATIVE.dtype)
        knots = self.get_spline_knots(
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
            x_center=torch.zeros_like(inputs),
            y_center=torch.zeros_like(inputs),
        )
        outputs, logabsdet = _lrs_decode(inputs, knots)

        left_y, right_y = knots.y[..., 0], knots.y[..., -1]
        left_x, right_x = knots.x[..., 0], knots.x[..., -1]
        left_d, right_d = knots.derivatives[..., 0], knots.derivatives[..., -1]
        left_mask, right_mask = inputs < left_y, inputs > right_y
        outputs = torch.where(left_mask, left_x + (inputs - left_y) / left_d, outputs)
        outputs = torch.where(
            right_mask, right_x + (inputs - right_y) / right_d, outputs
        )
        logabsdet = torch.where(left_mask, -left_d.log(), logabsdet)
        logabsdet = torch.where(right_mask, -right_d.log(), logabsdet)

        return outputs.to(dtype=original_dtype), logabsdet.to(dtype=original_dtype)


class LearnableLRS(nn.Module):
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
    x_center: Tensor  # (*H,)
    y_center: Tensor  # (*H,)

    spline: UnconstrainedLinearRationalSpline

    def __init__(
        self,
        num_heads: int | tuple[int, ...],
        *,
        num_bins: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        use_fp64: bool = True,
    ) -> None:
        super().__init__()
        #  Constants
        self.num_bins = int(num_bins)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.n_heads = torch.Size(
            (num_heads,) if isinstance(num_heads, int) else num_heads
        )
        left, right = x_bounds
        bottom, top = y_bounds
        assert left < right
        assert bottom < top
        slope = (top - bottom) / (right - left)
        assert slope > 0.0
        width_init = (right - left) / self.num_bins
        height_init = (top - bottom) / self.num_bins
        min_bin_width = float(DEFAULT_MIN_BIN_WIDTH)
        min_bin_height = float(DEFAULT_MIN_BIN_HEIGHT)
        assert width_init > min_bin_width
        assert height_init > min_bin_height

        def _inverse_softplus(value: float, /) -> float:
            return value + math.log(-math.expm1(-value))

        # Parameters
        self.widths = nn.Parameter(
            torch.full(
                (*self.n_heads, num_bins),
                _inverse_softplus(width_init - min_bin_width),
            )
        )
        self.heights = nn.Parameter(
            torch.full(
                (*self.n_heads, num_bins),
                _inverse_softplus(height_init - min_bin_height),
            )
        )
        self.lambdas = nn.Parameter(torch.zeros(*self.n_heads, num_bins))
        self.derivatives = nn.Parameter(
            torch.full((*self.n_heads, num_bins + 1), _inverse_softplus(slope))
        )
        self.x_center = nn.Parameter(torch.full(self.n_heads, 0.5 * (left + right)))
        self.y_center = nn.Parameter(torch.full(self.n_heads, 0.5 * (bottom + top)))
        # Submodules
        self.spline = UnconstrainedLinearRationalSpline(use_fp64=use_fp64)

    def spline_parameters(
        self, batch_shape: tuple[int, ...], /
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Expand spline parameters to match the batch shape."""
        widths = self.spline.MIN_BIN_WIDTH + F.softplus(self.widths)
        heights = self.spline.MIN_BIN_HEIGHT + F.softplus(self.heights)
        lambdas = torch.sigmoid(self.lambdas)
        derivatives = F.softplus(self.derivatives)

        return (
            widths.expand(*batch_shape, *widths.shape),
            heights.expand(*batch_shape, *heights.shape),
            lambdas.expand(*batch_shape, *lambdas.shape),
            derivatives.expand(*batch_shape, *derivatives.shape),
        )

    @torch.no_grad()
    def marginalize(self, kept: list[int] | Tensor) -> LearnableLRS:
        r"""Marginalize out the specified variables.

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
        new = LearnableLRS(
            num_heads=new_heads,
            num_bins=self.num_bins,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
        ).to(device=device)

        # remove the specified variables from the parameters
        marg_widths = self.widths.index_select(dim=-2, index=kept)
        marg_heights = self.heights.index_select(dim=-2, index=kept)
        marg_lambdas = self.lambdas.index_select(dim=-2, index=kept)
        marg_derivatives = self.derivatives.index_select(dim=-2, index=kept)
        marg_x_center = self.x_center.index_select(dim=-1, index=kept)
        marg_y_center = self.y_center.index_select(dim=-1, index=kept)

        new.widths.copy_(marg_widths)
        new.heights.copy_(marg_heights)
        new.lambdas.copy_(marg_lambdas)
        new.derivatives.copy_(marg_derivatives)
        new.x_center.copy_(marg_x_center)
        new.y_center.copy_(marg_y_center)
        return new

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Forward pass of the flow.

        Args:
            x (..., *H, $K): input tensor

        Returns:
            y (..., *H, $K): transformed tensor
            ldj (..., *H): log determinant of the Jacobian
        """
        batch_shape = x.shape[: -len(self.n_heads)] if self.n_heads else x.shape
        widths, heights, lambdas, derivatives = self.spline_parameters(batch_shape)
        y, logabsdet = self.spline.encode_and_logabsdet(
            x - self.x_center,
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
        )
        y = y + self.y_center
        return y, logabsdet.sum(dim=-1) if self.n_heads else logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Inverse pass of the flow.

        Args:
            y (..., *H, $K): input tensor

        Returns:
            x (..., *H, $K): transformed tensor
            ldj (..., *H): log determinant of the Jacobian
        """
        batch_shape = y.shape[: -len(self.n_heads)] if self.n_heads else y.shape
        widths, heights, lambdas, derivatives = self.spline_parameters(batch_shape)
        y = y - self.y_center
        x, logabsdet = self.spline.decode_and_logabsdet(
            y,
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
        )
        x = x + self.x_center
        return x, logabsdet.sum(dim=-1) if self.n_heads else logabsdet


class ConditionalLRS(nn.Module):
    width_model: nn.Module
    height_model: nn.Module
    lambda_model: nn.Module
    derivative_model: nn.Module
    spline: UnconstrainedLinearRationalSpline

    # buffers
    widths: Tensor
    heights: Tensor
    lambdas: Tensor
    derivatives: Tensor
    x_center: Tensor
    y_center: Tensor

    def __init__(
        self,
        dim_context: int,
        *,
        num_bins: int,
        num_heads: int | tuple[int, ...] = (),
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        use_fp64: bool = True,
    ) -> None:
        super().__init__()
        self.head_shape = torch.Size(
            (num_heads,) if isinstance(num_heads, int) else num_heads
        )
        self.num_bins = int(num_bins)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.spline = UnconstrainedLinearRationalSpline(use_fp64=use_fp64)
        self.dim_context = dim_context
        left, right = x_bounds
        bottom, top = y_bounds
        assert left < right
        assert bottom < top
        slope = (top - bottom) / (right - left)
        assert slope > 0.0
        width_init = (right - left) / self.num_bins
        height_init = (top - bottom) / self.num_bins
        min_bin_width = float(DEFAULT_MIN_BIN_WIDTH)
        min_bin_height = float(DEFAULT_MIN_BIN_HEIGHT)
        assert width_init > min_bin_width
        assert height_init > min_bin_height

        self.width_model = nn.Linear(dim_context, num_bins)
        self.height_model = nn.Linear(dim_context, num_bins)
        self.lambda_model = nn.Linear(dim_context, num_bins)
        self.derivative_model = nn.Linear(dim_context, num_bins + 1)
        self.x_center = nn.Parameter(torch.full(self.head_shape, 0.5 * (left + right)))
        self.y_center = nn.Parameter(torch.full(self.head_shape, 0.5 * (bottom + top)))

        width_bias = inverse_softplus(
            torch.tensor(width_init - min_bin_width, dtype=self.width_model.bias.dtype)
        ).item()
        height_bias = inverse_softplus(
            torch.tensor(
                height_init - min_bin_height, dtype=self.height_model.bias.dtype
            )
        ).item()
        derivative_bias = inverse_softplus(
            torch.tensor(slope, dtype=self.derivative_model.bias.dtype)
        ).item()

        with torch.no_grad():
            for module in (
                self.width_model,
                self.height_model,
                self.lambda_model,
                self.derivative_model,
            ):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            self.width_model.bias.fill_(width_bias)
            self.height_model.bias.fill_(height_bias)
            self.derivative_model.bias.fill_(derivative_bias)

        self.register_buffer("widths", None, persistent=False)
        self.register_buffer("heights", None, persistent=False)
        self.register_buffer("lambdas", None, persistent=False)
        self.register_buffer("derivatives", None, persistent=False)

    def forward(self, context: Tensor, /) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Condition the spline parameters on the given context tensor."""
        if context.shape[-1] != self.dim_context:
            raise ValueError(
                f"context dim must be {self.dim_context}, got {context.shape[-1]}"
            )

        widths = self.spline.MIN_BIN_WIDTH + F.softplus(self.width_model(context))
        heights = self.spline.MIN_BIN_HEIGHT + F.softplus(self.height_model(context))
        lambdas = torch.sigmoid(self.lambda_model(context))
        derivatives = F.softplus(self.derivative_model(context))
        self.widths = widths.detach()
        self.heights = heights.detach()
        self.lambdas = lambdas.detach()
        self.derivatives = derivatives.detach()
        return widths, heights, lambdas, derivatives

    def encode_and_logabsdet(
        self,
        x: Tensor,  # (..., *H, $K)
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H)
        r"""Forward pass of the flow."""
        # Shape legend:
        # ...: batch_shape
        # *H: head_shape
        # $K: number of values (dynamic)
        # D: context dim (static)
        if context.shape[:-1] != x.shape:
            raise ValueError(
                f"context batch/head/value shape {context.shape[:-1]} must match x "
                f"shape {x.shape}"
            )
        widths, heights, lambdas, derivatives = self(context)
        x_center = self.x_center.expand(*x.shape[:-1]).unsqueeze(-1)
        y_center = self.y_center.expand(*x.shape[:-1]).unsqueeze(-1)

        y, logabsdet = self.spline.encode_and_logabsdet(
            x - x_center,
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
        )
        return y + y_center, logabsdet.sum(dim=-1)

    def decode_and_logabsdet(
        self,
        y: Tensor,  # (..., *H, $K)
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H)
        r"""Inverse pass of the flow."""
        if context.shape[:-1] != y.shape:
            raise ValueError(
                f"context batch/head/value shape {context.shape[:-1]} must match y "
                f"shape {y.shape}"
            )
        widths, heights, lambdas, derivatives = self(context)
        x_center = self.x_center.expand(*y.shape[:-1]).unsqueeze(-1)
        y_center = self.y_center.expand(*y.shape[:-1]).unsqueeze(-1)

        x, logabsdet = self.spline.decode_and_logabsdet(
            y - y_center,
            widths=widths,
            heights=heights,
            lambdas=lambdas,
            derivatives=derivatives,
        )
        return x + x_center, logabsdet.sum(dim=-1)


class ConditionalSplineFlow(ModuleSequence[ConditionalLRS]):
    r"""Implements a sequence of conditional rational linear spline layers."""

    @classmethod
    def from_iterable(
        cls, layers: Iterable[ConditionalLRS], /
    ) -> ConditionalSplineFlow:
        r"""Create a ConditionalSplineFlow from an iterable of conditional LRS layers."""
        new = ConditionalSplineFlow.__new__(ConditionalSplineFlow)
        super(ConditionalSplineFlow, new).__init__(layers)
        return new

    def __init__(
        self,
        dim_context: int,
        *,
        num_heads: int | tuple[int, ...] = (),
        num_flow_layers: int,
        num_bins: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        use_fp64: bool = True,
    ) -> None:
        layers = [
            ConditionalLRS(
                dim_context,
                num_bins=num_bins,
                num_heads=num_heads,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                use_fp64=use_fp64,
            )
            for _ in range(num_flow_layers)
        ]
        super().__init__(layers)

    def encode_and_logabsdet(
        self,
        x: Tensor,  # (..., *H, $K)
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H)
        logabsdet = torch.zeros_like(x[..., 0])
        for layer in self:
            x, ldj = layer.encode_and_logabsdet(x, context)
            logabsdet = logabsdet + ldj
        return x, logabsdet

    def decode_and_logabsdet(
        self,
        y: Tensor,  # (..., *H, $K)
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H)
        logabsdet = torch.zeros_like(y[..., 0])
        for layer in reversed(self):
            y, ldj = layer.decode_and_logabsdet(y, context)
            logabsdet = logabsdet + ldj
        return y, logabsdet


class SplineFlow(ModuleSequence[LearnableLRS]):
    r"""Implements a sequence of rational linear spline layers."""

    @classmethod
    def from_iterable(cls, layers: Iterable[LearnableLRS], /) -> SplineFlow:
        r"""Create a SplineFlow from an iterable of LRS layers."""
        new = SplineFlow.__new__(SplineFlow)
        super(SplineFlow, new).__init__(layers)
        return new

    def __init__(
        self,
        num_heads: int | tuple[int, ...] = (),
        *,
        num_flow_layers: int,
        num_bins: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        use_fp64: bool = True,
    ) -> None:
        layers = [
            LearnableLRS(
                num_heads,
                num_bins=num_bins,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                use_fp64=use_fp64,
            )
            for _ in range(num_flow_layers)
        ]
        super().__init__(layers)

    def marginalize(self, variables: list[int] | Tensor) -> SplineFlow:
        r"""Marginalize out the specified variables."""
        return SplineFlow.from_iterable(layer.marginalize(variables) for layer in self)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        logabsdet = torch.zeros_like(x[..., 0]) if x.ndim else torch.zeros_like(x)
        for layer in self:
            x, ldj = layer.encode_and_logabsdet(x)
            logabsdet = logabsdet + ldj
        return x, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        logabsdet = torch.zeros_like(y[..., 0]) if y.ndim else torch.zeros_like(y)
        for layer in reversed(self):
            y, ldj = layer.decode_and_logabsdet(y)
            logabsdet = logabsdet + ldj
        return y, logabsdet


def inverse_softplus(y: Tensor) -> Tensor:
    r"""Compute the inverse of the softplus function.

    y = softplus(x) = log(1 + exp(x))
    x = inverse_softplus(y) = log(exp(y) - 1) = y + log(1 - exp(-y)) = y + log(-expm1(-y))
    """
    return y + torch.log(-torch.expm1(-y))


class MultiHeadGaussian(nn.Module):
    r"""Implements a multi-head Gaussian distribution."""

    normalization_constant: Tensor
    r"""CONST: Normalization constant of a Gaussian distribution."""
    num_heads: Final[int]
    r"""CONST: Shape of heads"""
    num_features: Final[int]
    r"""CONST: Number of features in input."""

    # parameters/buffers
    means: Tensor
    r"""PARAM: Means of the gaussians."""
    scale_tril: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""PARAM: Parameters determining the covariances."""

    # non-permanent buffers
    eye: Tensor
    r"""BUFFER: Identity matrix."""
    covs: Tensor
    r"""BUFFER: Covariances of the gaussians."""
    cholesky_factor: Tensor  # shape: (n_gaussians, n_inputs, n_inputs)
    r"""BUFFER: Cholesky factor of the covariance matrix."""
    samples: Tensor
    r"""BUFFER: Stored samples when sampling."""
    latents: Tensor
    r"""BUFFER: Stored latents when evaluating log_probs."""
    log_probs: Tensor
    r"""BUFFER: Stored log_probs when evaluating log_probs."""

    @staticmethod
    def _sample_default_means(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default means $μᵢ∼𝓝(0,1)$, normalized."""
        means = torch.randn(n_heads, n_feats)
        means = means / means.norm(dim=-1, keepdim=True)
        return means

    @staticmethod
    def _sample_default_covs(n_heads: int, n_feats: int) -> Tensor:
        r"""Sample default covariances."""
        noise = (2 * torch.rand(n_heads, n_feats, n_feats) - 1) / (2 * n_feats)
        return torch.eye(n_feats) + noise

    def __init__(
        self,
        num_feats: int,
        *,
        num_heads: int,
        means: Optional[Tensor] = None,
        covs: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        # CONSTANTS
        self.num_heads = int(num_heads)
        self.num_features = int(num_feats)
        normalization_constant = (
            0.5 * self.num_features * math.log(2 * math.pi)
        )  # -log (2π)^{-k/2}
        self.register_buffer(
            "normalization_constant", torch.tensor(normalization_constant)
        )
        self.register_buffer("eye", torch.eye(num_feats, dtype=torch.bool))

        # BUFFERS
        self.register_buffer("covs", torch.empty(0), persistent=False)
        self.register_buffer("cholesky_factor", torch.empty(0), persistent=False)
        self.register_buffer("samples", torch.empty(0), persistent=False)
        self.register_buffer("latents", torch.empty(0), persistent=False)
        self.register_buffer("log_probs", torch.empty(0), persistent=False)

        # initialize the means
        self.means = nn.Parameter(
            torch.as_tensor(means)
            if means is not None
            else self._sample_default_means(num_heads, num_feats)
        )
        # initialize the covariances
        self.scale_tril = nn.Parameter(  # not a parameter!
            torch.as_tensor(covs).tril()
            if covs is not None
            else self._sample_default_covs(num_heads, num_feats).tril()
        )

        assert self.means.shape == (num_heads, num_feats)
        assert self.scale_tril.shape == (num_heads, num_feats, num_feats)

    @torch.no_grad()
    def marginalize(self, kept: list[int] | Tensor, /) -> MultiHeadGaussian:
        r"""Marginalize the distribution over the given indices.

        Given p(x,y) = 𝓝(μ, Σ), μ=[μₓ, μᵧ], Σ=[[Σₓₓ, Σₓᵧ], [Σᵧₓ, Σᵧᵧ]],
        the marginal distribution p(y) is given by 𝓝(μᵧ, Σᵧᵧ).
        """
        device = self.means.device
        kept = torch.as_tensor(kept, device=device)

        if kept.dtype == torch.bool:
            assert kept.shape == (self.num_features,)
            num_kept = int(kept.sum().item())
        else:
            assert kept.min() >= 0
            assert kept.max() < self.num_features
            num_kept = len(kept)

        orig_means = self.means
        marg_means = orig_means[:, kept]
        orig_covs = self.get_covariance()
        marg_covs = orig_covs[..., kept, :][..., :, kept]
        marg_chol = cholesky(marg_covs)
        marg_diag = marg_chol.diagonal(dim1=-2, dim2=-1)
        marg_tril = torch.where(
            self.eye[..., kept, :][..., :, kept],
            self._map_diagonal_inverse(marg_diag).unsqueeze(-1),
            marg_chol,
        )

        marg_model = MultiHeadGaussian(
            num_heads=self.num_heads,
            num_feats=num_kept,
        ).to(device=device)
        marg_model.means.copy_(marg_means)
        marg_model.scale_tril.copy_(marg_tril)

        # double check covariance match
        assert torch.allclose(marg_covs, marg_model.get_covariance())

        return marg_model

    @torch.no_grad()
    def condition(
        self, variables: list[int] | Tensor, values: list[float] | Tensor, /
    ) -> MultiHeadGaussian:
        r"""Condition the distribution on the given indices and values.

        Args:
            variables: The indices to condition on.
            values: The values to condition on.

        Given p(u,v) = 𝓝(μ, Σ), μ=[μᵤ, μᵥ], Σ=[[Σᵤᵤ, Σᵤᵥ], [Σᵥᵤ, Σᵥᵥ]],
        the conditional distribution is given by

        p(u∣v=v⁎) = 𝓝(u ∣ μᵤ + Σᵤᵥ Σᵥᵥ⁻¹ (v⁎-μᵥ), Σᵤᵤ - Σᵤᵥ Σᵥᵥ⁻¹ Σᵥᵤ)
        note that for the cholesky factorization,
        """
        device = self.means.device
        values = torch.as_tensor(values, device=device)
        variables = torch.as_tensor(variables, device=device)
        remaining = torch.tensor(
            [i for i in range(self.num_features) if i not in variables],
            device=device,
        )
        assert variables.max() < self.num_features
        assert variables.shape == values.shape[-1:]

        mu_u = self.means[:, remaining]
        mu_v = self.means[:, variables]
        orig_covs = self.get_covariance()
        sigma_uu = orig_covs[..., remaining, :][..., :, remaining]
        # sigma_uv = orig_covs[..., remaining, :][..., :, variables]
        sigma_vu = orig_covs[..., variables, :][..., :, remaining]
        sigma_vv = orig_covs[..., variables, :][..., :, variables]

        L = cholesky(sigma_vv)
        S = solve_triangular(L, sigma_vu, upper=False)  # S = L⁻¹ Σᵤᵥ, s = L⁻¹(v⁎ - μᵥ)
        s = solve_triangular(L, (values - mu_v).unsqueeze(-1), upper=False)
        cond_means = mu_u + torch.einsum("kpm, kpd -> km", S, s)  # μᵤ + Sᵀ s
        cond_covs = sigma_uu - torch.einsum("kpm, kpn -> kmn", S, S)  # Σᵤᵤ - Sᵀ S
        cond_chol = cholesky(cond_covs)

        # convert cholesky factor to unconstrained parameters (reverting softplus)
        tril_diag = self._map_diagonal_inverse(cond_chol.diagonal(dim1=-2, dim2=-1))
        cond_tril = torch.where(
            self.eye[..., remaining, :][..., :, remaining],
            tril_diag.unsqueeze(-1),
            cond_chol,
        )

        cond_model = MultiHeadGaussian(
            num_heads=self.num_heads,
            num_feats=self.num_features - len(variables),
        ).to(device=device)

        cond_model.means.copy_(cond_means)
        cond_model.scale_tril.copy_(cond_tril)

        # double check covariance match
        assert torch.allclose(cond_covs, cond_model.get_covariance())

        return cond_model

    def _map_diagonal(self, diag: Tensor) -> Tensor:
        r"""Map the diagonal of the cholesky factor to positive values."""
        return F.softplus(diag) + 1e-6

    def _map_diagonal_inverse(self, diag: Tensor) -> Tensor:
        r"""Map the diagonal of the cholesky factor to unconstrained values."""
        return inverse_softplus(diag - 1e-6)

    def get_cholesky(self) -> Tensor:
        r"""Compute cholesky factor of covariance matrix."""
        lower = self.scale_tril.tril()
        diag = lower.diagonal(dim1=-2, dim2=-1)
        diag = self._map_diagonal(diag)
        # (D, D), (M, D, 1), (M, D, D) -> (M, D, D)
        self.cholesky_factor = torch.where(self.eye, diag.unsqueeze(-1), lower)
        return self.cholesky_factor

    def get_covariance(self) -> Tensor:
        r"""Compute covariance matrix from cholesky factor."""
        L = self.get_cholesky()  # M x D x D
        self.covs = torch.einsum("mij,mkj->mik", L, L)  # L Lᵀ
        return self.covs

    def forward(self, x: Tensor) -> Tensor:
        r"""Transform $x -> y = Lx + μ$.

        Args:
            x (..., H, D): input tensor

        Returns:
            y (..., H, D): transformed tensor
        """
        L = self.get_cholesky()
        y = self.means + torch.einsum("...mj, mij -> ...mi", x, L)
        return y

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        r"""Transform $y -> x = L⁻¹(y-μ)$.

        Args:
            y (..., H, D): input tensor

        Returns:
            x (..., H, D): transformed tensor
            ldj (H): log determinant of the Jacobian
        """
        L = self.get_cholesky()

        # compute z = L⁻¹(x-μ)
        y = y - self.means
        y = y.unsqueeze(-1)  # (..., H, D) -> (..., H, D, 1)
        # (..., D, D), (..., D, 1) -> (..., D, 1)
        u = solve_triangular(L, y, upper=False)
        u = u.squeeze(-1)  # (..., D, 1) -> (..., D)

        # compute log |det L⁻¹| = - log |det L|
        #      = -log ∏ᵢ Lᵢᵢ = -∑ᵢ log Lᵢᵢ
        ldj = -L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return u, ldj

    def sample(self, size: int | tuple[int, ...]) -> Tensor:
        r"""Sample from the model.

        Args:
            size (int | tuple[int, ...]): size of the sample

        Returns:
            u (..., H, D): sample
        """
        shape = (size,) if isinstance(size, int) else size
        shape = (*shape, self.num_heads, self.num_features)
        z = torch.randn(*shape, device=self.normalization_constant.device)
        u = self.forward(z)
        self.samples = u  # store buffer for post-hoc analysis
        return u

    def log_prob(self, u: Tensor, /) -> Tensor:
        r"""Compute the log-likelihood of the input.

        Args:
            u (..., H, D): input tensor

        Returns:
            log_prob (..., H): log likelihood
        """
        self.latents = u  # store buffer for post-hoc analysis

        # parse through the gaussians
        z, ldj = self.inverse(u)

        # compute the base log-likelihood
        # log p(u) = -½ D log(2π) - ½‖z‖² - log|det L|
        log_prob = -self.normalization_constant - 0.5 * (z * z).sum(-1)  # (..., H)
        log_prob = log_prob + ldj  # (..., H)
        self.log_probs = log_prob  # store buffer for post-hoc analysis
        return log_prob


class MarginalizableNormalizingFlow(nn.Module):
    r"""Implements a Marginalizable Normalizing Flow (unconditional density model)."""

    num_features: Final[int]
    r"""Number of features in input."""
    num_components: Final[int]
    r"""Number of mixture components."""
    num_flow_layers: Final[int]
    r"""Number of rational linear spline layers."""
    num_bins: Final[int]
    r"""Number of bins in the rational linear splines."""
    bounds: Final[tuple[float, float]]
    r"""Tail bound of the rational linear splines."""

    mixture_params: Tensor  # shape: (n_gaussians,)
    r"""PARAM: Parameters determining the weights."""

    # buffers
    mixture_weights: Tensor
    r"""BUFFER: Mixture weights of the gaussians."""
    mixture_logits: Tensor
    r"""BUFFER: Mixture weight logits log wₖ."""
    sample_indices: Tensor
    r"""BUFFER: Indices of the mixture components."""
    latents: Tensor
    r"""BUFFER: Latent state of the model."""
    log_probs_per_head: Tensor
    r"""BUFFER: NLL of each Gaussian component."""
    log_probs: Tensor
    r"""BUFFER: NLL of the Gaussian base distribution."""
    samples: Tensor
    r"""BUFFER: Samples from the model."""
    logits: Tensor
    r"""BUFFER: Mixture logits."""

    # submodules
    flow: SplineFlow
    base: MultiHeadGaussian

    def __init__(
        self,
        num_feats: int,
        *,
        num_heads: int,
        num_flow_layers: int,
        num_bins: int = 16,
        bounds: tuple[float, float] = (-5, +5),
    ) -> None:
        super().__init__()
        # constants
        self.num_features = num_feats
        self.num_flow_layers = num_flow_layers
        self.num_components = num_heads
        self.num_bins = num_bins
        self.bounds = bounds if isinstance(bounds, tuple) else (bounds, bounds)

        # parameters
        initial_mixture = torch.ones(num_heads) / num_heads  # initialize 𝜔 = 1/M
        self.mixture_params = nn.Parameter(initial_mixture)

        # non-permanent buffers
        self.register_buffer("mixture_weights", torch.empty(0), persistent=False)
        self.register_buffer("mixture_logits", torch.empty(0), persistent=False)
        self.register_buffer("sample_indices", torch.empty(0), persistent=False)
        self.register_buffer("latents", torch.empty(0), persistent=False)
        self.register_buffer("samples", torch.empty(0), persistent=False)
        self.register_buffer("log_probs", torch.empty(0), persistent=False)
        self.register_buffer("log_probs_per_head", torch.empty(0), persistent=False)

        # NOTE: splines should be local, so instead of mapping fixed [left, right] -> [bottom, top]
        #  we should perform an offset accoring to the mean of each gaussian.

        # submodules
        self.flow = SplineFlow(
            (num_heads, num_feats),
            num_flow_layers=num_flow_layers,
            num_bins=num_bins,
            x_bounds=self.bounds,
            y_bounds=self.bounds,
        )
        self.base = MultiHeadGaussian(num_heads=num_heads, num_feats=num_feats)

    def get_mixture_weights(self) -> Tensor:
        r"""Compute mixture weights from mixture parameters."""
        weights = self.mixture_params.softmax(dim=0)
        self.mixture_weights = weights
        return weights

    def get_mixture_logits(self) -> Tensor:
        r"""Compute mixture logits from mixture parameters."""
        logits = self.mixture_params.log_softmax(dim=0)
        self.mixture_logits = logits
        return logits

    def log_prob(self, inputs: Tensor) -> Tensor:
        r"""Compute the log-likelihood of the input."""
        # (..., D) -> ...

        # create copy for each component (..., D) -> (..., H, D)
        x = inputs.unsqueeze(-2).repeat_interleave(self.num_components, dim=-2)

        # ℹ️ TRICK - shift by mean
        # x = x - self.base.means

        # parse through the flow
        u, log_det_flow = self.flow.encode_and_logabsdet(x)

        # ℹ️ TRICK - shift by mean
        # u = u + self.base.means

        self.latents = u  # store buffer for post-hoc analysis

        # compute the base log probability
        #  NOTE: shape (..., H) instead of (..., H, D), since operations are element-wise
        log_probs = self.base.log_prob(u) + log_det_flow  # (..., H)
        self.log_probs_per_head = log_probs  # store log pₖ(x) in buffer

        # compute the mixture logits
        logits = self.get_mixture_logits()  # (H)

        # (..., M), (..., M) -> ...
        log_prob = (logits + log_probs).logsumexp(dim=-1)  # (H), (..., H) -> (...)
        self.log_probs = log_prob  # store log p(x) in buffer
        return log_prob

    def sample(self, num: int) -> Tensor:
        r"""Sample from the model."""
        # sample from the base distribution
        u = self.base.sample(num)
        self.latents = u  # store buffer for post-hoc analysis

        # ℹ️ TRICK - shift by mean
        # u = u - self.base.means

        x, _ = self.flow.decode_and_logabsdet(u)

        # ℹ️ TRICK - shift by mean
        # x = x + self.base.means

        # select mixture components (N, M, D), (N) -> (N, D)
        select = self.sample_mixture(num)
        idx = torch.arange(num, device=select.device)
        samples = x[idx, select]  # NOTE: this is NOT the same as x[:, idx]
        self.samples = samples  # store buffer for post-hoc analysis
        return samples

    def sample_mixture(self, num: int) -> Tensor:  # shape: N
        r"""Return indices of the mixture."""
        p = self.get_mixture_weights()
        indices = torch.multinomial(p, num, replacement=True)
        self.sample_indices = indices  # save buffer for post-hoc analysis
        return indices

    @torch.no_grad()
    def marginalize(self, kept: list[int] | Tensor) -> MarginalizableNormalizingFlow:
        r"""Return a new MarginalizableNormalizingFlow with the specified features marginalized out."""
        device = self.mixture_params.device

        kept = torch.as_tensor(kept, device=device)
        if kept.dtype == torch.bool:
            assert kept.shape == (self.num_features,)
            num_kept = int(kept.sum().item())
        else:
            assert kept.min() >= 0
            assert kept.max() < self.num_features
            num_kept = len(kept)

        new = MarginalizableNormalizingFlow(
            num_feats=num_kept,
            num_heads=self.num_components,
            num_flow_layers=self.num_flow_layers,
            num_bins=self.num_bins,
            bounds=self.bounds,
        ).to(device=device)
        new.mixture_params.copy_(self.mixture_params)
        new.flow = self.flow.marginalize(kept)
        new.base = self.base.marginalize(kept)

        return new

    @torch.no_grad()
    def condition(
        self, variables: list[int] | Tensor, values: list[float] | Tensor
    ) -> MarginalizableNormalizingFlow:
        r"""Return a new MarginalizableNormalizingFlow with the specified features conditioned on the given values.

        Since p(x) = ∑ₖ wₖpₖ(f⁻¹(x))|det Jₖ(f⁻¹(x))|, we can compute the conditional mixture weights as follows:
        we have p(x∣y) = ∑ₖ wₖ'pₖ(f⁻¹(x)∣y)|det Jₖ(f⁻¹(x))|,
        where log wₖ' = log wₖ + log pₖ(y) - log p(y).

        """
        device = self.mixture_params.device
        dtype = self.mixture_params.dtype
        values = torch.as_tensor(values, device=device, dtype=dtype)
        assert values.shape == (len(variables),)
        variables = torch.as_tensor(variables, device=device)
        remaining = torch.tensor(
            [i for i in range(self.num_features) if i not in variables], device=device
        )
        marg_ndim = len(variables)
        cond_ndim = len(remaining)

        # compute the conditional mixture weights
        # log wₖ' = log wₖ + log pₖ(v⁎) - log p(v⁎)
        marg_model = self.marginalize(variables)  # p(v)
        log_p_v = marg_model.log_prob(values)  # log p(v⁎), shape=()
        log_p_v_k = marg_model.log_probs_per_head  # log pₖ(v⁎), shape=(H)
        log_w_k = marg_model.mixture_logits
        log_w = log_w_k + log_p_v_k - log_p_v

        # compute the conditional base distribution
        marg_flow = self.flow.marginalize(variables)
        latent_values, _ = marg_flow.encode_and_logabsdet(values)
        assert latent_values.shape == (self.num_components, marg_ndim)

        cond_model = MarginalizableNormalizingFlow(  # p(u | v)
            num_feats=cond_ndim,
            num_heads=self.num_components,
            num_flow_layers=self.num_flow_layers,
            num_bins=self.num_bins,
            bounds=self.bounds,
        ).to(device=device)

        cond_model.mixture_params.copy_(log_w)
        cond_model.flow = self.flow.marginalize(remaining)
        cond_model.base = self.base.condition(variables, latent_values)

        return cond_model


class PositionalEmbedding(nn.Module):
    r"""Time2Vec positional embedding.

    References:
        Time2Vec: Learning a Vector Representation of Time
        https://arxiv.org/abs/1907.05321
    """

    def __init__(self, num_frequencies: int) -> None:
        super().__init__()
        self.num_freq = num_frequencies
        self.frequencies = nn.Parameter(
            torch.logspace(0, num_frequencies - 1, num_frequencies, base=2.0)
        )
        self.offsets = nn.Parameter(torch.zeros(num_frequencies))

    # (...,) -> (..., F+1)
    def forward(self, t: Tensor, /) -> Tensor:
        r"""Compute the positional embedding for the given time step."""
        # sin(at+b)
        t = t.unsqueeze(-1)  # (..., 1)
        z = F.linear(t, self.frequencies[..., None], self.offsets)  # (..., F)
        return torch.cat([t, z.sin()], dim=-1)


class ChannelEmbedding(nn.Module):
    r"""Channel embedding (one-hot)."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    # (...) -> (..., C)
    def forward(self, c: Tensor, /) -> Tensor:
        valid = c >= 0  # invalid values marked with -1
        c = torch.where(valid, c, 0)
        e = F.one_hot(c, num_classes=self.num_channels).float()
        return torch.where(valid[..., None], e, nan)


class MultiHeadAttention(nn.Module):
    r"""Computes multi-head attention.

    .. math:: hᵢ = Attn(QWᵢ𐞥, KWᵢᵏ, VWᵢᵛ), r = concat(h₁, …, h_H)Wᵒ
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        *,
        dim_head: int,
        num_heads: int,
        dim_output: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.dim_head = dim_head
        self.dim_hidden = dim_head * num_heads
        self.dim_output = dim_output
        self.num_heads = num_heads

        self.q_proj = nn.Linear(q_dim, self.dim_hidden, bias=bias)
        self.k_proj = nn.Linear(k_dim, self.dim_hidden, bias=bias)
        self.v_proj = nn.Linear(v_dim, self.dim_hidden, bias=bias)
        self.out_proj = nn.Linear(self.dim_hidden, dim_output, bias=bias)

    def forward(
        self,
        q: Tensor,  # (..., $Q, d_q)
        k: Tensor,  # (..., $X, d_k)
        v: Tensor,  # (..., $X, d_v)
        /,
        *,
        query_mask: Tensor | None = None,  # Bool[(..., $Q)]
        key_mask: Tensor | None = None,  # Bool[(..., $X)]
    ) -> Tensor:  # (..., $Q, d_out)
        query_mask = (  # broadcast (..., $Q) -> (..., $Q, d_out)
            query_mask[..., :, None]
            if query_mask is not None
            else ~q.isnan().any(dim=-1).unsqueeze(-1)
        )
        key_mask = (  # broadcast (..., $X) -> (..., H, $Q, $X)
            key_mask[..., None, None, :]
            if key_mask is not None
            else ~k.isnan().any(dim=-1).unsqueeze(-2).unsqueeze(-2)
        )

        q = self.q_proj(q.nan_to_num(0.0))  # (..., $Q, H×d_h)
        k = self.k_proj(k.nan_to_num(0.0))  # (..., $X, H×d_h)
        v = self.v_proj(v.nan_to_num(0.0))  # (..., $X, H×d_h)

        q = q.unflatten(-1, (self.num_heads, self.dim_head))  # (..., $Q, H, d_h)
        k = k.unflatten(-1, (self.num_heads, self.dim_head))  # (..., $X, H, d_h)
        v = v.unflatten(-1, (self.num_heads, self.dim_head))  # (..., $X, H, d_h)

        h = F.scaled_dot_product_attention(  # (..., H, $Q, d_h)
            q.swapaxes(-2, -3),  # (..., H, $Q, d_h)
            k.swapaxes(-2, -3),  # (..., H, $X, d_h)
            v.swapaxes(-2, -3),  # (..., H, $X, d_h)
            attn_mask=key_mask,  # (..., H, $Q, $X)
            dropout_p=0.0,
        )
        # recombine heads
        h = h.swapaxes(-2, -3).flatten(-2)  # (..., $Q, H×d_h)
        y = self.out_proj(h)  # (..., $Q, d_out)
        return torch.where(query_mask, y, nan)  # mask out invalid queries


class MixtureWeightsModel(nn.Module):
    r"""Implements the mixture model used by moses.

    Given mixture-query embeddings ``β ∈ ℝᶜˣᴹ`` and a sequence of encoder
    embeddings ``h ∈ ℝᴺˣᴰ``, this module returns one mixture-weight vector per
    batch element.

    The paper writes

    .. math:: w = softmax(MHA(β, 𝐡, 𝐡)).

    We interpret this as a shorthand for the attention map induced by the
    learned queries ``β`` over the sequence ``h``: multi-head attention
    produces one attended embedding per query, which is then projected to a
    scalar logit and normalized across the query axis.
    """

    num_components: Final[int]
    num_heads: Final[int]
    dim_input: Final[int]
    dim_hidden: Final[int]
    mixture_queries: Tensor
    attention: MultiHeadAttention
    output_proj: nn.Linear

    def __init__(
        self,
        *,
        dim_input: int,
        dim_hidden: int,
        num_components: int,
        num_attn_heads: int,
    ) -> None:
        super().__init__()
        if num_components <= 0:
            raise ValueError(f"{num_components=} must be positive.")
        if num_attn_heads <= 0:
            raise ValueError(f"{num_attn_heads=} must be positive.")
        if dim_input <= 0:
            raise ValueError(f"{dim_input=} must be positive.")
        if dim_hidden <= 0:
            raise ValueError(f"{dim_hidden=} must be positive.")
        if dim_hidden % num_attn_heads != 0:
            raise ValueError(f"{dim_hidden=} must be divisible by {num_attn_heads=}.")

        self.num_components = num_components
        self.num_heads = num_attn_heads
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden

        self.mixture_queries = nn.Parameter(  # (C, M)
            torch.empty(num_components, dim_hidden),
        )
        nn.init.xavier_normal_(self.mixture_queries)

        self.attention = MultiHeadAttention(
            q_dim=dim_hidden,
            k_dim=dim_input,
            v_dim=dim_input,
            dim_head=dim_hidden // num_attn_heads,
            num_heads=num_attn_heads,
            dim_output=dim_hidden,
        )
        self.output_proj = nn.Linear(dim_hidden, 1)

    def forward(
        self,
        embeddings: Tensor,  # (..., $N, D)
        *,
        valid_mask: Tensor,  # Bool[(..., $N)]
    ) -> Tensor:  # (..., C), one normalized weight vector per batch element
        r"""Compute one mixture-weight vector per batch element.

        Args:
            embeddings: Sequence embeddings with shape ``(..., N, D)``.
            valid_mask: Optional boolean mask selecting valid sequence entries.
                If omitted, it is inferred from finite rows of ``embeddings``.

        Returns:
            Mixture weights with shape ``(..., C)``. Each batch element sums to
            1 across the mixture-query axis.
        """
        *batch_shape, seq_len, dim = embeddings.shape

        assert dim == self.dim_input
        assert valid_mask.dtype == torch.bool
        assert valid_mask.shape == (*batch_shape, seq_len)

        queries = self.mixture_queries.expand(
            *batch_shape,
            self.num_components,
            self.dim_hidden,
        )
        attended = self.attention(
            queries,
            embeddings,
            embeddings,
            key_mask=valid_mask,
        )
        logits = self.output_proj(attended).squeeze(dim=-1)  # (..., C)
        return logits.softmax(dim=-1)


class SeparableEncoder(nn.Module):
    r"""Implements the encoder used by moses.

    .. math::
        𝐱 = [pos_embed(t), one-hot(c), v]
        𝐪 = [pos_embed(t), one-hot(c)]
        𝐡ᵒᵇˢ = MHA(𝐱, 𝐱, 𝐱)
        𝐡 = MHA(𝐪, 𝐡ᵒᵇˢ, 𝐡ᵒᵇˢ)
    """

    def __init__(
        self,
        *,
        dim_output: int,
        dim_head: int,
        num_heads: int,
        num_components: int,
        num_frequencies: int,
        num_channels: int,
    ) -> None:
        super().__init__()
        self.dim_output = dim_output
        self.num_components = num_components
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        self.positional_embedding = PositionalEmbedding(num_frequencies)
        self.channel_embedding = ChannelEmbedding(num_channels)
        self.ctx_embed_dim = (num_frequencies + 1) + num_channels + 1
        self.qry_embed_dim = (num_frequencies + 1) + num_channels

        self.context_self_attention = MultiHeadAttention(
            q_dim=self.ctx_embed_dim,
            k_dim=self.ctx_embed_dim,
            v_dim=self.ctx_embed_dim,
            dim_head=dim_head,
            dim_output=dim_output,
            num_heads=num_heads,
        )
        self.cross_attention = MultiHeadAttention(
            q_dim=self.qry_embed_dim,
            k_dim=dim_output,
            v_dim=dim_output,
            dim_head=dim_head,
            dim_output=num_components * dim_output,
            num_heads=num_heads,
        )

    def forward(
        self,
        *,
        query_times: Tensor,  # Float[(..., $Q)], padded with NaN
        query_channels: Tensor,  # Long[(..., $Q)], padded with -1
        query_valid: Tensor | None = None,  # Bool[(..., $Q)],
        context_times: Tensor,  # Float[(..., $X)], padded with NaN
        context_channels: Tensor,  # Long[(..., $X)], padded with -1
        context_values: Tensor,  # Float[(..., $X)], padded with NaN
        context_valid: Tensor | None = None,  # Bool[(..., $X)],
    ) -> tuple[Tensor, Tensor]:  # (..., $X, M), (..., D, $X, M), padded with NaN
        r"""Compute per-query embeddings and mixture weights.

        Args:
            query_times: Query timestamps with shape ``(..., $Q)``.
                Invalid or padded positions should be encoded as ``NaN``
                when ``query_valid`` is omitted.
            query_channels: Query channel indices with shape ``(..., $Q)``.
                Invalid or padded positions should be encoded as ``-1``.
            query_valid: Optional boolean mask with shape ``(..., $Q)`` marking
                valid query positions. If omitted, validity is inferred from ``query_times``.
            context_times: Context timestamps with shape ``(..., $X)``.
                Invalid or padded positions should be encoded as ``NaN``
                when ``context_valid`` is omitted.
            context_channels: Context channel indices with shape ``(..., $X)``.
                Invalid or padded positions should be encoded as ``-1``.
            context_values: Context values with shape ``(..., $X)``.
                Invalid or padded positions should be encoded as ``NaN``.
            context_valid: Optional boolean mask with shape ``(..., $X)``
                marking valid context positions. If omitted, validity is
                inferred from ``context_times`` and ``context_values``.

        Returns:
            𝐡ᵒᵇˢ: Context embeddings with shape ``(..., $X, M)``.
            𝐡: Per-query embeddings with shape ``(..., D, $Q, M)``.
        """
        qry_valid = ~query_times.isnan() if query_valid is None else query_valid
        ctx_valid = (
            ~(context_times.isnan() | context_values.isnan())
            if context_valid is None
            else context_valid
        )

        context_times = context_times.masked_fill(~ctx_valid, 0.0)
        query_times = query_times.masked_fill(~qry_valid, 0.0)

        D = self.num_components
        M = self.dim_output

        x = torch.cat(  # (..., $X, D)
            [
                self.positional_embedding(context_times),
                self.channel_embedding(context_channels),
                context_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        q = torch.cat(  # (..., $Q, F)
            [
                self.positional_embedding(query_times),
                self.channel_embedding(query_channels),
            ],
            dim=-1,
        )

        h_obs = self.context_self_attention(x, x, x)  # (..., $X, M)
        h_mix = self.cross_attention(q, h_obs, h_obs)  # (..., $Q, D×M)
        h_mix = h_mix.reshape(*query_times.shape, D, M)  # (..., $Q, D, M)
        h_mix = h_mix.swapaxes(-2, -3)  # (..., D, $Q, M)
        return (
            h_obs.masked_fill(~ctx_valid[..., None], nan),
            h_mix.masked_fill(~qry_valid[..., None, :, None], nan),
        )


class ConditionalGaussian(nn.Module):
    r"""Implements the conditional Gaussian distribution used by moses.

    Given context embedding 𝐡∈ℝ^{D×K×M}, where

        D: number of mixture components
        K: number of query values
        M: dimensionality of each embedding,

    then this is a Normal distribution over ℝᴷ for each component.

    the mean μ(𝐡) and covariance Σ(𝐡) are computed as
    μ(𝐡) = 𝐡W, Σ(𝐡) = σ²𝕀 + (𝐡θ)(𝐡θ)ᵀ/√M.

    Since Σ(𝐡) is a low-rank update of an isotropic covariance, we can compute
    the log-likelihood and sample from the distribution efficiently using the
    Woodbury identity.
    """

    eye: Tensor
    r"""BUFFER: cached identity matrix."""

    def __init__(
        self,
        latent_dim: int,
        covariance_rank: int | None = None,
        num_heads: int | tuple[int, ...] = (),
        cov_scale_min: float = 1e-4,
    ) -> None:
        super().__init__()
        self.head_shape = (num_heads,) if isinstance(num_heads, int) else num_heads
        self.latent_dim = latent_dim
        covariance_rank = (
            max(1, latent_dim // 16) if covariance_rank is None else covariance_rank
        )
        self.covariance_rank = covariance_rank
        self.cov_scale_min = cov_scale_min

        if covariance_rank <= 0:
            raise ValueError(f"covariance_rank must be positive, got {covariance_rank}")
        if cov_scale_min < 0:
            raise ValueError(f"cov_scale_min must be non-negative, got {cov_scale_min}")

        # μ(𝐡) = 𝐡W, Σ(𝐡) = σ²𝕀 + (𝐡θ)(𝐡θ)ᵀ/√M
        self.mean_param = nn.Parameter(torch.randn(*self.head_shape, latent_dim))
        self.cov_param = nn.Parameter(
            torch.randn(*self.head_shape, latent_dim, covariance_rank)
        )
        self.cov_scale_param = nn.Parameter(torch.zeros(self.head_shape))
        self.scale = 0.5 / math.sqrt(latent_dim)
        self.register_buffer("eye", torch.eye(covariance_rank))

    def cov_scale(self) -> Tensor:
        r"""Return the positive isotropic covariance scale σ."""
        return self.cov_scale_min + F.celu(self.cov_scale_param) + 1.0

    def embed(
        self,
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H, $K, F)
        r"""Compute the mean and low-rank factor of the conditional Gaussian."""
        mean = torch.einsum("...kd, ...d -> ...k", context, self.mean_param)
        cov_factor = torch.einsum("...kd, ...df -> ...kf", context, self.cov_param)
        return mean, self.scale * cov_factor

    def forward(
        self,
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), (..., *H, $K, F)
        r"""Alias for :meth:`embed` to preserve module-call semantics."""
        return self.embed(context)

    def _log_prob(
        self, x: Tensor, mean: Tensor, cov_factor: Tensor, cov_scale: Tensor
    ) -> Tensor:
        # Write Σ = σ²(I + VVᵀ) with V = U / σ and U = cov_factor. Woodbury
        # gives (I + VVᵀ)⁻¹ = I - V(I + VᵀV)⁻¹Vᵀ, so the only factorization is
        # the small rank×rank system I + VᵀV.
        centered = x - mean  # (..., *H, K)
        event_size = x.shape[-1]
        inv_cov_scale = cov_scale.reciprocal()
        scaled_factor = cov_factor * inv_cov_scale[..., None, None]

        gram = scaled_factor.mT @ scaled_factor  # (..., *H, F, F)
        chol = cholesky(self.eye + gram)  # (..., *H, F, F)

        # xᵀΣ⁻¹x = σ⁻²(xᵀx - xᵀV(I + VᵀV)⁻¹Vᵀx) = σ⁻²(‖x‖² - ‖L⁻¹ Vᵀx‖²)
        projected = scaled_factor.mT @ centered.unsqueeze(-1)  # (..., *H, F, 1)
        whitened = solve_triangular(chol, projected, upper=False)
        quadratic = inv_cov_scale.square() * (
            centered.square().sum(dim=-1) - whitened.square().sum(dim=(-2, -1))
        ).clamp_min(0)

        # log p(x) = -½(xᵀΣ⁻¹x + log det Σ + K log 2π)
        # log det Σ = 2K log σ + 2∑ᵢ log Lᵢᵢ for Σ = σ²(I + VVᵀ).
        return -0.5 * (
            quadratic
            + event_size * (_LOG2PI + 2 * cov_scale.log())
            + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        )

    def _sample(
        self,
        size: tuple[int, ...],
        mean: Tensor,
        cov_factor: Tensor,
        cov_scale: Tensor,
    ) -> Tensor:
        # Write Σ = σ²I + UUᵀ with U = cov_factor. Then [σI, U] is a
        # rectangular square root because [σI, U][σI, U]ᵀ = σ²I + UUᵀ = Σ.
        white_noise = torch.randn(
            (*size, *mean.shape),
            dtype=mean.dtype,
            device=mean.device,
        )
        # Sampling ε ∼ 𝓝(0, Iₖ) and ξ ∼ 𝓝(0, Iᵣ) independently is equivalent to
        # drawing [ε; ξ] ∼ 𝓝(0, Iₖ₊ᵣ) and applying the usual reparameterization
        # x = μ + [σI, U][ε; ξ] = μ + σε + Uξ, without forming a dense K×K
        # factor.
        rank_noise = torch.randn(
            (*size, *cov_factor.shape[:-2], cov_factor.shape[-1]),
            dtype=cov_factor.dtype,
            device=cov_factor.device,
        )
        return (
            mean
            + cov_scale.unsqueeze(-1) * white_noise
            + torch.einsum("...kf, ...f -> ...k", cov_factor, rank_noise)
        )

    def log_prob(
        self,
        x: Tensor,  # (..., *H, $K)
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> Tensor:  # (..., *H)
        r"""Compute the log-likelihood of the input."""
        mean, cov_factor = self.embed(context)
        return self._log_prob(x, mean, cov_factor, self.cov_scale())

    def sample(
        self,
        size: tuple[int, ...],
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> Tensor:  # (..., *H, $K)
        r"""Sample a Gaussian distribution from the conditional distribution."""
        mean, cov_factor = self.embed(context)
        return self._sample(size, mean, cov_factor, self.cov_scale())

    def sample_and_log_prob(
        self,
        size: tuple[int, ...],
        context: Tensor,  # (..., *H, $K, D)
        /,
    ) -> tuple[Tensor, Tensor]:  # (..., *H, $K), # (..., *H)
        r"""Sample a Gaussian distribution from the conditional distribution."""
        mean, cov_factor = self.embed(context)
        cov_scale = self.cov_scale()
        samples = self._sample(size, mean, cov_factor, cov_scale)
        log_prob = self._log_prob(samples, mean, cov_factor, cov_scale)
        return samples, log_prob


class Moses(nn.Module):
    r"""Context-conditioned mixture normalizing flow for irregular time-series forecasting.

    Combines a GraFITi encoder with a per-query mixture of spline flows.  The
    encoder maps context observations and query positions to per-query
    embeddings ``H (..., K, M)``, which are projected to per-query mixture
    logits and Gaussian means.  One learned spline flow per mixture component
    maps Gaussian latents to the data space.

    Since the spline flows have context-independent parameters, the analytical
    marginalisation property of :class:`MarginalizableNormalizingFlow` is
    preserved (though not yet exposed on this class).

    Args:
        input_dim: Number of observed channels ``D``.
        latent_dim: GraFITi / linear-projection embedding size ``M``.
        num_components: Number of mixture components ``C``.
        num_flow_layers: Number of spline layers per component.
        num_bins: Number of rational-spline bins.
        bounds: Symmetric spline input/output domain ``(lo, hi)``.
        num_encoder_layers: Number of GraFITi attention layers.
        num_encoder_heads: Number of GraFITi attention heads.
    """

    num_components: Final[int]
    num_bins: Final[int]
    bounds: Final[tuple[float, float]]

    # sub-modules / parameters
    encoder: Grafiti
    mixture_weight_model: nn.Linear  # M → C  (mixture logits)
    mean_proj: nn.Linear  # M → C  (Gaussian mean per component)
    log_std: Tensor  # (C,), nn.Parameter — shared log-std per component
    component_flows: nn.Module  # C × SplineFlow, each n_heads=()

    @classmethod
    def from_config(
        cls,
        *,
        input_dim: int,
        latent_dim: int = 128,
        num_components: int = 4,
        num_flow_layers: int = 3,
        num_bins: int = 16,
        bounds: tuple[float, float] = (-5.0, 5.0),
        num_encoder_layers: int = 3,
        num_encoder_heads: int = 4,
    ) -> Moses:
        raise NotImplementedError

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int = 128,
        num_components: int = 4,
        num_flow_layers: int = 3,
        num_bins: int = 16,
        bounds: tuple[float, float] = (-5.0, 5.0),
        num_encoder_layers: int = 3,
        num_encoder_heads: int = 4,
        mixture_weight_model: nn.Module,
        covariance_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.num_bins = num_bins
        self.bounds = bounds

        self.encoder = Grafiti(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            output_mode="embeddings",
        )

        self.base_distribution = ConditionalGaussian(
            latent_dim=latent_dim,
            covariance_rank=covariance_rank,
            num_heads=num_components,
        )

        # One head per component. (eq 14)
        self.component_flows = ConditionalSplineFlow(
            dim_context=latent_dim,
            num_heads=num_components,
            num_flow_layers=num_flow_layers,
            num_bins=num_bins,
            x_bounds=bounds,
            y_bounds=bounds,
        )

        # X -> w(X) (eq 15)
        self.mixture_weight_model = MixtureWeightsModel(
            num_components=num_components,
            dim_input=latent_dim,
            dim_hidden=latent_dim,
            num_attn_heads=4,
        )

    def _encode(
        self,
        *,
        time_points: Tensor,  # (..., $T)
        context_values: Tensor,  # (..., $T, D)
        context_mask: Tensor,  # (..., $T, D), bool
        query_mask: Tensor,  # (..., $T, D), bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Run encoder and compute per-query mixture parameters.

        Returns:
            H: Per-query embeddings ``(..., K, M)``.
            log_w: Per-query log mixture weights ``(..., K, C)``.
            mu: Per-query Gaussian means ``(..., K, C)``.
            sigma: Shared std per component ``(C,)``.
        """
        H = self.encoder(  # (..., K, M)
            time_points,
            context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        log_w = self.mixture_weight_model(H).log_softmax(dim=-1)  # (..., K, C)
        mu = self.mean_proj(H)  # (..., K, C)
        sigma = F.softplus(self.log_std) + 1e-6  # (C,)
        return H, log_w, mu, sigma

    def _predict(
        self,
        *,
        query_times: Tensor,  # Float[(..., $K)], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[(..., $K, F)]  padded False
        context_times: Tensor,  # Float[(..., $N)], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[(..., $N, D)], padded False
        context_values: Tensor,  # Float[(..., $N, D)], padded NaN, sparse
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Run encoder and compute per-query mixture parameters.

        Returns:
            H: Per-query embeddings ``(..., K, M)``.
            log_w: Per-query log mixture weights ``(..., K, C)``.
            mu: Per-query Gaussian means ``(..., K, C)``.
            sigma: Shared std per component ``(C,)``.
        """
        request = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
        )
        H = self.encoder.forward(  # (..., K, M)
            timestamps=request.timestamps,
            context_values=request.context_values,
            context_mask=request.context_mask,
            query_mask=request.query_mask,
        )
        log_w = self.mixture_weight_model(H).log_softmax(dim=-1)  # (..., K, C)
        mu = self.mean_proj(H)  # (..., K, C)
        sigma = F.softplus(self.log_std) + 1e-6  # (C,)
        return H, log_w, mu, sigma

    def log_prob(
        self,
        values: Tensor,  # (..., $T, D)
        *,
        query_times: Tensor,  # Float[(..., $K)], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[(..., $K, F)]  padded False
        context_times: Tensor,  # Float[(..., $N)], padded NaN, non-decreasing
        context_values: Tensor,  # Float[(..., $N, D)], padded NaN, sparse
        context_mask: Tensor,  # Bool[(..., $N, D)], padded False
    ) -> Tensor:  # (...,)
        r"""Compute the joint log-likelihood of the target values.

        Evaluates the marginal mixture-flow log-prob at each query position
        and sums across query positions (joint under independence).

        Args:
            values: Observed target values; finite at every ``True`` position
                in ``query_mask``.
            query_times: Sorted time stamps for all query time steps.
            query_mask: Boolean mask selecting query (target) positions.
            context_times: Sorted time stamps for all context time steps.
            context_values: Context observations; ``NaN`` at unobserved positions.
            context_mask: Boolean mask selecting observed context positions.

        Returns:
            Joint log-likelihood with shape ``(...,)``.
        """
        *batch_shape, _, _ = context_values.shape

        H, log_w, mu, sigma = self._predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        valid_mask = H.isfinite().all(dim=-1)  # (..., K)
        max_K = H.shape[-2]

        # Pack query values: (..., T, D) → (..., K), NaN at unused K slots.
        # query_mask and valid_mask have the same True count per batch element
        # (GraFITi packs targets in row-major time-sorted order).
        y_flat = values.new_full((*batch_shape, max_K), nan)
        y_safe = torch.where(valid_mask, y_flat, 0.0)  # avoid NaN in flow inputs
        y_safe[valid_mask] = values[query_mask]

        # Per-component flow: encode each scalar independently.
        z_list: list[Tensor] = []
        ldj_list: list[Tensor] = []
        for flow in self.component_flows:
            z_c, ldj_c = flow.encode_and_logabsdet(y_safe)  # (..., K), (..., K)
            z_list.append(z_c)
            ldj_list.append(ldj_c)
        z = torch.stack(z_list, dim=-1)  # (..., K, C)
        ldj = torch.stack(ldj_list, dim=-1)  # (..., K, C)

        # Gaussian log-prob per (query, component).
        log_p_z = (
            -0.5 * ((z - mu) / sigma).square() - sigma.log() - 0.5 * _LOG2PI
        )  # (..., K, C)

        # Mixture combination → per-query marginal log-prob.
        log_p = (log_w + log_p_z + ldj).logsumexp(dim=-1)  # (..., K)

        # Sum over valid query slots; zero-out padding.
        return torch.where(valid_mask, log_p, 0.0).sum(dim=-1)  # (...,)

    def sample(
        self,
        size: int | tuple[int, ...] = (),  # *S
        *,
        query_times: Tensor,  # Float[(..., $K)], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[(..., $K, F)]  padded False
        context_times: Tensor,  # Float[(..., $N)], padded NaN, non-decreasing
        context_values: Tensor,  # Float[(..., $N, D)], padded NaN, sparse
        context_mask: Tensor,  # Bool[(..., $N, D)], padded False
    ) -> Tensor:  # (*S, ..., $K, D)
        r"""Sample from the conditional marginal distribution at each query position.

        Args:
            size: Number of samples (leading sample dimensions ``*S``).
            query_times: Sorted time stamps for all query time steps.
            query_mask: Boolean mask selecting query (target) positions.
            context_times: Sorted time stamps for all context time steps.
            context_values: Context observations; ``NaN`` at unobserved positions.
            context_mask: Boolean mask selecting observed context positions.

        Returns:
            Samples with shape ``(*S, ..., T, D)``, ``NaN`` at non-query positions.
        """
        *batch_shape, num_steps, num_channels = context_values.shape
        sample_shape = (size,) if isinstance(size, int) else tuple(size)

        H, log_w, mu, sigma = self._predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        valid_mask = H.isfinite().all(dim=-1)  # (..., K)
        max_K = H.shape[-2]

        # Expand mixture parameters to sample shape.
        log_w_s = log_w.expand(*sample_shape, *log_w.shape)  # (*S, ..., K, C)
        mu_s = mu.expand(*sample_shape, *mu.shape)  # (*S, ..., K, C)

        # Sample a mixture component per query point.
        probs_flat = log_w_s.exp().reshape(-1, self.num_components)  # (N, C)
        c_idx = (
            torch.multinomial(probs_flat, num_samples=1)
            .squeeze(-1)
            .reshape(*sample_shape, *batch_shape, max_K)
        )  # (*S, ..., K)

        # Draw from the selected Gaussian component.
        mu_sel = mu_s.gather(-1, c_idx.unsqueeze(-1)).squeeze(-1)  # (*S, ..., K)
        sigma_sel = sigma[c_idx]  # (*S, ..., K)
        z = mu_sel + sigma_sel * torch.randn_like(mu_sel)  # (*S, ..., K)

        # Invert all component flows on z, then select the drawn component.
        y_per_comp = torch.stack(
            [flow.decode_and_logabsdet(z)[0] for flow in self.component_flows],
            dim=-1,
        )  # (*S, ..., K, C)
        y_flat = y_per_comp.gather(-1, c_idx.unsqueeze(-1)).squeeze(-1)  # (*S, ..., K)

        # Mask padding slots and unpack from (..., K) → (..., T, D).
        valid_mask_s = valid_mask.expand(*sample_shape, *valid_mask.shape)
        y_flat = torch.where(valid_mask_s, y_flat, nan)

        samples = y_flat.new_full(
            (*sample_shape, *batch_shape, num_steps, num_channels), nan
        )
        query_mask_s = query_mask.expand(*sample_shape, *query_mask.shape)
        samples[query_mask_s] = y_flat[valid_mask_s]

        return samples  # (*S, ..., T, D)
