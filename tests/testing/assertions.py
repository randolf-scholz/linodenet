__all__ = ["TestSuite"]

import warnings

import torch
from torch import Tensor


class TestSuite:
    ATOL = 1e-6
    RTOL = 1e-6

    @staticmethod
    def _worst_offender(
        residual: Tensor,
        expected: Tensor,
        actual: Tensor,
    ) -> tuple[tuple[int, ...], object, object, float, float]:
        flat_index = int(residual.reshape(-1).argmax().item())
        worst_index = (
            ()
            if residual.ndim == 0
            else tuple(
                int(index.item())
                for index in torch.unravel_index(
                    torch.tensor(flat_index, device=residual.device),
                    residual.shape,
                )
            )
        )
        worst_value = actual.reshape(-1)[flat_index].item()
        worst_expected = expected.reshape(-1)[flat_index].item()
        worst_abs_err = float(residual.reshape(-1)[flat_index].item())
        worst_rel_err = float(
            (residual / expected).abs().reshape(-1)[flat_index].item()
        )
        return worst_index, worst_value, worst_expected, worst_abs_err, worst_rel_err

    def assert_upper_bounded(
        self,
        value: Tensor | float,
        upper: Tensor | float,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| ≤ (1+rtol) |right| + atol."""
        __tracebackhide__ = True

        x_hat = torch.as_tensor(value)
        bound = torch.as_tensor(upper, device=x_hat.device, dtype=x_hat.dtype)
        x_hat, bound = torch.broadcast_tensors(x_hat, bound)
        upper_bound = (1 + rtol) * bound + atol
        assert (upper_bound >= 0.0).all(), "upper bound must be non-negative"
        ok = x_hat <= upper_bound

        abs_violation = (x_hat - upper_bound).clamp_min(0)
        rel_violation = abs_violation / upper_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            (
                worst_index,
                worst_value,
                worst_upper_bound,
                worst_abs_err,
                worst_rel_err,
            ) = self._worst_offender(abs_violation, upper_bound, x_hat)
            msg = (
                f"Values exceed bound! "
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
                f"\n\tworst offender index={worst_index}"
                f"\n\tworst offender value={worst_value!r}"
                f"\n\tworst offender upper bound={worst_upper_bound!r}"
                f"\n\tworst offender abs violation={worst_abs_err:8.2e}"
                f"\n\tworst offender rel violation={worst_rel_err:8.2e}"
            )
            raise AssertionError(msg)

        if warn_loose and (max_abs_err < 1e-3 or max_rel_err < 1e-3):
            warnings.warn(
                f"Bounds are loose:"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})",
                stacklevel=2,
            )

    def assert_lower_bounded(
        self,
        value: Tensor | float,
        lower: Tensor | float,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| ≥ (1-rtol) |right| - atol."""
        __tracebackhide__ = True

        x_hat = torch.as_tensor(value)
        bound = torch.as_tensor(lower, device=x_hat.device, dtype=x_hat.dtype)
        x_hat, bound = torch.broadcast_tensors(x_hat, bound)
        lower_bound = (1 - rtol) * bound - atol
        assert (1 - rtol) >= 0.0
        ok = x_hat >= lower_bound

        abs_violation = (lower_bound - x_hat).clamp_min(0)
        rel_violation = abs_violation / lower_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            (
                worst_index,
                worst_value,
                worst_lower_bound,
                worst_abs_err,
                worst_rel_err,
            ) = self._worst_offender(abs_violation, lower_bound, x_hat)
            msg = (
                f"Values exceed bound! "
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
                f"\n\tworst offender index={worst_index}"
                f"\n\tworst offender value={worst_value!r}"
                f"\n\tworst offender lower bound={worst_lower_bound!r}"
                f"\n\tworst offender abs violation={worst_abs_err:8.2e}"
                f"\n\tworst offender rel violation={worst_rel_err:8.2e}"
            )
            raise AssertionError(msg)

        if warn_loose and (max_abs_err < 1e-3 or max_rel_err < 1e-3):
            warnings.warn(
                f"Bounds are loose:"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})",
                stacklevel=2,
            )

    def assert_close(
        self,
        value: Tensor | float,
        expected: Tensor | float,
        *,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> None:
        r"""Checks that |value - expected| ≤ rtol|expected| + atol."""
        __tracebackhide__ = True

        x_hat = torch.as_tensor(value)
        x_ref = torch.as_tensor(expected, device=x_hat.device, dtype=x_hat.dtype)
        x_hat, x_ref = torch.broadcast_tensors(x_hat, x_ref)
        residual = (x_hat - x_ref).abs()
        magnitude = x_ref.abs()
        ok = residual <= rtol * magnitude + atol

        if not ok.all():
            worst_index, worst_value, worst_expected, worst_abs_err, worst_rel_err = (
                self._worst_offender(residual, x_ref, x_hat)
            )
            max_abs_err = residual.max().item()
            mean_abs_err = residual.mean().item()
            median_abs_err = residual.median().item()
            max_rel_err = (residual / magnitude).max().item()
            mean_rel_err = (residual / magnitude).nanmean().item()
            median_rel_err = (residual / magnitude).nanmedian().item()
            msg = (
                f"Values not close! "
                # f"\n\tleft:  {value.tolist()}"
                # f"\n\tright: {true_value.tolist()}"
                f"\n\tmax    abs error={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs error={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs error={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel error={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel error={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel error={median_rel_err:8.2e}  (expected {rtol})"
                f"\n\tworst offender index={worst_index}"
                f"\n\tworst offender value={worst_value!r}"
                f"\n\tworst offender expected={worst_expected!r}"
                f"\n\tworst offender abs error={worst_abs_err:8.2e}"
                f"\n\tworst offender rel error={worst_rel_err:8.2e}"
            )
            raise AssertionError(msg)

    def assert_not_close(
        self,
        value: Tensor | float,
        expected: Tensor | float,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> None:
        r"""Checks that |value - expected| > rtol|expected| + atol."""
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        expected = torch.as_tensor(expected)
        residual = (value - expected).abs()
        magnitude = expected.abs()
        ok = residual > rtol * magnitude + atol

        if not ok.all():
            max_abs_err = residual.max().item()
            mean_abs_err = residual.mean().item()
            median_abs_err = residual.median().item()
            max_rel_err = (residual / magnitude).max().item()
            mean_rel_err = (residual / magnitude).nanmean().item()
            median_rel_err = (residual / magnitude).nanmedian().item()
            msg = (
                f"Values unexpectedly close! "
                f"\n\tmax    abs error={max_abs_err:8.2e}  (expected > {atol})"
                f"\n\tmean   abs error={mean_abs_err:8.2e}  (expected > {atol})"
                f"\n\tmedian abs error={median_abs_err:8.2e}  (expected > {atol})"
                f"\n\tmax    rel error={max_rel_err:8.2e}  (expected > {rtol})"
                f"\n\tmean   rel error={mean_rel_err:8.2e}  (expected > {rtol})"
                f"\n\tmedian rel error={median_rel_err:8.2e}  (expected > {rtol})"
            )
            raise AssertionError(msg)
