__all__ = ["TestCase"]

import warnings

import torch
from torch import Tensor


class TestCase:
    ATOL = 1e-6
    RTOL = 1e-6

    def assert_upper_bounded(
        self,
        value: Tensor | float,
        bound: Tensor | float,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| ≤ (1+rtol) |right| + atol."""
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        bound = torch.as_tensor(bound)
        upper_bound = (1 + rtol) * bound + atol
        assert upper_bound >= 0.0
        ok = value <= upper_bound

        abs_violation = (value - upper_bound).clamp_min(0)
        rel_violation = abs_violation / upper_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            msg = (
                f"Values exceed bound! "
                f"\n\tvalue: {value.tolist()}"
                f"\n\tbound: {bound.tolist()}"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
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
        bound: Tensor | float,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_loose: bool = False,
    ) -> None:
        r"""Check that |left| ≥ (1-rtol) |right| - atol."""
        __tracebackhide__ = True

        value = torch.as_tensor(value)
        bound = torch.as_tensor(bound)
        lower_bound = (1 - rtol) * bound - atol
        assert (1 - rtol) >= 0.0
        ok = value >= lower_bound

        abs_violation = (lower_bound - value).clamp_min(0)
        rel_violation = abs_violation / lower_bound.abs()

        max_abs_err = abs_violation.max().item()
        mean_abs_err = abs_violation.mean().item()
        median_abs_err = abs_violation.median().item()
        max_rel_err = rel_violation.max().item()
        mean_rel_err = rel_violation.nanmean().item()
        median_rel_err = rel_violation.nanmedian().item()

        if not ok.all():
            msg = (
                f"Values exceed bound! "
                f"\n\tvalue: {value.tolist()}"
                f"\n\tbound: {bound.tolist()}"
                f"\n\tmax    abs violation={max_abs_err:8.2e}  (expected {atol})"
                f"\n\tmean   abs violation={mean_abs_err:8.2e}  (expected {atol})"
                f"\n\tmedian abs violation={median_abs_err:8.2e}  (expected {atol})"
                f"\n\tmax    rel violation={max_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmean   rel violation={mean_rel_err:8.2e}  (expected {rtol})"
                f"\n\tmedian rel violation={median_rel_err:8.2e}  (expected {rtol})"
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
            flat_index = residual.reshape(-1).argmax().item()
            worst_index = (
                ()
                if residual.ndim == 0
                else tuple(
                    torch.unravel_index(
                        torch.tensor(flat_index, device=residual.device),
                        residual.shape,
                    )
                )
            )
            worst_value = x_hat.reshape(-1)[flat_index].item()
            worst_expected = x_ref.reshape(-1)[flat_index].item()
            worst_abs_err = residual.reshape(-1)[flat_index].item()
            worst_rel_err = (residual / magnitude).reshape(-1)[flat_index].item()
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
