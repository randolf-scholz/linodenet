import pytest
import torch

from .assertions import TestSuite


class TestBoundedAssertions(TestSuite):
    def test_assert_upper_bounded_broadcasts_and_equalizes_dtype(self) -> None:
        value = torch.tensor([[1.0], [1.5]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        self.assert_upper_bounded(value, bound)

    def test_assert_upper_bounded_allows_signed_slack(self) -> None:
        value = torch.tensor([-0.95, 1.05], dtype=torch.float32)
        bound = torch.tensor([-1.0, 1.0], dtype=torch.float64)

        self.assert_upper_bounded(value, bound, rtol=0.1)

    def test_assert_lower_bounded_broadcasts_and_equalizes_dtype(self) -> None:
        value = torch.tensor([[2.5], [3.5]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        self.assert_lower_bounded(value, bound)

    def test_assert_upper_bounded_reports_worst_offender(self) -> None:
        value = torch.tensor([[2.0], [1.0]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        with pytest.raises(
            AssertionError, match=r"worst offender upper bound="
        ) as exc_info:
            self.assert_upper_bounded(value, bound)

        assert "worst offender index=(0, 0)" in str(exc_info.value)
        assert "worst offender value=2.0" in str(exc_info.value)

    def test_assert_upper_bounded_reports_negative_bound_violation(self) -> None:
        value = torch.tensor([-0.7], dtype=torch.float32)
        bound = torch.tensor([-1.0], dtype=torch.float64)

        with pytest.raises(
            AssertionError, match=r"worst offender upper bound="
        ) as exc_info:
            self.assert_upper_bounded(value, bound, rtol=0.1)

        assert "worst offender value=-0.699999988079071" in str(exc_info.value)
        assert "worst offender upper bound=-0.899999" in str(exc_info.value)

    def test_assert_lower_bounded_reports_worst_offender(self) -> None:
        value = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        with pytest.raises(
            AssertionError, match=r"worst offender lower bound="
        ) as exc_info:
            self.assert_lower_bounded(value, bound)

        assert "worst offender index=(0, 1)" in str(exc_info.value)
        assert "worst offender value=1.0" in str(exc_info.value)

    def test_assert_magnitude_bounded_uses_absolute_values(self) -> None:
        value = torch.tensor([-1.8, 0.4], dtype=torch.float32)
        right = torch.tensor([2.0, -1.0], dtype=torch.float64)

        self.assert_magnitude_bounded(value, right, scale=1.0)

    def test_assert_magnitude_bounded_reports_scale(self) -> None:
        value = torch.tensor([-3.0, 0.5], dtype=torch.float32)
        right = torch.tensor([2.0, -1.0], dtype=torch.float64)

        with pytest.raises(
            AssertionError, match=r"worst offender actual scale="
        ) as exc_info:
            self.assert_magnitude_bounded(value, right, scale=1.0)

        assert "worst offender index=(0,)" in str(exc_info.value)
        assert "worst offender right=2.0" in str(exc_info.value)
        assert "worst offender expected scale=1.00e+00" in str(exc_info.value)
