import pytest
import torch

from .assertions import TestSuite


class TestBoundedAssertions(TestSuite):
    def test_assert_upper_bounded_broadcasts_and_equalizes_dtype(self) -> None:
        value = torch.tensor([[1.0], [1.5]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        self.assert_upper_bounded(value, bound)

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

    def test_assert_lower_bounded_reports_worst_offender(self) -> None:
        value = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        bound = torch.tensor([1.5, 2.5], dtype=torch.float64)

        with pytest.raises(
            AssertionError, match=r"worst offender lower bound="
        ) as exc_info:
            self.assert_lower_bounded(value, bound)

        assert "worst offender index=(0, 1)" in str(exc_info.value)
        assert "worst offender value=1.0" in str(exc_info.value)
