import pytest

from linodenet.domains import Interval, UnionOfIntervals


def test_interval_add_shifts_bounds() -> None:
    interval = Interval.from_string("(0, 1]")

    assert interval + 2.0 == Interval.from_string("(2, 3]")


def test_interval_add_positive_infinity_collapses_upward() -> None:
    interval = Interval.from_string("(0, 1]")

    assert interval + float("inf") == Interval.from_string("(inf, inf]")


def test_interval_sub_shifts_bounds() -> None:
    interval = Interval.from_string("(0, 1]")

    assert interval - 2.0 == Interval.from_string("(-2, -1]")


def test_interval_sub_positive_infinity_collapses_downward() -> None:
    interval = Interval.from_string("(0, 1]")

    assert interval - float("inf") == Interval.from_string("(-inf, -inf]")


def test_interval_mul_positive_scales_bounds() -> None:
    interval = Interval.from_string("(1, 2]")

    assert interval * 3.0 == Interval.from_string("(3, 6]")


def test_interval_mul_positive_infinity() -> None:
    interval = Interval.from_string("(1, 2]")

    assert interval * float("inf") == Interval.from_string("(inf, inf]")


def test_interval_mul_negative_flips_bounds_and_inclusivity() -> None:
    interval = Interval.from_string("(1, 2]")

    assert interval * -2.0 == Interval.from_string("[-4, -2)")


def test_interval_mul_negative_infinity_flips_bounds_and_inclusivity() -> None:
    interval = Interval.from_string("(1, 2]")

    assert interval * float("-inf") == Interval.from_string("[-inf, -inf)")


def test_interval_mul_zero_collapses_to_closed_origin() -> None:
    interval = Interval.from_string("(1, 2]")

    assert interval * 0.0 == Interval.from_string("[0, 0]")


def test_interval_subset_relation() -> None:
    assert Interval.from_string("(0, 1)") <= Interval.from_string("[0, 1]")
    assert Interval.from_string("[0, 1]") <= Interval.from_string("[-1, 2]")


def test_interval_subset_relation_respects_open_and_closed_bounds() -> None:
    assert not Interval.from_string("[0, 1]") <= Interval.from_string("(0, 1]")
    assert not Interval.from_string("[0, 1]") <= Interval.from_string("[0, 1)")


def test_interval_subset_relation_accepts_string_on_right() -> None:
    assert Interval.from_string("[0, 1]") <= "[-1, 2]"


def test_interval_subset_relation_accepts_union_on_right() -> None:
    assert Interval.from_string("[0, 1]") <= UnionOfIntervals.from_string(
        "[-2, -1] | [0, 2]"
    )


def test_interval_subset_relation_accepts_string_on_left() -> None:
    assert "[0, 1]" <= Interval.from_string("[-1, 2]")


def test_interval_equality_accepts_string_on_right() -> None:
    assert Interval.from_string("[0, 1]") == "[0, 1]"


def test_interval_equality_accepts_string_on_left() -> None:
    assert "[0, 1]" == Interval.from_string("[0, 1]")


def test_interval_rejects_unrelated_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = Interval.from_string("[0, 1]") <= 1


def test_union_of_intervals_add_applies_to_each_member() -> None:
    union = UnionOfIntervals.from_string("[-2, -1] | (1, 2]")

    assert union + 2.0 == UnionOfIntervals.from_string("[0, 1] | (3, 4]")


def test_union_of_intervals_sub_negative_infinity() -> None:
    union = UnionOfIntervals.from_string("[-2, -1] | (1, 2]")

    assert union - float("-inf") == UnionOfIntervals.from_string("[inf, inf]")


def test_union_of_intervals_mul_negative_flips_and_merges() -> None:
    union = UnionOfIntervals.from_string("[-2, -1] | (1, 2]")

    assert union * -2.0 == UnionOfIntervals.from_string("[-4, -2) | [2, 4]")


def test_union_of_intervals_subset_relation() -> None:
    assert UnionOfIntervals.from_string("[0, 1] | [3, 4]") <= "[-1, 2] | [3, 5]"
    assert UnionOfIntervals.from_string("[0, 1] | [3, 4]") <= Interval.from_string(
        "[-1, 5]"
    )


def test_union_of_intervals_subset_relation_detects_gaps() -> None:
    assert not UnionOfIntervals.from_string("[0, 2]") <= "[0, 1] | [1.5, 2]"


def test_union_of_intervals_rejects_unrelated_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = UnionOfIntervals.from_string("[0, 1]") <= 1
