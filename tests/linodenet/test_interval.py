import pytest

from linodenet.domains import Interval, UnionOfIntervals


def test_interval_init_accepts_string() -> None:
    assert Interval("[0, 1)") == Interval("[0, 1)")


def test_interval_init_accepts_interval() -> None:
    assert Interval(Interval("[0, 1)")) == Interval("[0, 1)")


def test_interval_init_rejects_mixed_string_and_bounds() -> None:
    with pytest.raises(TypeError):
        Interval("[0, 1)", 1.0)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]


def test_interval_add_shifts_bounds() -> None:
    interval = Interval("(0, 1]")

    assert interval + 2.0 == Interval("(2, 3]")


def test_interval_add_positive_infinity_collapses_upward() -> None:
    interval = Interval("(0, 1]")

    assert interval + float("inf") == Interval("(inf, inf]")


def test_interval_sub_shifts_bounds() -> None:
    interval = Interval("(0, 1]")

    assert interval - 2.0 == Interval("(-2, -1]")


def test_interval_sub_positive_infinity_collapses_downward() -> None:
    interval = Interval("(0, 1]")

    assert interval - float("inf") == Interval("(-inf, -inf]")


def test_interval_mul_positive_scales_bounds() -> None:
    interval = Interval("(1, 2]")

    assert interval * 3.0 == Interval("(3, 6]")


def test_interval_mul_positive_infinity() -> None:
    interval = Interval("(1, 2]")

    assert interval * float("inf") == Interval("(inf, inf]")


def test_interval_mul_negative_flips_bounds_and_inclusivity() -> None:
    interval = Interval("(1, 2]")

    assert interval * -2.0 == Interval("[-4, -2)")


def test_interval_mul_negative_infinity_flips_bounds_and_inclusivity() -> None:
    interval = Interval("(1, 2]")

    assert interval * float("-inf") == Interval("[-inf, -inf)")


def test_interval_mul_zero_collapses_to_closed_origin() -> None:
    interval = Interval("(1, 2]")

    assert interval * 0.0 == Interval("[0, 0]")


def test_interval_subset_relation() -> None:
    assert Interval("(0, 1)") <= Interval("[0, 1]")
    assert Interval("[0, 1]") <= Interval("[-1, 2]")


def test_interval_subset_relation_respects_open_and_closed_bounds() -> None:
    assert not Interval("[0, 1]") <= Interval("(0, 1]")
    assert not Interval("[0, 1]") <= Interval("[0, 1)")


def test_interval_subset_relation_accepts_string_on_right() -> None:
    assert Interval("[0, 1]") <= "[-1, 2]"


def test_interval_subset_relation_accepts_union_on_right() -> None:
    assert Interval("[0, 1]") <= UnionOfIntervals("[-2, -1]", "[0, 2]")


def test_interval_subset_relation_accepts_string_on_left() -> None:
    assert "[0, 1]" <= Interval("[-1, 2]")


def test_interval_equality_accepts_string_on_right() -> None:
    assert Interval("[0, 1]") == "[0, 1]"


def test_interval_equality_accepts_string_on_left() -> None:
    assert "[0, 1]" == Interval("[0, 1]")


def test_interval_rejects_unrelated_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = Interval("[0, 1]") <= 1


def test_union_of_intervals_add_applies_to_each_member() -> None:
    union = UnionOfIntervals("[-2, -1]", "(1, 2]")

    assert union + 2.0 == UnionOfIntervals("[0, 1]", "(3, 4]")


def test_union_of_intervals_init_accepts_mixed_interval_and_string_inputs() -> None:
    union = UnionOfIntervals(Interval("[-2, -1]"), "(1, 2]")

    assert union == UnionOfIntervals("[-2, -1]", "(1, 2]")


def test_union_of_intervals_sub_negative_infinity() -> None:
    union = UnionOfIntervals("[-2, -1]", "(1, 2]")

    assert union - float("-inf") == UnionOfIntervals("[inf, inf]")


def test_union_of_intervals_mul_negative_flips_and_merges() -> None:
    union = UnionOfIntervals("[-2, -1]", "(1, 2]")

    assert union * -2.0 == UnionOfIntervals("[-4, -2)", "[2, 4]")


def test_union_of_intervals_subset_relation() -> None:
    assert UnionOfIntervals("[0, 1]", "[3, 4]") <= "[-1, 2] | [3, 5]"
    assert UnionOfIntervals("[0, 1]", "[3, 4]") <= Interval("[-1, 5]")


def test_union_of_intervals_subset_relation_detects_gaps() -> None:
    assert not UnionOfIntervals("[0, 2]") <= "[0, 1] | [1.5, 2]"


def test_union_of_intervals_rejects_unrelated_type_ordering() -> None:
    with pytest.raises(TypeError):
        _ = UnionOfIntervals("[0, 1]") <= 1
