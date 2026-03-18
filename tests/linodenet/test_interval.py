import pytest

from linodenet.domains import Interval, IntervalUnion


class TestInterval:
    def test_init(self) -> None:
        expected = Interval("[0, 1)")

        assert Interval("[0, 1)") == expected
        assert Interval(Interval("[0, 1)")) == expected

        with pytest.raises(TypeError):
            Interval("[0, 1)", 1.0)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]

    def test_arithmetic(self) -> None:
        shifted = Interval("(0, 1]")
        scaled = Interval("(1, 2]")

        assert shifted + 2.0 == Interval("(2, 3]")
        assert shifted + float("inf") == Interval("(inf, inf]")
        assert shifted - 2.0 == Interval("(-2, -1]")
        assert shifted - float("inf") == Interval("(-inf, -inf]")
        assert scaled * 3.0 == Interval("(3, 6]")
        assert scaled * float("inf") == Interval("(inf, inf]")
        assert scaled * -2.0 == Interval("[-4, -2)")
        assert scaled * float("-inf") == Interval("[-inf, -inf)")
        assert scaled * 0.0 == Interval("[0, 0]")

    def test_comparisons(self) -> None:
        assert Interval("(0, 1)") <= Interval("[0, 1]")
        assert Interval("[0, 1]") <= Interval("[-1, 2]")
        assert Interval("[0, 1]") <= "[-1, 2]"
        assert Interval("[0, 1]") <= IntervalUnion("[-2, -1]", "[0, 2]")
        assert "[0, 1]" <= Interval("[-1, 2]")
        assert Interval("[0, 1]") == "[0, 1]"
        assert "[0, 1]" == Interval("[0, 1]")

        assert not Interval("[0, 1]") <= Interval("(0, 1]")
        assert not Interval("[0, 1]") <= Interval("[0, 1)")

        with pytest.raises(TypeError):
            _ = Interval("[0, 1]") <= 1

    def test_union_operator(self) -> None:
        interval = Interval("[0, 1]")

        assert interval | Interval("(1, 2]") == IntervalUnion("[0, 2]")
        assert interval | "[-2, -1] | (2, 3]" == IntervalUnion(
            "[-2, -1]",
            "[0, 1]",
            "(2, 3]",
        )
        assert "[-2, -1]" | interval == IntervalUnion("[-2, -1]", "[0, 1]")

        with pytest.raises(TypeError):
            _ = interval | 1


class TestIntervalUnion:
    def test_init(self) -> None:
        union = IntervalUnion(Interval("[-2, -1]"), "(1, 2]")

        assert union == IntervalUnion("[-2, -1]", "(1, 2]")
        assert IntervalUnion("(-inf, 0) | (0, +inf)") == IntervalUnion(
            "(-inf, 0)",
            "(0, +inf)",
        )

    def test_arithmetic(self) -> None:
        union = IntervalUnion("[-2, -1]", "(1, 2]")

        assert union + 2.0 == IntervalUnion("[0, 1]", "(3, 4]")
        assert union - float("-inf") == IntervalUnion("[inf, inf]")
        assert union * -2.0 == IntervalUnion("[-4, -2)", "[2, 4]")

    def test_subset_relations(self) -> None:
        assert IntervalUnion("[0, 1]", "[3, 4]") <= "[-1, 2] | [3, 5]"
        assert IntervalUnion("[0, 1]", "[3, 4]") <= Interval("[-1, 5]")
        assert not IntervalUnion("[0, 2]") <= "[0, 1] | [1.5, 2]"

        with pytest.raises(TypeError):
            _ = IntervalUnion("[0, 1]") <= 1

    def test_union_operator(self) -> None:
        union = IntervalUnion("[-2, -1]", "(1, 2]")

        assert union | Interval("[-1, 1]") == IntervalUnion("[-2, 2]")
        assert union | "[3, 4]" == IntervalUnion("[-2, -1]", "(1, 2]", "[3, 4]")
        assert "[-4, -3]" | union == IntervalUnion("[-4, -3]", "[-2, -1]", "(1, 2]")

        with pytest.raises(TypeError):
            _ = union | 1
