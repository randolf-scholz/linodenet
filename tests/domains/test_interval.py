from math import nan

import pytest

from linodenet.domains import Interval, RealDomain


class TestInterval:
    def test_init(self) -> None:
        expected = Interval("[0, 1)")

        assert Interval("[0, 1)") == expected
        assert Interval(Interval("[0, 1)")) == expected
        assert Interval("(NAN, NAN)") is Interval.EMPTY
        assert Interval("[NAN, 1]") is Interval.EMPTY
        assert (
            Interval(nan, 1.0, lower_inclusive=True, upper_inclusive=True)
            is Interval.EMPTY
        )
        assert Interval(Interval.EMPTY) is Interval.EMPTY

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
        assert Interval("[0, 1]") <= RealDomain("[-2, -1]", "[0, 2]")
        assert "[0, 1]" <= Interval("[-1, 2]")
        assert Interval("[0, 1]") == "[0, 1]"
        assert "[0, 1]" == Interval("[0, 1]")

        assert not Interval("[0, 1]") <= Interval("(0, 1]")
        assert not Interval("[0, 1]") <= Interval("[0, 1)")

        with pytest.raises(TypeError):
            _ = Interval("[0, 1]") <= 1

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[0, 1]", "[1, 2]", "[0, 2]"),
            ("(0, 1]", "[1, 2)", "(0, 2)"),
            ("[0, 2]", "[1, 3]", "[0, 3]"),
            ("[0, 2)", "(1, 3]", "[0, 3]"),
        ],
    )
    def test_union_operator_overlapping_cases(
        self,
        left: str,
        right: str,
        expected: str,
    ) -> None:
        assert Interval(left) | Interval(right) == Interval(expected)
        assert Interval(right) | Interval(left) == Interval(expected)

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[0, 1]", "(1, 2]", "[0, 1] | (1, 2]"),
            ("[0, 1]", "[2, 3]", "[0, 1] | [2, 3]"),
            ("[-2, -1]", "[0, 1]", "[-2, -1] | [0, 1]"),
            ("[0, 1]", "[-2, -1] | (2, 3]", "[-2, -1] | [0, 1] | (2, 3]"),
        ],
    )
    def test_union_operator_disjoint_cases(
        self,
        left: str,
        right: str,
        expected: str,
    ) -> None:
        assert Interval(left) | right == RealDomain(expected)
        assert right | Interval(left) == RealDomain(expected)

    def test_union_operator_rejects_non_intervals(self) -> None:
        with pytest.raises(TypeError):
            _ = Interval("[0, 1]") | 1

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[0, 2]", "[1, 3]", "[1, 2]"),
            ("[0, 2]", "(1, 3]", "(1, 2]"),
            ("[0, 2]", "[1, 3)", "[1, 2]"),
            ("[0, 2]", "(1, 3)", "(1, 2]"),
            ("[0, 2)", "[1, 3]", "[1, 2)"),
            ("[0, 2)", "(1, 3]", "(1, 2)"),
            ("[0, 2)", "[1, 3)", "[1, 2)"),
            ("[0, 2)", "(1, 3)", "(1, 2)"),
            ("(0, 2]", "[1, 3]", "[1, 2]"),
            ("(0, 2]", "(1, 3]", "(1, 2]"),
            ("(0, 2]", "[1, 3)", "[1, 2]"),
            ("(0, 2]", "(1, 3)", "(1, 2]"),
            ("(0, 2)", "[1, 3]", "[1, 2)"),
            ("(0, 2)", "(1, 3]", "(1, 2)"),
            ("(0, 2)", "[1, 3)", "[1, 2)"),
            ("(0, 2)", "(1, 3)", "(1, 2)"),
        ],
    )
    def test_intersection_operator_overlapping_cases(
        self,
        left: str,
        right: str,
        expected: str,
    ) -> None:
        assert Interval(left) & Interval(right) == Interval(expected)
        assert Interval(right) & Interval(left) == Interval(expected)

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[0, 1]", "[1, 2]", "[1, 1]"),
            ("[0, 1)", "[1, 2]", "(NAN, NAN)"),
            ("[0, 1]", "(1, 2]", "(NAN, NAN)"),
            ("[0, 1)", "(1, 2]", "(NAN, NAN)"),
            ("[0, 1]", "[2, 3]", "(NAN, NAN)"),
            ("(0, 1)", "[-2, 0)", "(NAN, NAN)"),
        ],
    )
    def test_intersection_operator_boundary_and_disjoint_cases(
        self,
        left: str,
        right: str,
        expected: str,
    ) -> None:
        assert Interval(left) & Interval(right) == Interval(expected)
        assert Interval(right) & Interval(left) == Interval(expected)

    def test_intersection_operator_rejects_non_intervals(self) -> None:
        with pytest.raises(TypeError):
            _ = Interval("[0, 1]") & 1

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[0, 2]", "[1, 3]", False),
            ("[0, 1]", "[1, 2]", False),
            ("[0, 1)", "[1, 2]", True),
            ("[0, 1]", "(1, 2]", True),
            ("[0, 1]", "[2, 3]", True),
            ("(0, 1)", "[-2, 0)", True),
            ("(NAN, NAN)", "[0, 1]", True),
        ],
    )
    def test_isdisjoint(
        self,
        left: str,
        right: str,
        expected: bool,
    ) -> None:
        assert Interval(left).is_disjoint(Interval(right)) is expected
        assert Interval(right).is_disjoint(Interval(left)) is expected

    @pytest.mark.parametrize(
        ("interval", "expected"),
        [
            ("[0, 1]", False),
            ("(NAN, NAN)", True),
            ("[NAN, 1]", True),
            ("[0, NAN]", True),
        ],
    )
    def test_isempty(self, interval: str, expected: bool) -> None:
        assert Interval(interval).is_empty() is expected


class TestRealDomain:
    def test_init(self) -> None:
        union = RealDomain(Interval("[-2, -1]"), "(1, 2]")

        assert union == RealDomain("[-2, -1]", "(1, 2]")
        assert RealDomain("(-inf, 0) | (0, +inf)") == RealDomain(
            "(-inf, 0)",
            "(0, +inf)",
        )

    def test_arithmetic(self) -> None:
        union = RealDomain("[-2, -1]", "(1, 2]")

        assert union + 2.0 == RealDomain("[0, 1]", "(3, 4]")
        assert union - float("-inf") == RealDomain("[inf, inf]")
        assert union * -2.0 == RealDomain("[-4, -2)", "[2, 4]")

    def test_subset_relations(self) -> None:
        assert RealDomain("[0, 1]", "[3, 4]") <= "[-1, 2] | [3, 5]"
        assert RealDomain("[0, 1]", "[3, 4]") <= Interval("[-1, 5]")
        assert not RealDomain("[0, 2]") <= "[0, 1] | [1.5, 2]"

        with pytest.raises(TypeError):
            _ = RealDomain("[0, 1]") <= 1

    def test_union_operator(self) -> None:
        union = RealDomain("[-2, -1]", "(1, 2]")

        assert union | Interval("[-1, 1]") == RealDomain("[-2, 2]")
        assert union | "[3, 4]" == RealDomain("[-2, -1]", "(1, 2]", "[3, 4]")
        assert "[-4, -3]" | union == RealDomain("[-4, -3]", "[-2, -1]", "(1, 2]")

        with pytest.raises(TypeError):
            _ = union | 1

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("[-2, 1] | [3, 5]", "[0, 4]", "[0, 1] | [3, 4]"),
            ("[-2, -1] | (1, 2]", "[-1, 1]", "[-1, -1]"),
            ("[-2, -1] | (1, 2]", "[3, 4]", "(NAN, NAN)"),
            ("[-2, -1] | (1, 2]", "(1.5, 3]", "(1.5, 2]"),
        ],
    )
    def test_intersection_operator(
        self,
        left: str,
        right: str,
        expected: str,
    ) -> None:
        assert RealDomain(left) & right == RealDomain(expected)
        assert right & RealDomain(left) == RealDomain(expected)

    def test_intersection_operator_rejects_non_domains(self) -> None:
        with pytest.raises(TypeError):
            _ = RealDomain("[0, 1]") & 1
