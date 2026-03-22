import pytest

from linodenet.domains import Interval, RealDomain


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
        assert Interval("[0, 1]") <= RealDomain("[-2, -1]", "[0, 2]")
        assert "[0, 1]" <= Interval("[-1, 2]")
        assert Interval("[0, 1]") == "[0, 1]"
        assert "[0, 1]" == Interval("[0, 1]")

        assert not Interval("[0, 1]") <= Interval("(0, 1]")
        assert not Interval("[0, 1]") <= Interval("[0, 1)")

        with pytest.raises(TypeError):
            _ = Interval("[0, 1]") <= 1

    def test_union_operator(self) -> None:
        interval = Interval("[0, 1]")

        assert interval | Interval("(1, 2]") == RealDomain("[0, 2]")
        assert interval | "[-2, -1] | (2, 3]" == RealDomain(
            "[-2, -1]",
            "[0, 1]",
            "(2, 3]",
        )
        assert "[-2, -1]" | interval == RealDomain("[-2, -1]", "[0, 1]")

        with pytest.raises(TypeError):
            _ = interval | 1

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
        assert Interval(left).isdisjoint(Interval(right)) is expected
        assert Interval(right).isdisjoint(Interval(left)) is expected


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
