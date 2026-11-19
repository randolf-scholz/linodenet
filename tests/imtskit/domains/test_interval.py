from math import nan

import pytest
import torch

from imtskit.domains import (
    Interval,
    IntervalUnion,
    RealDomain,
    RealDomains,
    ScalarDomain,
)


class TestInterval:
    def test_scalar_domain_inheritance(self) -> None:
        assert isinstance(Interval("[0, 1]"), ScalarDomain)
        assert isinstance(IntervalUnion("(0,1)"), ScalarDomain)

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
            Interval("[0, 1)", 1.0)  # type: ignore

    def test_init_logs_debug_on_invalid_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level("DEBUG"),
            pytest.raises(ValueError, match=r"Invalid interval string: \[0, 1"),
        ):
            Interval("[0, 1")

        assert "Failed to parse interval string" in caplog.text

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
            ("[0, 1]", "(1, 2]", "[0, 2]"),
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
        assert Interval(left) | right == IntervalUnion(expected)
        assert right | Interval(left) == IntervalUnion(expected)

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
            _ = Interval("[0, 1]") & 1  # type: ignore

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

    def test_parse(self) -> None:
        assert Interval.parse("[0, 1]") == Interval("[0, 1]")
        assert Interval.parse("[∞, +∞]") == Interval("[inf, inf]")
        assert Interval.parse("[-∞, ∞]") == Interval("[-inf, inf]")
        assert Interval.parse("[0, 1") is None

    def test_parse_logs_debug_on_invalid_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("DEBUG"):
            assert Interval.parse("[0, 1") is None

        assert "Failed to parse interval string" in caplog.text


class TestRealDomain:
    @pytest.mark.parametrize(
        "domain",
        [Interval("[0, 1]"), IntervalUnion("[0, 1]"), RealDomains.UNIT_INTERVAL],
    )
    def test_conversion(self, domain: RealDomain) -> None:
        assert domain.intervals == (Interval("[0, 1]"),)
        union = IntervalUnion(domain)
        assert union == IntervalUnion("[0, 1]")
        assert union.intervals == domain.intervals
        assert IntervalUnion.parse(union) is union
        assert IntervalUnion.parse(domain) == union
        assert not domain.is_empty()

    @pytest.mark.parametrize(
        "left",
        [Interval("[0, 1]"), IntervalUnion("[0, 1]"), RealDomains.UNIT_INTERVAL],
    )
    @pytest.mark.parametrize(
        "right",
        [Interval("[0, 1]"), IntervalUnion("[0, 1]"), RealDomains.UNIT_INTERVAL],
    )
    def test_equivalent_representations(
        self, left: RealDomain, right: RealDomain
    ) -> None:
        assert left <= right
        assert left >= right
        assert not left < right
        assert not left > right

    @pytest.mark.parametrize(
        "unit",
        [Interval("[0, 1]"), IntervalUnion("[0, 1]"), RealDomains.UNIT_INTERVAL],
    )
    @pytest.mark.parametrize(
        "nonzero",
        [IntervalUnion("(-inf, 0) | (0, inf)"), RealDomains.NONZERO],
    )
    def test_mixed_operations(self, unit: RealDomain, nonzero: RealDomain) -> None:
        assert unit & nonzero == IntervalUnion("(0, 1]")
        assert nonzero & unit == IntervalUnion("(0, 1]")
        assert unit | nonzero == IntervalUnion("(-inf, inf)")
        assert nonzero | unit == IntervalUnion("(-inf, inf)")
        assert not unit <= nonzero
        assert not unit >= nonzero
        assert not nonzero <= unit
        assert not nonzero >= unit
        assert IntervalUnion(unit, nonzero) == IntervalUnion("(-inf, inf)")

    @pytest.mark.parametrize(
        "empty",
        [
            Interval.EMPTY,
            IntervalUnion(),
            IntervalUnion(Interval.EMPTY),
            RealDomains.EMPTY,
        ],
    )
    @pytest.mark.parametrize(
        "domain",
        [Interval("[0, 1]"), IntervalUnion("[0, 1]"), RealDomains.UNIT_INTERVAL],
    )
    def test_empty_operations(self, empty: RealDomain, domain: RealDomain) -> None:
        assert empty.is_empty()
        assert empty.intervals == ()
        assert IntervalUnion(empty).intervals == ()
        assert empty <= empty
        assert not empty < empty
        assert empty < domain
        assert domain > empty
        assert not domain <= empty
        assert (empty & domain).is_empty()
        assert (domain & empty).is_empty()
        assert empty | domain == IntervalUnion("[0, 1]")
        assert domain | empty == IntervalUnion("[0, 1]")

    @pytest.mark.parametrize(
        "empty",
        [
            Interval.EMPTY,
            IntervalUnion(),
            IntervalUnion(Interval.EMPTY),
            RealDomains.EMPTY,
        ],
    )
    def test_empty_membership(self, empty: RealDomain) -> None:
        values = torch.tensor([[float("-inf"), 0.0], [float("inf"), float("nan")]])
        result = empty.check(values)
        assert result.shape == values.shape
        assert result.dtype == torch.bool
        assert not result.any()

    def test_empty_union(self) -> None:
        empty = IntervalUnion()
        assert empty.intervals == ()
        assert len(empty) == 0
        assert tuple(empty) == ()
        assert empty[:].is_empty()
        assert IntervalUnion("[0, 1]")[:0].is_empty()
        assert (empty & empty).intervals == ()
        assert (empty | empty).intervals == ()
        assert (-empty).intervals == ()
        assert (empty + 1.0).intervals == ()
        assert (empty * 0.0).intervals == ()
        assert empty.is_disjoint(RealDomains.ANY)
        assert str(empty) == str(Interval.EMPTY)

    def test_enum_reflected_operations(self) -> None:
        assert "[0, 2]" & RealDomains.UNIT_INTERVAL == IntervalUnion("[0, 1]")
        assert "[1, 2]" | RealDomains.UNIT_INTERVAL == IntervalUnion("[0, 2]")
        assert RealDomains.UNIT_INTERVAL < "[0, 2]"
        assert RealDomains.UNIT_INTERVAL > "(0, 1)"
        assert RealDomains.Interval("(-1, ∞)") == Interval("(-1, inf)")
        assert "Interval" not in RealDomains.__members__

    @pytest.mark.parametrize(
        "method",
        [
            "__le__",
            "__lt__",
            "__ge__",
            "__gt__",
            "__and__",
            "__rand__",
            "__or__",
            "__ror__",
        ],
    )
    def test_enum_unsupported_operand(self, method: str) -> None:
        assert getattr(RealDomains.UNIT_INTERVAL, method)(1) is NotImplemented

    def test_init(self) -> None:
        union = IntervalUnion(Interval("[-2, -1]"), "(1, 2]")

        assert union == IntervalUnion("[-2, -1]", "(1, 2]")
        assert IntervalUnion("(-inf, 0) | (0, +inf)") == IntervalUnion(
            "(-inf, 0)",
            "(0, +inf)",
        )
        assert IntervalUnion("[-2, -1] | (1, 2]") == IntervalUnion(
            "[-2, -1]",
            "(1, 2]",
        )

    def test_parse(self) -> None:
        assert IntervalUnion.parse("[-2, -1] | (1, 2]") == IntervalUnion(
            "[-2, -1]",
            "(1, 2]",
        )
        assert IntervalUnion.parse("[0, 1] | ") is None

    def test_arithmetic(self) -> None:
        union = IntervalUnion("[-2, -1]", "(1, 2]")

        assert union + 2.0 == IntervalUnion("[0, 1]", "(3, 4]")
        assert union - float("-inf") == IntervalUnion("[inf, inf]")
        assert union * -2.0 == IntervalUnion("[-4, -2)", "[2, 4]")

    def test_unary_operators(self) -> None:
        union = IntervalUnion("[-2, -1]", "(1, 2]")

        assert +union is union
        assert -union == IntervalUnion("[-2, -1)", "[1, 2]")

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
        assert IntervalUnion(left) & right == IntervalUnion(expected)
        assert right & IntervalUnion(left) == IntervalUnion(expected)

    def test_intersection_operator_rejects_non_domains(self) -> None:
        with pytest.raises(TypeError):
            _ = IntervalUnion("[0, 1]") & 1
