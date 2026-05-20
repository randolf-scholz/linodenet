r"""Test Signature utility."""

import pytest

from signatures import GenericType, Identifier, Parser, signature


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("(m, n) -> (n, m)", "{(m, n) -> (n, m)}"),
        ("{(m, n) -> (n, m)}", "{(m, n) -> (n, m)}"),
        ("{A -> B} -> C", "{{A -> B} -> C}"),
        ("A -> {B -> C}", "{A -> {B -> C}}"),
        ("{A -> B} -> {C -> D}", "{{A -> B} -> {C -> D}}"),
        (
            "{(..., n, d) -> (...)} -> (..., n, d)",
            "{{(..., n, d) -> (..., )} -> (..., n, d)}",
        ),
        ("[(m, n), {(n) -> (m)}] -> (m, n)", "{[(m, n), {(n) -> (m)}] -> (m, n)}"),
        (
            "[Tensor[(m, n)]?, Label?] -> Output?",
            "{[Tensor[(m, n)]?, Label?] -> Output?}",
        ),
    ],
)
def test_signature(argument: str, expected: str) -> None:
    r"""Test Signature utility."""
    sig = Parser.parse_signature(argument)
    assert str(sig) == expected


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("(n) -> (n-1)", "{(n) -> (n - 1)}"),
        ("(n) -> (n+1)", "{(n) -> (n + 1)}"),
        ("(2n-1) -> (n)", "{(2n - 1) -> (n)}"),
        ("(n-1, m+2) -> (m-3)", "{(n - 1, m + 2) -> (m - 3)}"),
    ],
)
def test_signature_affine_dims(argument: str, expected: str) -> None:
    r"""Affine dims should preserve signed offsets when parsed."""
    sig = Parser.parse_signature(argument)
    assert str(sig) == expected


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("(n) -> (n-1)", "{(n) -> (n - 1)}"),
        ("(n) -> (n+1)", "{(n) -> (n + 1)}"),
    ],
)
def test_signature_affine_dim_repr(argument: str, expected: str) -> None:
    r"""Affine dims should print with spaced `+` and `-` operators."""
    sig = signature(argument)
    assert str(sig) == expected
    assert repr(sig) == expected


def test_signature_singleton_shape_output() -> None:
    r"""Singleton output shapes should not gain a synthetic dimension."""
    sig = Parser.parse_signature("(m, n) -> [(m, k), (k,), (n, k)]")
    assert str(sig) == "{(m, n) -> [(m, k), (k), (n, k)]}"


@pytest.mark.parametrize(
    "argument",
    [
        "A -> B -> C",
        "{A -> B -> C}",
        "(?) -> ()",
    ],
)
def test_reject_chained_signature(argument: str) -> None:
    r"""Invalid signatures should be rejected."""
    with pytest.raises(RuntimeError):
        Parser.parse_signature(argument)


def test_parse_optional_args() -> None:
    r"""Optional args should round-trip through the parser."""
    arglist = Parser.parse_arglist("[Tensor[(m, n)]?, Label?]")

    assert arglist[0].optional is True
    assert isinstance(arglist[0].value, GenericType)
    assert arglist[0].value.id == Identifier("Tensor")
    assert arglist[1].optional is True
    assert arglist[1].value == Identifier("Label")


@pytest.mark.parametrize(
    ("argument", "expected_optional"),
    [
        ("[Label]", [False]),
        ("[Label?]", [True]),
        ("[Tensor[(m, n)]?, Label?]", [True, True]),
    ],
)
def test_parse_arg_optional_flags(argument: str, expected_optional: list[bool]) -> None:
    r"""Optional markers should populate `Arg.optional`."""
    arglist = Parser.parse_arglist(argument)
    assert [arg.optional for arg in arglist] == expected_optional


@pytest.mark.parametrize(
    "argument",
    [
        "[Label?, Tensor[(m, n)]]",
        "[Label?, Output]",
    ],
)
def test_reject_required_arg_after_optional_arg(argument: str) -> None:
    r"""Required args cannot follow optional args in an arglist."""
    with pytest.raises(SyntaxError, match="cannot follow an optional argument"):
        Parser.parse_arglist(argument)


def test() -> None:
    examples = [
        "[x]",
        "[Tensor[(3, 224, 224)], Label]",
        "[Tensor[(3, ...)], Tensor[(n, m)], Flag]",
        "[Foo[Bar, Baz], (1, 2, n, *rest, **kwrest), ...]",  # last "..." is invalid ArgType -> will error
        "[T1, T2, Tensor[(1, 2, 3)], Tensor[(n, ...)], Tensor[(**k, *x, d)]]",
    ]

    for src in examples:
        print("SOURCE:", src)
        try:
            ast = Parser.parse_arglist(src)
            print("AST:   ", ast)
        except SyntaxError as e:
            print("ERROR:", e)
        print("-" * 60)

    examples_sig = [
        "[Tensor[(3, 224, 224)]] -> Tensor[(3, 224, 224)]",
        "Tensor[(..., n)] -> [Tensor[(n, m)], Flag]",
    ]

    for src in examples_sig:
        print("SIGNATURE SOURCE:", src)
        try:
            sig = Parser.parse_signature(src)
            print("SIG:   ", sig)
        except SyntaxError as e:
            print("ERROR:", e)
        print("-" * 60)
