r"""Test Signature utility."""

from collections.abc import Callable

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
        ("() -> ()", "{() -> ()}"),
        ("(n,) -> (m,)", "{(n) -> (m)}"),
        ("(...,) -> (..., n,)", "{(..., ) -> (..., n)}"),
        ("(*rest, $n, 3) -> ()", "{(*rest, $n, 3) -> ()}"),
        ("(2n-$m+3) -> (-2)", "{(2n - $m + 3) -> (-2)}"),
        ("[] -> []", "{[] -> []}"),
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
    r"""Affine dims should print with spaced ``+`` and ``-`` operators."""
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


@pytest.mark.parametrize(
    "argument",
    [
        "",
        "A",
        "A ->",
        "{A -> B",
        "[A",
        "(n,",
        "(*",
        "($",
        "(2$",
        "Tensor[",
    ],
)
def test_reject_truncated_signature(argument: str) -> None:
    r"""Incomplete productions should raise syntax errors rather than index errors."""
    with pytest.raises(RuntimeError) as exc_info:
        Parser.parse_signature(argument)
    assert isinstance(exc_info.value.__cause__, SyntaxError)


@pytest.mark.parametrize(
    ("parse", "argument", "error_type"),
    [
        (Parser.parse_signature, "A -> B C", RuntimeError),
        (Parser.parse_arglist, "[A] B", SyntaxError),
        (Parser.parse_identifier, "A B", SyntaxError),
    ],
)
def test_reject_trailing_tokens(
    parse: Callable[[str], object], argument: str, *, error_type: type[Exception]
) -> None:
    r"""Every public entry point should require complete input consumption."""
    with pytest.raises(error_type) as exc_info:
        parse(argument)
    error = exc_info.value.__cause__ or exc_info.value
    assert "expected end of input" in str(error)


@pytest.mark.parametrize(
    "argument",
    [
        "[Tensor[(3, ...)], Tensor[(n, m)], Flag]",
        "[Foo[Bar, Baz], (1, 2, n, *rest, **kwrest), ...]",
        "[T1, T2, Tensor[(1, 2, 3)], Tensor[(n, ...)], Tensor[(**k, *x, d)]]",
        "[A,]",
        "[A",
        "[Tensor[(n,)]",
    ],
)
def test_reject_invalid_arglist(argument: str) -> None:
    r"""Malformed argument lists should fail, including misplaced ellipses."""
    with pytest.raises(SyntaxError):
        Parser.parse_arglist(argument)


@pytest.mark.parametrize("argument", ["A", "Tensor", "axis_1"])
def test_parse_identifier(argument: str) -> None:
    r"""Identifier parsing should return the complete identifier."""
    assert Parser.parse_identifier(argument) == Identifier(argument)


@pytest.mark.parametrize("argument", ["", "[A]", "A?"])
def test_reject_invalid_identifier(argument: str) -> None:
    r"""Identifier parsing should reject empty input and extra syntax."""
    with pytest.raises(SyntaxError):
        Parser.parse_identifier(argument)


@pytest.mark.parametrize("argument", ["A -> @", ") @"])
def test_signature_lexical_error_wrapping(argument: str) -> None:
    r"""Later lexical errors should be wrapped and precede grammar errors."""
    with pytest.raises(RuntimeError) as exc_info:
        Parser.parse_signature(argument)
    error = exc_info.value
    assert isinstance(error.__cause__, SyntaxError)
    assert "Unexpected character '@'" in str(error.__cause__)
    assert error.__notes__ == [str(error.__cause__)]


def test_signature_initial_lexical_error() -> None:
    r"""Failure to read the first token should remain an unwrapped syntax error."""
    with pytest.raises(SyntaxError, match="Unexpected character '@' at position 0"):
        Parser.parse_signature("@ -> A")


@pytest.mark.parametrize(
    ("parse", "argument", "context"),
    [
        (Parser.parse_arglist, "[A] @", "arglist"),
        (Parser.parse_identifier, "A @", "identifier"),
    ],
)
def test_lexical_error_notes(
    parse: Callable[[str], object], argument: str, *, context: str
) -> None:
    r"""Eager tokenization errors should retain the public entry point's context."""
    with pytest.raises(SyntaxError, match="Unexpected character '@'") as exc_info:
        parse(argument)
    assert exc_info.value.__notes__ == [f"Failed to parse {context} from {argument!r}"]


def test_signature_many_dimensions() -> None:
    r"""All dimensions should be consumed without recursive calls per dimension."""
    shape = f"({', '.join(f'n{i}' for i in range(1000))})"
    assert str(Parser.parse_signature(f"{shape} -> ()")) == f"{{{shape} -> ()}}"
