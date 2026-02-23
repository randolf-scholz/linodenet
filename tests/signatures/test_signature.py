r"""Test Signature utility."""

import pytest

from signatures import Parser


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("(m, n) -> (n, m)", "(m, n) -> (n, m)"),
    ],
)
def test_signature(argument: str, expected: str) -> None:
    r"""Test Signature utility."""
    sig = Parser.parse_signature(argument)
    assert str(sig) == expected


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
